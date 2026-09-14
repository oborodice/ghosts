#!/usr/bin/env python3
# 画像ベースの診断分類器。生の数値特徴量ではなく、実際にレンダリングした画像を入力にして
# 本物/偽物を判別する。VAEの学習・推論には一切組み込まれない、独立した事後診断用のスクリプト
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from vae_checkpoint import Checkpoint, load_checkpoint
from vae_eval_common import (
    SEGMENTS_PER_CURVE,
    SplitData,
    attract_to_latent_prior,
    bezier_polyline,
    existence_mask_from_logits,
    load_train_data,
    load_validation_data,
    strokes_to_curves,
)
from vae_model import flatten_input, select_device, unflatten_output

NEAREST_REAL_FILTER_THRESHOLD = 1.0  # これより実在字に近い生成サンプルは、ラベルの矛盾(ほぼ同じ入力なのに本物・偽物の両方に現れる)を避けるため除外する
VIEWBOX_SIZE = 109.0  # KanjiVGのSVGのviewBoxサイズ(データの座標系そのもの。training/data/kanjivg/*.svg参照)
# 診断用レンダリング解像度。フロントエンドの実装(描画バッファのサイズなど)がどうなっているかは切り離し、
# 「小さいアイコン程度の表示サイズで見てもなお判別できてしまうか」を確認するための値を、それ単体で
# 妥当かどうかで直接決める(目視で崩れず読み取れることを確認済み)
CANVAS_SIZE = 128
LINE_WIDTH = 4  # 字全体に対して自然な太さになるよう目視で選んだ値
BLUR_RADIUS = 0.5  # ラスタライズ特有のジャギー(輪郭のギザつき)を均し、サブピクセル単位の情報を分類器に渡さないための軽いぼかし
HIDDEN_CHANNELS = (16, 32)  # 「小さいCNN」で十分という判断(既存の数値特徴量分類器と同程度のパラメータ規模感)
DECISION_THRESHOLD = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15
SEED = 0


class _ImageClassifier(nn.Module):
    def __init__(self, image_size: int, hidden_channels: tuple[int, int]) -> None:
        super().__init__()
        c1, c2 = hidden_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # プーリング段数からの手計算(image_size // 4)は、self.convの構成を変えたときに追従し忘れて
        # 形状不一致を起こしやすいため、実際に1回通して畳み込み後のサイズを直接求める
        with torch.no_grad():
            conv_out_features = self.conv(torch.zeros(1, 1, image_size, image_size)).numel()
        self.head = nn.Linear(conv_out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1)).squeeze(-1)  # logits


class _ClassifierMetrics(NamedTuple):
    accuracy: float
    auc: float
    false_positive_rate: float  # 偽物を本物と誤判定した割合
    false_negative_rate: float  # 本物を偽物と誤判定した割合


def _render_curves_to_array(curves: list[tuple[complex, complex, complex]]) -> np.ndarray:
    scale = CANVAS_SIZE / VIEWBOX_SIZE
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(img)
    r = LINE_WIDTH / 2
    for start, control, end in curves:
        poly = bezier_polyline(start, control, end, SEGMENTS_PER_CURVE) * scale
        points = [tuple(p) for p in poly]
        draw.line(points, fill=255, width=LINE_WIDTH, joint="curve")
        # PILのdraw.lineは端点が四角く切れるため、筆で書いたような自然な丸みを出すために円を足す
        for p in (points[0], points[-1]):
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    if BLUR_RADIUS > 0:
        img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return np.asarray(img, dtype=np.float32) / 255.0


def _render_batch(strokes: np.ndarray, existence: np.ndarray) -> torch.Tensor:
    images = np.stack(
        [
            _render_curves_to_array(strokes_to_curves(strokes[i], existence[i].astype(bool)))
            for i in range(len(strokes))
        ]
    )
    return torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)


def _sample_candidates(
    checkpoint: Checkpoint, mu_real: torch.Tensor, count: int
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    # 後段のフィルタで減る分を見込み、countより多めにサンプリングする。実測ではフィルタで除外される
    # サンプルはごく僅か(数千件に1件程度)なため、係数・加算値自体は厳密なチューニング値ではなく
    # 余裕を持たせた値
    oversample = int(count * 1.2) + 50
    z_raw = torch.randn(oversample, checkpoint.latent_dim, device=mu_real.device)
    with torch.no_grad():
        z = attract_to_latent_prior(z_raw, mu_real)
        nearest_dist = torch.cdist(z, mu_real).min(dim=1).values
        recon = checkpoint.model.decode(z)
        strokes_recon, existence_logits = unflatten_output(recon, checkpoint.shape)
        strokes_recon = (strokes_recon * checkpoint.std + checkpoint.mean).cpu().numpy()
        existence_pred = existence_mask_from_logits(existence_logits)
    return strokes_recon, existence_pred, nearest_dist


def _filter_to_count(
    strokes_recon: np.ndarray, existence_pred: np.ndarray, nearest_dist: torch.Tensor, count: int
) -> tuple[np.ndarray, np.ndarray]:
    # 実在字に極端に近いサンプルは、ラベルの矛盾(ほぼ同じ入力なのに本物・偽物の両方に現れる)を避けるため除外する
    keep = (nearest_dist >= NEAREST_REAL_FILTER_THRESHOLD).cpu().numpy()
    filtered_count = int(keep.sum())
    print(f"Generated {len(nearest_dist)} samples, filtered out {len(nearest_dist) - filtered_count} near-duplicate(s)")
    if filtered_count < count:
        raise RuntimeError(f"Not enough fake examples after filtering: {filtered_count} < {count}")

    idx = np.where(keep)[0][:count]
    return strokes_recon[idx], existence_pred[idx]


def _build_labeled_dataset(
    checkpoint: Checkpoint, mu_real: torch.Tensor, split_data: SplitData, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    real_x = _render_batch(split_data.strokes, split_data.existence)
    strokes_recon, existence_pred, nearest_dist = _sample_candidates(checkpoint, mu_real, len(real_x))
    fake_strokes, fake_existence = _filter_to_count(strokes_recon, existence_pred, nearest_dist, len(real_x))
    fake_x = _render_batch(fake_strokes, fake_existence)

    x = torch.cat([real_x, fake_x]).to(device)
    y = torch.cat([torch.ones(len(real_x)), torch.zeros(len(fake_x))]).to(device)  # ラベルは本物=1、偽物=0
    return x, y


def _run_epoch(loader: DataLoader, model: _ImageClassifier, optimizer: torch.optim.Optimizer | None) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    with torch.set_grad_enabled(is_training):
        for x_batch, y_batch in loader:
            loss = F.binary_cross_entropy_with_logits(model(x_batch), y_batch)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(x_batch)
    return total_loss / len(loader.dataset)


def _setup_training(
    train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor
) -> tuple[_ImageClassifier, torch.optim.Optimizer, DataLoader, DataLoader]:
    model = _ImageClassifier(CANVAS_SIZE, HIDDEN_CHANNELS).to(train_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)
    return model, optimizer, train_loader, test_loader


def _train_classifier(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> _ImageClassifier:
    model, optimizer, train_loader, test_loader = _setup_training(train_x, train_y, test_x, test_y)

    best_test_loss = float("inf")
    best_state = model.state_dict()
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _run_epoch(train_loader, model, optimizer)
        test_loss = _run_epoch(test_loader, model, None)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} test_loss={test_loss:.4f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def _compute_metrics(model: _ImageClassifier, x: torch.Tensor, y: torch.Tensor) -> _ClassifierMetrics:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
        preds = (probs > DECISION_THRESHOLD).float()

    is_real, is_fake = y == 1, y == 0
    return _ClassifierMetrics(
        accuracy=(preds == y).float().mean().item(),
        auc=roc_auc_score(y.cpu().numpy(), probs.cpu().numpy()),
        false_positive_rate=(preds[is_fake] == 1).float().mean().item(),
        false_negative_rate=(preds[is_real] == 0).float().mean().item(),
    )


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    checkpoint = load_checkpoint(device)
    train_data = load_train_data(checkpoint, device)
    val_data = load_validation_data(checkpoint, device)

    with torch.no_grad():
        mu_real, _ = checkpoint.model.encode(flatten_input(train_data.strokes_standardized, train_data.existence_tensor))

    print("Building train dataset (rendering real, sampling and rendering fake)...")
    train_x, train_y = _build_labeled_dataset(checkpoint, mu_real, train_data, device)
    print("Building test dataset...")
    test_x, test_y = _build_labeled_dataset(checkpoint, mu_real, val_data, device)

    model = _train_classifier(train_x, train_y, test_x, test_y)
    metrics = _compute_metrics(model, test_x, test_y)
    real_train_count = int(train_y.sum().item())
    real_test_count = int(test_y.sum().item())
    print(
        f"real_train={real_train_count} fake_train={len(train_y) - real_train_count} "
        f"real_test={real_test_count} fake_test={len(test_y) - real_test_count}"
    )
    print(
        f"accuracy={metrics.accuracy:.4f} auc={metrics.auc:.5f} "
        f"false_positive_rate={metrics.false_positive_rate:.4f} false_negative_rate={metrics.false_negative_rate:.4f}"
    )


if __name__ == "__main__":
    main()
