#!/usr/bin/env python3
# 生成結果が統計的にどれだけ実在字らしいかの診断ツール。「本物(実データ)か偽物(生成結果)か」を
# 判別する小さい分類器を学習させ、その正答率を見る。交差数・角度・孤立ストロークのように人間が名指しした
# 軸だけでなく、統計的な違いがありさえすれば(人間が気づいていない軸も含めて)自動的に手がかりとして
# 拾い上げられるため、まだ見つかっていない差が残っているかどうかの目安になる。
# VAEの学習・推論(生成)には一切組み込まれない、独立した事後診断用のスクリプト
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from vae_checkpoint import Checkpoint, load_checkpoint
from vae_eval_common import attract_to_latent_prior, existence_mask_from_logits, load_train_data, load_validation_data
from vae_model import flatten_input, select_device, unflatten_output

NEAREST_REAL_FILTER_THRESHOLD = 1.0  # これより実在字に近い生成サンプルは、ラベルの矛盾を避けるため除外する
HIDDEN_DIMS = (1024, 512)  # VAEのencoderと同じ構造で十分という判断
DECISION_THRESHOLD = 0.5  # シグモイド出力を本物/偽物に振り分ける閾値。vae_eval_common.EXISTENCE_THRESHOLDと同じ考え方
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15
SEED = 0


class _Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int]) -> None:
        super().__init__()
        hidden1, hidden2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


class _ClassifierMetrics(NamedTuple):
    accuracy: float
    auc: float
    false_positive_rate: float  # 偽物を本物と誤判定した割合
    false_negative_rate: float  # 本物を偽物と誤判定した割合


def _masked_flatten_input(strokes: torch.Tensor, existence: torch.Tensor) -> torch.Tensor:
    # 実データは存在しないスロットのストローク特徴量が0埋めされているが、decoderの出力にはそのような
    # 制約が無い(compute_lossのストローク系損失はexistenceでマスクした上でしか評価していないため)。
    # 素のflatten_inputをそのまま使うと、分類器が「漢字らしさ」ではなく「存在しないスロットの値が
    # 実データの決まったパディング値と一致しているか」というデータ表現上の抜け道を学習してしまうため、
    # 両方とも同じようにexistenceでマスクしてから入力する
    masked_strokes = strokes * existence.unsqueeze(-1)
    return torch.cat([masked_strokes.flatten(1), existence], dim=1)


def _sample_candidates(checkpoint: Checkpoint, mu_real: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        existence_pred = torch.from_numpy(existence_mask_from_logits(existence_logits)).float().to(mu_real.device)
    return strokes_recon, existence_pred, nearest_dist


def _generate_fake_examples(checkpoint: Checkpoint, mu_real: torch.Tensor, count: int) -> torch.Tensor:
    strokes_recon, existence_pred, nearest_dist = _sample_candidates(checkpoint, mu_real, count)

    # ラベルの矛盾(ほぼ同じ入力なのに本物・偽物の両方に現れる)を避けるため、実在字に極端に近いサンプルは除外する
    keep = nearest_dist >= NEAREST_REAL_FILTER_THRESHOLD
    filtered_count = int(keep.sum().item())
    print(f"Generated {len(nearest_dist)} samples, filtered out {len(nearest_dist) - filtered_count} near-duplicate(s)")
    if filtered_count < count:
        raise RuntimeError(f"Not enough fake examples after filtering: {filtered_count} < {count}")

    strokes_recon, existence_pred = strokes_recon[keep][:count], existence_pred[keep][:count]
    return _masked_flatten_input(strokes_recon, existence_pred)


def _combine_real_fake(real_x: torch.Tensor, fake_x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # ラベルは本物=1、偽物=0(_compute_metricsのis_real/is_fakeもこの約束事に基づく)
    x = torch.cat([real_x, fake_x], dim=0)
    y = torch.cat([torch.ones(len(real_x)), torch.zeros(len(fake_x))]).to(device)
    return x, y


def _run_epoch(loader: DataLoader, model: _Classifier, optimizer: torch.optim.Optimizer | None) -> float:
    # optimizerがNoneのとき(test時)は重み更新を行わないeval modeとして扱う。train_vae.pyと同じパターン
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


def _train_classifier(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> _Classifier:
    model = _Classifier(train_x.shape[1], HIDDEN_DIMS).to(train_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)

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


def _compute_metrics(model: _Classifier, x: torch.Tensor, y: torch.Tensor) -> _ClassifierMetrics:
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

    real_train_x = _masked_flatten_input(train_data.strokes_standardized, train_data.existence_tensor)
    real_test_x = _masked_flatten_input(val_data.strokes_standardized, val_data.existence_tensor)
    fake_train_x = _generate_fake_examples(checkpoint, mu_real, len(real_train_x))
    fake_test_x = _generate_fake_examples(checkpoint, mu_real, len(real_test_x))

    train_x, train_y = _combine_real_fake(real_train_x, fake_train_x, device)
    test_x, test_y = _combine_real_fake(real_test_x, fake_test_x, device)

    model = _train_classifier(train_x, train_y, test_x, test_y)
    metrics = _compute_metrics(model, test_x, test_y)
    print(
        f"real_train={len(real_train_x)} fake_train={len(fake_train_x)} "
        f"real_test={len(real_test_x)} fake_test={len(fake_test_x)}"
    )
    print(
        f"accuracy={metrics.accuracy:.4f} auc={metrics.auc:.5f} "
        f"false_positive_rate={metrics.false_positive_rate:.4f} false_negative_rate={metrics.false_negative_rate:.4f}"
    )


if __name__ == "__main__":
    main()
