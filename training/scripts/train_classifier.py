#!/usr/bin/env python3
# 生成した字の評価の物差しにする文字認識のモデル(glyph/classifier.py)を、学習データ(build_dataset.py で作る)で学習する。
# 3書風(明朝・手書き・ゴシックから1つずつ)を学習から外し、見たことのない書風でも読めるかを確かめる(物差しが書風の癖ではなく、
# 字の形を見ているか)。生成物の粗さに強くするため、学習のときに位置ずれ・拡大縮小・ぼかし・ノイズを加える。
# 学習のあと、実在字で特徴の距離の分布を測り、生成物の評価に使うしきい値と一緒に data/glyph_classifier.pt へ保存する
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from glyph.classifier import Calibration, GlyphClassifier, calibrate

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HELD_OUT_STYLES = ("Zen Old Mincho", "Yomogi", "LINE Seed JP")
SEED = 0  # 物差しを作り直しても同じになるよう、乱数を固定する
EPOCHS = 12
BATCH = 256
LEARNING_RATE = 2e-3  # 1サイクルの学習率の予定の最大値
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
EVALUATION_CHUNK = 1024
CALIBRATION_KANJI = 1000  # 距離の分布を測るのに使う漢字の数(同じ字を別の書風で描いた組を、字ごとに1組)
# 増強(生成物の粗さに強くするため)
MAX_SHIFT = 3 / 64  # 位置ずれの最大(画像の幅に対する割合。64pxで3px)
SCALE_RANGE = (0.9, 1.1)
BLUR_PROBABILITY = 0.5  # バッチごとに、[1, 2, 1]のぼかしをかける確率
NOISE_STD = 0.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "glyphs_64.npz")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "glyph_classifier.pt")
    return parser.parse_args()


def _split_held_out_styles(styles: np.ndarray, style_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # 学習用と検証用(学習から外した書風)の添字
    held_out = np.isin(styles, [style_names.index(name) for name in HELD_OUT_STYLES])
    return np.nonzero(~held_out)[0], np.nonzero(held_out)[0]


def _load_batch(images: torch.Tensor, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    # 学習データはuint8のまま持ち、使う分だけ0〜1の小数にする(全体を小数にすると、約22万枚で約3.7GBになる)
    return images[indices].to(device).float().div(255).unsqueeze(1)


def _augment(images: torch.Tensor) -> torch.Tensor:
    count = len(images)
    scale = torch.empty(count, device=images.device).uniform_(*SCALE_RANGE)
    affine = torch.zeros(count, 2, 3, device=images.device)
    affine[:, 0, 0] = affine[:, 1, 1] = 1 / scale
    # affine_grid の平行移動は、画像の半分の幅を1とする単位なので、画像の幅に対する割合の2倍にする
    affine[:, :, 2] = torch.empty(count, 2, device=images.device).uniform_(-2 * MAX_SHIFT, 2 * MAX_SHIFT)
    images = F.grid_sample(images, F.affine_grid(affine, images.shape, align_corners=False), align_corners=False)
    if torch.rand(1).item() < BLUR_PROBABILITY:
        kernel = torch.tensor([1.0, 2.0, 1.0], device=images.device)
        images = F.conv2d(images, (kernel[:, None] * kernel[None] / kernel.sum() ** 2).view(1, 1, 3, 3), padding=1)
    return (images + NOISE_STD * torch.randn_like(images)).clamp(0, 1)


def _train_epoch(model: GlyphClassifier, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                 images: torch.Tensor, labels: torch.Tensor, train_indices: np.ndarray, epoch: int, device: torch.device) -> float:
    # 返り値は、このエポックの学習データでの正答率
    model.train()
    order = np.random.default_rng(epoch).permutation(train_indices)
    correct = 0
    for start in range(0, len(order), BATCH):
        batch = order[start:start + BATCH]
        inputs, targets = _augment(_load_batch(images, batch, device)), labels[batch].to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, label_smoothing=LABEL_SMOOTHING)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        correct += (logits.argmax(1) == targets).sum().item()
    return correct / len(order)


@torch.no_grad()
def _accuracy(model: GlyphClassifier, images: torch.Tensor, labels: torch.Tensor, indices: np.ndarray, device: torch.device) -> float:
    correct = 0
    for start in range(0, len(indices), EVALUATION_CHUNK):
        chunk = indices[start:start + EVALUATION_CHUNK]
        correct += (model(_load_batch(images, chunk, device)).argmax(1).cpu() == labels[chunk]).sum().item()
    return correct / len(indices)


def _train(model: GlyphClassifier, images: torch.Tensor, labels: torch.Tensor, train_indices: np.ndarray, val_indices: np.ndarray,
           device: torch.device) -> float:
    # 学習から外した書風での正答率が最も良かったエポックの重みに戻し、その正答率を返す
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, LEARNING_RATE, total_steps=EPOCHS * (len(train_indices) // BATCH + 1))
    best_accuracy, best_state = -1.0, None
    for epoch in range(EPOCHS):
        started = time.time()
        train_accuracy = _train_epoch(model, optimizer, scheduler, images, labels, train_indices, epoch, device)
        model.eval()
        accuracy = _accuracy(model, images, labels, val_indices, device)
        print(f"epoch {epoch + 1}: train acc {train_accuracy:.3f} | held-out-style acc {accuracy:.3f} | {time.time() - started:.0f}s", flush=True)
        if accuracy > best_accuracy:
            best_accuracy, best_state = accuracy, {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return best_accuracy


def _calibration_pairs(labels: np.ndarray, num_classes: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # 字ごとに、別々の書風で描いた2枚の組(同じ字)。2枚ない字は飛ばす
    first_indices, second_indices = [], []
    for kanji_index in rng.choice(num_classes, min(CALIBRATION_KANJI, num_classes), replace=False):
        candidates = np.nonzero(labels == kanji_index)[0]
        if len(candidates) >= 2:
            first, second = rng.choice(candidates, 2, replace=False)
            first_indices.append(first)
            second_indices.append(second)
    return np.array(first_indices), np.array(second_indices)


@torch.no_grad()
def _features(model: GlyphClassifier, images: torch.Tensor, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.cat([model.features(_load_batch(images, indices[start:start + EVALUATION_CHUNK], device)) for start in range(0, len(indices), EVALUATION_CHUNK)])


def _measure_calibration(model: GlyphClassifier, images: torch.Tensor, labels: np.ndarray, num_classes: int, device: torch.device) -> Calibration:
    # 距離の分布: 同じ字の組と、組の片方を隣の組のものと入れ替えた組(別の字)
    first_indices, second_indices = _calibration_pairs(labels, num_classes, np.random.default_rng(SEED))
    first_features, second_features = _features(model, images, first_indices, device), _features(model, images, second_indices, device)
    calibration = calibrate((first_features - second_features).norm(dim=1).cpu(), (first_features - second_features.roll(1, 0)).norm(dim=1).cpu())
    if calibration.different_glyph_near_distance <= calibration.same_glyph_far_distance:
        # 別の字の距離が同じ字の距離より近いことがあると、しきい値が意味をなさない(学習が足りない物差し)
        print("WARNING: the classifier does not separate different kanji from the same kanji; the thresholds are not usable", flush=True)
    return calibration


def main() -> None:
    args = _parse_args()
    torch.manual_seed(SEED)
    device = torch.accelerator.current_accelerator() or torch.device("cpu")

    data = np.load(args.data)
    images, labels = torch.from_numpy(data["images"]), torch.from_numpy(data["labels"]).long()
    num_classes = len(data["kanji"])
    train_indices, val_indices = _split_held_out_styles(data["styles"], list(data["style_names"]))
    print(f"device {device} | train {len(train_indices)} | val (held-out styles {HELD_OUT_STYLES}) {len(val_indices)} | classes {num_classes}", flush=True)

    model = GlyphClassifier(num_classes).to(device)
    best_accuracy = _train(model, images, labels, train_indices, val_indices, device)

    calibration = _measure_calibration(model, images, data["labels"], num_classes, device)
    print(f"best held-out-style acc {best_accuracy:.3f} | {calibration}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_classes": num_classes, "calibration": calibration._asdict(),
                "held_out_accuracy": best_accuracy, "data": str(args.data)}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
