#!/usr/bin/env python3
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from vae_model import (
    CHECKPOINT_PATH,
    HIDDEN_DIMS,
    LATENT_DIM,
    VAE,
    ModelShape,
    flatten_input,
    select_device,
    unflatten_output,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stroke_features.npz"

BETA = 1.0
KL_ANNEALING_EPOCHS = 30  # このepoch数をかけてβを0からBETAまで線形に引き上げる(warm-up)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
VAL_SPLIT = 0.1
SEED = 0
PATIENCE = 20
MAX_EPOCHS = 1000  # early stoppingが正常なら到達しない安全上限
STD_EPSILON = 1e-8  # 分散が0の特徴量があった場合のゼロ割回避


def load_stroke_features() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH)
    return data["strokes"], data["existence"]


def split_train_val_indices(kanji_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(kanji_count)
    val_size = int(kanji_count * VAL_SPLIT)
    return shuffled_indices[val_size:], shuffled_indices[:val_size]


def _compute_standardization_stats(
    strokes: np.ndarray, existence: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 標準化の統計量はtrainスプリットの実在ストローク(existence=1)のみから算出する
    train_valid_features = strokes[train_indices][existence[train_indices].astype(bool)]
    mean = train_valid_features.mean(axis=0)
    std = train_valid_features.std(axis=0)
    return mean, np.where(std < STD_EPSILON, STD_EPSILON, std)


def standardize(strokes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (strokes - mean) / std


def _build_dataset(indices: np.ndarray, strokes: np.ndarray, existence: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(strokes[indices], dtype=torch.float32),
        torch.tensor(existence[indices], dtype=torch.float32),
    )


class Datasets(NamedTuple):
    train: TensorDataset
    val: TensorDataset
    shape: ModelShape
    mean: np.ndarray
    std: np.ndarray


def _prepare_datasets() -> Datasets:
    strokes, existence = load_stroke_features()
    kanji_count, slot_count, feature_dim = strokes.shape
    shape = ModelShape(slot_count, feature_dim)

    train_indices, val_indices = split_train_val_indices(kanji_count)
    mean, std = _compute_standardization_stats(strokes, existence, train_indices)
    strokes_standardized = standardize(strokes, mean, std)

    train_dataset = _build_dataset(train_indices, strokes_standardized, existence)
    val_dataset = _build_dataset(val_indices, strokes_standardized, existence)
    return Datasets(train_dataset, val_dataset, shape, mean, std)


def _compute_loss(
    strokes: torch.Tensor,
    existence: torch.Tensor,
    strokes_recon: torch.Tensor,
    existence_logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    mask = existence.unsqueeze(-1)  # (B, slot_count, 1) -> strokesの特徴量方向へブロードキャスト
    # 特徴量・スロット方向は和、バッチ方向は平均を取る(sum→batch mean)。
    # 要素方向で平均を取るとKLダイバージェンスに対して再構成損失が相対的に小さくなり、posterior collapseを起こしやすくなるため避ける
    strokes_loss = (((strokes_recon - strokes) ** 2) * mask).sum(dim=(1, 2)).mean()
    existence_loss = (
        F.binary_cross_entropy_with_logits(existence_logits, existence, reduction="none")
        .sum(dim=1)
        .mean()
    )
    kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return strokes_loss + existence_loss + beta * kl_divergence


def _run_epoch(
    loader: DataLoader,
    model: VAE,
    shape: ModelShape,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    beta: float,
) -> float:
    # optimizerがNoneのとき(validation時)は重み更新を行わないeval modeとして扱う
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    with torch.set_grad_enabled(is_training):
        for strokes_batch, existence_batch in loader:
            strokes_batch, existence_batch = strokes_batch.to(device), existence_batch.to(device)
            recon, mu, logvar = model(flatten_input(strokes_batch, existence_batch))
            strokes_recon, existence_logits = unflatten_output(recon, shape)
            loss = _compute_loss(
                strokes_batch, existence_batch, strokes_recon, existence_logits, mu, logvar, beta
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * strokes_batch.size(0)
    return total_loss / len(loader.dataset)


def _save_checkpoint(model: VAE, shape: ModelShape, mean: np.ndarray, std: np.ndarray) -> None:
    # 生成(推論)に必要な情報のみを保存する(学習再開用のoptimizer状態・epoch数などは含まない)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dims": HIDDEN_DIMS,
            "latent_dim": LATENT_DIM,
            "slot_count": shape.slot_count,
            "feature_dim": shape.feature_dim,
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
        },
        CHECKPOINT_PATH,
    )


def _compute_beta(epoch: int) -> float:
    # 学習序盤にKL項がフルに効くと一部の潜在次元が使われなくなる(posterior collapse)ため、
    # KL_ANNEALING_EPOCHSかけてβを0からBETAまで線形に引き上げる
    return BETA * min(1.0, epoch / KL_ANNEALING_EPOCHS)


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    print(f"Using device: {device}")

    datasets = _prepare_datasets()
    train_loader = DataLoader(
        datasets.train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(datasets.val, batch_size=BATCH_SIZE, shuffle=False)

    model = VAE(datasets.shape, HIDDEN_DIMS, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        beta = _compute_beta(epoch)
        train_loss = _run_epoch(train_loader, model, datasets.shape, device, optimizer, beta)
        # 早期終了・チェックポイント選定はannealing中でも比較可能にするため、常に最終的なβ(=BETA)で評価する
        val_loss = _run_epoch(val_loader, model, datasets.shape, device, None, BETA)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} beta={beta:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            _save_checkpoint(model, datasets.shape, datasets.mean, datasets.std)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
