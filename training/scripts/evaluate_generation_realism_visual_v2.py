#!/usr/bin/env python3
# 画像ベースの診断分類器。生の数値特徴量ではなく、実際にレンダリングした画像を入力にして
# 本物/偽物を判別する。VAEの学習・推論には一切組み込まれない、独立した事後診断用のスクリプト
import argparse
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from vae_checkpoint_v2 import Checkpoint, load_checkpoint
from vae_classifier_dataset_v2 import CANVAS_SIZE, render_batch, sample_fake_strokes
from vae_classifier_model_v2 import HIDDEN_CHANNELS, ImageClassifier, save_classifier
from vae_data_v2 import prepare_datasets
from vae_eval_common_v2 import Batch, encode_batch, load_batch, true_strokes_real
from vae_model_v2 import CHECKPOINT_PATH, select_device

DECISION_THRESHOLD = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15
SEED = 0


class _ClassifierMetrics(NamedTuple):
    accuracy: float
    auc: float
    false_positive_rate: float  # 偽物を本物と誤判定した割合
    false_negative_rate: float  # 本物を偽物と誤判定した割合


def _build_labeled_dataset(
    checkpoint: Checkpoint, mu_real: torch.Tensor, batch: Batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    true_strokes = true_strokes_real(checkpoint, batch)
    true_existence = batch.stroke_existence.cpu().numpy().astype(bool)
    fake_strokes, fake_existence = sample_fake_strokes(checkpoint, mu_real, true_strokes, true_existence)

    real_x = render_batch(true_strokes.start, true_strokes.end, true_strokes.offsets, true_existence)
    fake_x = render_batch(fake_strokes.start, fake_strokes.end, fake_strokes.offsets, fake_existence)

    x = torch.cat([real_x, fake_x]).to(device)
    y = torch.cat([torch.ones(len(real_x)), torch.zeros(len(fake_x))]).to(device)  # ラベルは本物=1、偽物=0
    return x, y


def _run_epoch(loader: DataLoader, model: ImageClassifier, optimizer: torch.optim.Optimizer | None) -> float:
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
) -> tuple[ImageClassifier, torch.optim.Optimizer, DataLoader, DataLoader]:
    model = ImageClassifier(CANVAS_SIZE, HIDDEN_CHANNELS).to(train_x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)
    return model, optimizer, train_loader, test_loader


def _train_classifier(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> ImageClassifier:
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


def _compute_metrics(model: ImageClassifier, x: torch.Tensor, y: torch.Tensor) -> _ClassifierMetrics:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to the VAE checkpoint to evaluate"
    )
    parser.add_argument("--save-model", type=Path, default=None, help="Path to save the trained classifier to")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(SEED)
    device = select_device()
    checkpoint = load_checkpoint(device, checkpoint_path=args.checkpoint)
    datasets = prepare_datasets()
    train_batch = load_batch(datasets, "train", device)
    val_batch = load_batch(datasets, "val", device)

    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, train_batch)

    print("Building train dataset (rendering real, sampling and rendering fake)...")
    train_x, train_y = _build_labeled_dataset(checkpoint, mu_real, train_batch, device)
    print("Building test dataset...")
    test_x, test_y = _build_labeled_dataset(checkpoint, mu_real, val_batch, device)

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

    if args.save_model:
        save_classifier(model, args.save_model)
        print(f"Saved classifier to {args.save_model}")


if __name__ == "__main__":
    main()
