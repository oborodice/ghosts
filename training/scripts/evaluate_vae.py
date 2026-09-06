#!/usr/bin/env python3
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from vae_eval_common import existence_mask_from_logits, load_checkpoint, load_validation_data
from vae_model import VAE, flatten_input, select_device, unflatten_output

ACTIVE_UNIT_THRESHOLD = 0.01  # 潜在次元ごとのKLがこれを下回る場合、その次元は「死んでいる」とみなす
WORST_SAMPLE_COUNT = 5


class ForwardResult(NamedTuple):
    strokes: torch.Tensor
    existence: torch.Tensor
    strokes_recon: torch.Tensor
    existence_logits: torch.Tensor


def _kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # 次元方向へは和を取らず、バッチ平均のみ取ることで次元ごとのKLを残す
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)


def _print_loss_breakdown(result: ForwardResult, kl_per_dim: torch.Tensor) -> None:
    print("== 1. Loss breakdown ==")
    # train_vae._compute_lossと同じ集約方法(sum→batch mean)で個別に集計する
    strokes, existence, strokes_recon, existence_logits = result
    mask = existence.unsqueeze(-1)
    strokes_loss = (((strokes_recon - strokes) ** 2) * mask).sum(dim=(1, 2)).mean()
    existence_loss = (
        F.binary_cross_entropy_with_logits(existence_logits, existence, reduction="none")
        .sum(dim=1)
        .mean()
    )
    kl_divergence = kl_per_dim.sum()

    print(f"strokes_loss:   {strokes_loss.item():.4f}")
    print(f"existence_loss: {existence_loss.item():.4f}")
    print(f"kl_divergence:  {kl_divergence.item():.4f}")
    print()


def _print_active_units(kl_per_dim: torch.Tensor) -> None:
    print("== 2. Per-dimension KL (active units) ==")
    for dim, kl in enumerate(kl_per_dim.tolist()):
        print(f"  dim {dim:2d}: {kl:.4f}")
    dead_count = (kl_per_dim < ACTIVE_UNIT_THRESHOLD).sum().item()
    print(f"Dead dimensions: {dead_count} / {len(kl_per_dim)} (threshold={ACTIVE_UNIT_THRESHOLD})")
    print()


def _print_weight_health(model: VAE) -> None:
    print("== 3. Weight health check ==")
    anomalies = [
        name for name, tensor in model.state_dict().items() if not torch.isfinite(tensor).all()
    ]
    if anomalies:
        print("Parameters with NaN/Inf detected:")
        for name in anomalies:
            print(f"  {name}")
    else:
        print("No NaN/Inf detected")
    print()


def _print_error_distribution(result: ForwardResult) -> None:
    print("== 4. Validation-wide error distribution ==")
    strokes, existence, strokes_recon, existence_logits = result
    feature_dim = strokes.shape[-1]
    mask = existence.unsqueeze(-1)
    # 実在するスロット・特徴量あたりの平均二乗誤差(画数による誤差の見かけ上の増減を避けるため、和ではなく平均を取る)
    sample_mse = ((strokes_recon - strokes) ** 2 * mask).sum(dim=(1, 2)) / (
        mask.sum(dim=(1, 2)) * feature_dim
    )
    sample_mse = sample_mse.cpu().numpy()

    existence_pred = existence_mask_from_logits(existence_logits)
    existence_true = existence.cpu().numpy().astype(bool)
    sample_accuracy = (existence_pred == existence_true).mean(axis=1)

    print(f"strokes MSE: mean={sample_mse.mean():.4f} max={sample_mse.max():.4f}")
    print(f"existence accuracy: mean={sample_accuracy.mean():.4f} min={sample_accuracy.min():.4f}")

    worst_indices = np.argsort(sample_mse)[::-1][:WORST_SAMPLE_COUNT]
    print(f"Validation samples with largest strokes MSE (top {WORST_SAMPLE_COUNT}):")
    for index in worst_indices:
        print(
            f"  index={index}: mse={sample_mse[index]:.4f} "
            f"existence_accuracy={sample_accuracy[index]:.4f}"
        )


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    data = load_validation_data(checkpoint, device)

    with torch.no_grad():
        # 損失の内訳・KLはtrain_vaeのvalidation lossと同じ経路(サンプリングzを含むforward)で再現する
        recon, mu, logvar = checkpoint.model(
            flatten_input(data.strokes_standardized, data.existence_tensor)
        )
        strokes_recon, existence_logits = unflatten_output(recon, checkpoint.shape)
        result = ForwardResult(
            data.strokes_standardized, data.existence_tensor, strokes_recon, existence_logits
        )

        # 誤差分布はワースト字形を実行のたびに入れ替えたくないため、ノイズのないmuから決定論的にデコードする
        recon_deterministic = checkpoint.model.decode(mu)
        strokes_recon_det, existence_logits_det = unflatten_output(
            recon_deterministic, checkpoint.shape
        )
        result_deterministic = ForwardResult(
            data.strokes_standardized, data.existence_tensor, strokes_recon_det, existence_logits_det
        )

    kl_per_dim = _kl_per_dim(mu, logvar)
    _print_loss_breakdown(result, kl_per_dim)
    _print_active_units(kl_per_dim)
    _print_weight_health(checkpoint.model)
    _print_error_distribution(result_deterministic)


if __name__ == "__main__":
    main()
