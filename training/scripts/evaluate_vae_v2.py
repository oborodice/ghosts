#!/usr/bin/env python3
# 頂点のみのこの段階でも意味が成り立つ再構成品質のチェック(損失の内訳・潜在次元ごとのKL・重みの健全性・
# 誤差分布・丸暗記化の確認・重複スロットの検出)を行う。交差数・3本以上合流・接続距離のチェックは
# ストロークのポインタ機構がまだ無いため対象外
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from vae_checkpoint_v2 import load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common_v2 import ACTIVE_UNIT_THRESHOLD, kl_per_dim as compute_kl_per_dim, vertex_distance_real
from vae_model_v2 import VAE, ModelShape, flatten_input, select_device, unflatten_output

WORST_SAMPLE_COUNT = 5
DUPLICATE_POSITION_THRESHOLD = 0.15  # 標準化後の座標間距離がこれ未満なら、デコーダが同じ頂点を複数スロットに重複して割り当てているとみなす閾値
EXISTENCE_THRESHOLD = 0.5  # existenceの確率(Sigmoid(existence_logits))をbool判定に変換する閾値


class ForwardResult(NamedTuple):
    vertices: torch.Tensor
    existence: torch.Tensor
    vertices_recon: torch.Tensor
    existence_logits: torch.Tensor


def _existence_mask_from_logits(existence_logits: torch.Tensor) -> np.ndarray:
    return (torch.sigmoid(existence_logits) > EXISTENCE_THRESHOLD).cpu().numpy()


def _print_loss_breakdown(result: ForwardResult, kl_per_dim: torch.Tensor) -> None:
    print("== 1. Loss breakdown ==")
    # vae_losses_v2.compute_lossと同じ集約方法(sum→batch mean)で個別に集計する
    vertices, existence, vertices_recon, existence_logits = result
    mask = existence.unsqueeze(-1)
    vertex_loss = (((vertices_recon - vertices) ** 2) * mask).sum(dim=(1, 2)).mean()
    existence_loss = (
        F.binary_cross_entropy_with_logits(existence_logits, existence, reduction="none").sum(dim=1).mean()
    )
    kl_divergence = kl_per_dim.sum()

    print(f"vertex_loss:    {vertex_loss.item():.4f}")
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
    anomalies = [name for name, tensor in model.state_dict().items() if not torch.isfinite(tensor).all()]
    if anomalies:
        print("Parameters with NaN/Inf detected:")
        for name in anomalies:
            print(f"  {name}")
    else:
        print("No NaN/Inf detected")
    print()


def _decode_deterministic(
    model: VAE, mu: torch.Tensor, shape: ModelShape, vertices_std: torch.Tensor, existence: torch.Tensor
) -> ForwardResult:
    # 誤差分布・丸暗記化チェックで共通して使う決定論的デコード(ノイズのないmuから)
    recon = model.decode(mu)
    vertices_recon, existence_logits = unflatten_output(recon, shape)
    return ForwardResult(vertices_std, existence, vertices_recon, existence_logits)


def _vertex_mse(result: ForwardResult) -> np.ndarray:
    vertices, existence, vertices_recon, _ = result
    feature_dim = vertices.shape[-1]
    mask = existence.unsqueeze(-1)
    # 実在するスロット・座標軸あたりの平均二乗誤差(頂点数による誤差の見かけ上の増減を避けるため、和ではなく平均を取る)
    sample_mse = ((vertices_recon - vertices) ** 2 * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * feature_dim)
    return sample_mse.cpu().numpy()


def _print_error_distribution(result: ForwardResult, mean: torch.Tensor, std: torch.Tensor) -> None:
    print("== 4. Validation-wide error distribution ==")
    vertices, existence, vertices_recon, existence_logits = result
    sample_mse = _vertex_mse(result)
    distance_real = vertex_distance_real(vertices, existence, vertices_recon, mean, std)

    existence_pred = _existence_mask_from_logits(existence_logits)
    existence_true = existence.cpu().numpy().astype(bool)
    sample_accuracy = (existence_pred == existence_true).mean(axis=1)

    print(f"vertex MSE (standardized scale): mean={sample_mse.mean():.4f} max={sample_mse.max():.4f}")
    print(f"vertex distance (real coordinate scale): mean={distance_real.mean():.4f} max={distance_real.max():.4f}")
    print(f"existence accuracy: mean={sample_accuracy.mean():.4f} min={sample_accuracy.min():.4f}")

    worst_indices = np.argsort(sample_mse)[::-1][:WORST_SAMPLE_COUNT]
    print(f"Validation samples with largest vertex MSE (top {WORST_SAMPLE_COUNT}):")
    for index in worst_indices:
        print(f"  index={index}: mse={sample_mse[index]:.4f} existence_accuracy={sample_accuracy[index]:.4f}")
    print()


def _avg_exp_logvar(logvar: torch.Tensor, alive_mask: torch.Tensor) -> float:
    # 生きている次元について、reparameterizeで乗せるノイズexp(logvar)の平均。
    # 1に近いほど事前分布相当のノイズを保っており、0に近いほどノイズなしの決定論的な符号化(丸暗記寄り)であることを示す
    return logvar.exp().mean(dim=0)[alive_mask].mean().item() if alive_mask.any() else float("nan")


def _print_memorization_check(
    val_result: ForwardResult,
    train_result: ForwardResult,
    val_logvar: torch.Tensor,
    train_logvar: torch.Tensor,
    kl_per_dim: torch.Tensor,
) -> None:
    # 生成(z〜N(0,1)からのdecode)は学習で一度も使っていないzから始まるため、
    # encoderが各学習データにノイズなしの点を割り当てて丸暗記していないか(過学習していないか)を確認する
    print("== 5. Memorization check ==")
    val_mse = _vertex_mse(val_result).mean()
    train_mse = _vertex_mse(train_result).mean()

    alive_mask = kl_per_dim >= ACTIVE_UNIT_THRESHOLD
    val_avg_exp_logvar = _avg_exp_logvar(val_logvar, alive_mask)
    # 丸暗記化はまさに学習で見た点(train)で起きるため、trainのexp(logvar)も別途確認する
    train_avg_exp_logvar = _avg_exp_logvar(train_logvar, alive_mask)

    print(f"train vertex MSE: {train_mse:.4f}")
    print(f"val vertex MSE:   {val_mse:.4f}")
    print(f"train/val gap:    {val_mse - train_mse:.4f} (larger suggests overfitting/memorization)")
    print(f"avg exp(logvar) (active dims): train={train_avg_exp_logvar:.4f} val={val_avg_exp_logvar:.4f}")
    print()


def _count_duplicate_slots(vertices_recon: torch.Tensor, existence_mask: torch.Tensor) -> np.ndarray:
    # 各サンプルで、存在すると判定されたスロット同士の座標が極端に近いペアを数える
    counts = []
    for sample_idx in range(vertices_recon.shape[0]):
        active_indices = existence_mask[sample_idx].nonzero(as_tuple=True)[0]
        positions = vertices_recon[sample_idx, active_indices]
        if len(active_indices) < 2:
            counts.append(0)
            continue
        distance = torch.cdist(positions, positions)
        distance.fill_diagonal_(float("inf"))
        # 対称行列のためペア(i, j)と(j, i)の両方がカウントされる。2で割って実際のペア数に直す
        counts.append((distance < DUPLICATE_POSITION_THRESHOLD).sum().item() // 2)
    return np.array(counts)


def _print_duplicate_slots(result: ForwardResult) -> None:
    print("== 6. Duplicate slot check ==")
    _, _, vertices_recon, existence_logits = result
    existence_mask = torch.from_numpy(_existence_mask_from_logits(existence_logits))
    duplicate_counts = _count_duplicate_slots(vertices_recon, existence_mask)

    print(f"Samples with >=1 near-duplicate slot pair: {(duplicate_counts > 0).sum()} / {len(duplicate_counts)}")
    print(f"Average near-duplicate pairs per sample: {duplicate_counts.mean():.4f}")
    print()


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    datasets = prepare_datasets()

    # TensorDataset.tensorsはフィールド名を持たないタプルなので、この(頂点, existence)という順序は
    # vae_data_v2._build_datasetがTensorDatasetを組み立てる際の引数順と対応させる必要がある
    val_vertices_std, val_existence = (t.to(device) for t in datasets.val.tensors)
    train_vertices_std, train_existence = (t.to(device) for t in datasets.train.tensors)

    with torch.no_grad():
        # 損失の内訳・KLは学習時のvalidation lossと同じ経路(サンプリングzを含むforward)で再現する
        recon, mu, logvar = checkpoint.model(flatten_input(val_vertices_std, val_existence))
        vertices_recon, existence_logits = unflatten_output(recon, checkpoint.shape)
        result = ForwardResult(val_vertices_std, val_existence, vertices_recon, existence_logits)

        # 誤差分布はワースト字形を実行のたびに入れ替えたくないため、ノイズのないmuから決定論的にデコードする
        result_deterministic = _decode_deterministic(
            checkpoint.model, mu, checkpoint.shape, val_vertices_std, val_existence
        )

        # 丸暗記化チェック用にtrain側も同じ経路で再構成する
        train_mu, train_logvar = checkpoint.model.encode(flatten_input(train_vertices_std, train_existence))
        train_result = _decode_deterministic(
            checkpoint.model, train_mu, checkpoint.shape, train_vertices_std, train_existence
        )

    kl_per_dim = compute_kl_per_dim(mu, logvar)
    _print_loss_breakdown(result, kl_per_dim)
    _print_active_units(kl_per_dim)
    _print_weight_health(checkpoint.model)
    _print_error_distribution(result_deterministic, checkpoint.mean, checkpoint.std)
    _print_memorization_check(result_deterministic, train_result, logvar, train_logvar, kl_per_dim)
    _print_duplicate_slots(result_deterministic)


if __name__ == "__main__":
    main()
