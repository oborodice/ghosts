#!/usr/bin/env python3
# 頂点+ストローク全体の再構成品質のチェック(損失の内訳・潜在次元ごとのKL・重みの健全性・誤差分布・
# 丸暗記化の確認・頂点の重複スロットの検出)を行う。生成した字の交差数・3本以上合流などの集計は、
# report_generation_stats_v2.pyが担う
import argparse
from pathlib import Path

import numpy as np
import torch

from vae_checkpoint_v2 import load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common import existence_mask_from_logits
from vae_eval_common_v2 import (
    ACTIVE_UNIT_THRESHOLD,
    Batch,
    count_duplicate_slots,
    encode_batch,
    kl_per_dim as compute_kl_per_dim,
    load_batch,
    vertex_distance_real,
)
from vae_losses import build_angle_gmm
from vae_losses_v2 import LossComponents, compute_loss
from vae_model_v2 import CHECKPOINT_PATH, VAE, DecoderOutput, select_device
from vae_training_v2 import BETA

WORST_SAMPLE_COUNT = 5
TOP_KL_DIM_COUNT = 5


def _print_loss_breakdown(losses: LossComponents) -> None:
    print("== 1. Loss breakdown ==")
    # 学習時の合計には、混ぜ合わせたzをdecodeして計算する損失(自己ループ・交差など)も加わるが、
    # 学習中の潜在のプールやストローク数のプールが要るため、ここでは再構成側の損失だけを示す。
    # 重みを掛けた合計(total)は、学習時のβに依存し、学習ログの検証損失とも一致しないため出さない
    print(
        "(reconstruction-side losses only, before weighting;"
        " the synthetic-z losses added during training are not included)"
    )
    names = [name for name in LossComponents._fields if name != "total"]
    label_width = max(len(name) for name in names) + 2  # 名前の後ろの":"と、値との間の空白の分
    for name in names:
        print(f"{name + ':':<{label_width}}{getattr(losses, name).item():.4f}")
    print()


def _print_active_units(kl_per_dim: torch.Tensor) -> None:
    print("== 2. Per-dimension KL (active units) ==")
    dead_count = (kl_per_dim < ACTIVE_UNIT_THRESHOLD).sum().item()
    print(f"Dead dimensions: {dead_count} / {len(kl_per_dim)} (threshold={ACTIVE_UNIT_THRESHOLD})")
    print(
        f"KL per dimension: total={kl_per_dim.sum():.2f} min={kl_per_dim.min():.4f}"
        f" median={kl_per_dim.median():.4f} max={kl_per_dim.max():.4f}"
    )
    # 情報が少数の次元に集まっていないかを見るため、KLの大きい次元と、その合計に占める割合を示す
    top_kl, top_dims = kl_per_dim.topk(TOP_KL_DIM_COUNT)
    top_dims_text = ", ".join(f"dim {dim}={kl:.3f}" for dim, kl in zip(top_dims.tolist(), top_kl.tolist()))
    top_share = top_kl.sum() / kl_per_dim.sum()
    print(f"Largest {TOP_KL_DIM_COUNT} dimensions ({top_share:.0%} of the total): {top_dims_text}")
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


def _masked_sample_mse(predicted: torch.Tensor, target: torch.Tensor, existence: torch.Tensor) -> np.ndarray:
    # サンプルごとの、実在するスロット・特徴量の軸あたりの平均二乗誤差(スロット数による誤差の見かけ上の
    # 増減を避けるため、和ではなく平均を取る)。predicted・target: (B, slots, features)、existence: (B, slots)
    mask = existence.unsqueeze(-1)
    sample_mse = ((predicted - target) ** 2 * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * target.shape[-1])
    return sample_mse.cpu().numpy()


def _vertex_mse(batch: Batch, decoder_output: DecoderOutput) -> np.ndarray:
    return _masked_sample_mse(decoder_output.vertex_features, batch.vertices, batch.vertex_existence)


def _pointer_accuracy(logits: torch.Tensor, target_index: torch.Tensor, stroke_existence: torch.Tensor) -> np.ndarray:
    # サンプルごとの、実在するストロークのうちポインタが正解頂点を選べた割合(top-1)
    correct = (logits.argmax(dim=-1) == target_index).float()
    return ((correct * stroke_existence).sum(dim=1) / stroke_existence.sum(dim=1)).cpu().numpy()


def _print_error_distribution(
    batch: Batch, decoder_output: DecoderOutput, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> None:
    print("== 4. Validation-wide error distribution ==")
    sample_mse = _vertex_mse(batch, decoder_output)
    distance_real = vertex_distance_real(
        batch.vertices, batch.vertex_existence, decoder_output.vertex_features, vertex_mean, vertex_std
    )
    existence_pred = existence_mask_from_logits(decoder_output.vertex_existence_logits)
    existence_true = batch.vertex_existence.cpu().numpy().astype(bool)
    vertex_existence_accuracy = (existence_pred == existence_true).mean(axis=1)

    start_accuracy = _pointer_accuracy(
        decoder_output.start_pointer_logits, batch.stroke_vertex_indices[..., 0], batch.stroke_existence
    )
    end_accuracy = _pointer_accuracy(
        decoder_output.end_pointer_logits, batch.stroke_vertex_indices[..., 1], batch.stroke_existence
    )

    stroke_offset_mse = _masked_sample_mse(decoder_output.stroke_offsets, batch.stroke_offsets, batch.stroke_existence)

    print(f"vertex MSE (standardized scale): mean={sample_mse.mean():.4f} max={sample_mse.max():.4f}")
    print(f"vertex distance (real coordinate scale): mean={distance_real.mean():.4f} max={distance_real.max():.4f}")
    print(f"vertex existence accuracy: mean={vertex_existence_accuracy.mean():.4f} min={vertex_existence_accuracy.min():.4f}")
    print(f"start pointer accuracy (top-1): mean={start_accuracy.mean():.4f} min={start_accuracy.min():.4f}")
    print(f"end pointer accuracy (top-1):   mean={end_accuracy.mean():.4f} min={end_accuracy.min():.4f}")
    print(f"stroke offset MSE (standardized scale): mean={stroke_offset_mse.mean():.4f} max={stroke_offset_mse.max():.4f}")

    worst_indices = np.argsort(sample_mse)[::-1][:WORST_SAMPLE_COUNT]
    print(f"Validation samples with largest vertex MSE (top {WORST_SAMPLE_COUNT}):")
    for index in worst_indices:
        print(f"  index={index}: mse={sample_mse[index]:.4f} existence_accuracy={vertex_existence_accuracy[index]:.4f}")
    print()


def _avg_exp_logvar(logvar: torch.Tensor, alive_mask: torch.Tensor) -> float:
    # 生きている次元について、reparameterizeで乗せるノイズexp(logvar)の平均。
    # 1に近いほど事前分布相当のノイズを保っており、0に近いほどノイズなしの決定論的な符号化(丸暗記寄り)であることを示す
    return logvar.exp().mean(dim=0)[alive_mask].mean().item() if alive_mask.any() else float("nan")


def _print_memorization_check(
    val_batch: Batch,
    val_decoder_output: DecoderOutput,
    train_batch: Batch,
    train_decoder_output: DecoderOutput,
    val_logvar: torch.Tensor,
    train_logvar: torch.Tensor,
    kl_per_dim: torch.Tensor,
) -> None:
    # 生成(z〜N(0,1)からのdecode)は学習で一度も使っていないzから始まるため、
    # encoderが各学習データにノイズなしの点を割り当てて丸暗記していないか(過学習していないか)を確認する
    print("== 5. Memorization check ==")
    val_mse = _vertex_mse(val_batch, val_decoder_output).mean()
    train_mse = _vertex_mse(train_batch, train_decoder_output).mean()

    alive_mask = kl_per_dim >= ACTIVE_UNIT_THRESHOLD
    val_avg_exp_logvar = _avg_exp_logvar(val_logvar, alive_mask)
    # 丸暗記化はまさに学習で見た点(train)で起きるため、trainのexp(logvar)も別途確認する
    train_avg_exp_logvar = _avg_exp_logvar(train_logvar, alive_mask)

    print(f"train vertex MSE: {train_mse:.4f}")
    print(f"val vertex MSE:   {val_mse:.4f}")
    print(f"train/val gap:    {val_mse - train_mse:.4f} (larger suggests overfitting/memorization)")
    print(f"avg exp(logvar) (active dims): train={train_avg_exp_logvar:.4f} val={val_avg_exp_logvar:.4f}")
    print()


def _print_duplicate_slots(decoder_output: DecoderOutput) -> None:
    print("== 6. Duplicate slot check ==")
    existence_mask = torch.from_numpy(existence_mask_from_logits(decoder_output.vertex_existence_logits))
    duplicate_counts = count_duplicate_slots(decoder_output.vertex_features, existence_mask)

    print(f"Samples with >=1 near-duplicate slot pair: {(duplicate_counts > 0).sum()} / {len(duplicate_counts)}")
    print(f"Average near-duplicate pairs per sample: {duplicate_counts.mean():.4f}")
    print()


def _decode_deterministic(model: VAE, mu: torch.Tensor) -> DecoderOutput:
    # 誤差分布・丸暗記化チェックで共通して使う決定論的デコード(ノイズのないmuから)
    return model.decode(mu)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to the VAE checkpoint to evaluate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = select_device()
    checkpoint = load_checkpoint(device, checkpoint_path=args.checkpoint)
    datasets = prepare_datasets()

    val_batch = load_batch(datasets, "val", device)
    train_batch = load_batch(datasets, "train", device)

    with torch.no_grad():
        # 損失の内訳・KLは学習時のvalidation lossと同じ経路(サンプリングzを含むforward)で再現する
        mu, logvar = encode_batch(checkpoint, val_batch)
        decoder_output = checkpoint.model.decode(checkpoint.model.reparameterize(mu, logvar))
        # 学習と同じ損失の関数で計算する。βは、出力しない合計(total)にだけ掛かるため、
        # このチェックポイントの学習時のβと違っても、出力する内訳の値は変わらない
        losses = compute_loss(
            val_batch.vertices, val_batch.vertex_existence, val_batch.stroke_vertex_indices,
            val_batch.stroke_offsets, val_batch.stroke_existence, decoder_output, mu, logvar, BETA,
            checkpoint.vertex_mean, checkpoint.vertex_std, checkpoint.stroke_offset_mean, checkpoint.stroke_offset_std,
            build_angle_gmm(datasets.angle_gmm_params, device),
        )

        # 誤差分布はワースト字形を実行のたびに入れ替えたくないため、ノイズのないmuから決定論的にデコードする
        val_decoder_output = _decode_deterministic(checkpoint.model, mu)

        # 丸暗記化チェック用にtrain側も同じ経路で再構成する
        train_mu, train_logvar = encode_batch(checkpoint, train_batch)
        train_decoder_output = _decode_deterministic(checkpoint.model, train_mu)

    kl_per_dim = compute_kl_per_dim(mu, logvar)
    _print_loss_breakdown(losses)
    _print_active_units(kl_per_dim)
    _print_weight_health(checkpoint.model)
    _print_error_distribution(val_batch, val_decoder_output, checkpoint.vertex_mean, checkpoint.vertex_std)
    _print_memorization_check(
        val_batch, val_decoder_output, train_batch, train_decoder_output, logvar, train_logvar, kl_per_dim
    )
    _print_duplicate_slots(val_decoder_output)


if __name__ == "__main__":
    main()
