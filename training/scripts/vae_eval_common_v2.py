#!/usr/bin/env python3
# 学習済みモデルの品質診断・複数候補の比較・目視確認のすべてが使う指標計算・データ読み込みをまとめる
from typing import Literal, NamedTuple

import numpy as np
import torch

from vae_checkpoint_v2 import Checkpoint
from vae_data_v2 import Datasets
from vae_model_v2 import DecoderOutput, flatten_input

ACTIVE_UNIT_THRESHOLD = 0.01  # 潜在次元ごとのKLがこれを下回る場合、その次元は「死んでいる」とみなす
DUPLICATE_POSITION_THRESHOLD = 0.15  # 標準化後の座標間距離がこれ未満なら、デコーダが同じ頂点を複数スロットに重複して割り当てているとみなす閾値

# 生成時にzを実データへ引き寄せるカーネル幅。次元数が多いほど同じbandwidthでもsoftmax重みが均一化し
# (次元の呪い)、実質的な近傍点数(=生成の新規性)が変わってしまうため、潜在次元数を倍増する前の構成と
# 同程度の有効サンプル数になるよう逆算した値(LATENT_DIMを変える場合は再計算が必要)
KERNEL_BANDWIDTH = 1.2


def attract_to_pool(z_raw: torch.Tensor, pool: torch.Tensor, bandwidth: float) -> torch.Tensor:
    # Nadaraya-Watson推定量(重み付き平均)でz_rawをpoolへ引き寄せる。学習時の合成z構築・生成時の
    # 両方がこの関数を経由することで、2つの実装が食い違う(過去に実際に起きた)ことを構造的に防ぐ
    dist_sq = torch.cdist(z_raw, pool) ** 2
    weights = torch.softmax(-dist_sq / (2 * bandwidth * bandwidth), dim=1)
    return weights @ pool


@torch.no_grad()
def attract_to_latent_prior(z_raw: torch.Tensor, mu_real: torch.Tensor) -> torch.Tensor:
    return attract_to_pool(z_raw, mu_real, KERNEL_BANDWIDTH)


class Batch(NamedTuple):
    vertices: torch.Tensor
    vertex_existence: torch.Tensor
    stroke_vertex_indices: torch.Tensor
    stroke_offsets: torch.Tensor
    stroke_existence: torch.Tensor


def load_batch(datasets: Datasets, split: Literal["train", "val"], device: torch.device) -> Batch:
    # TensorDataset.tensorsはフィールド名を持たないタプルなので、この順序はvae_data_v2._build_datasetが
    # TensorDatasetを組み立てる際の引数順と対応させる必要がある
    tensors = getattr(datasets, split).tensors
    return Batch(*(t.to(device) for t in tensors))


def encode_batch(checkpoint: Checkpoint, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    x = flatten_input(
        batch.vertices, batch.vertex_existence, batch.stroke_vertex_indices,
        batch.stroke_offsets, batch.stroke_existence, checkpoint.shape,
    )
    return checkpoint.model.encode(x)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # 次元方向へは和を取らず、バッチ平均のみ取ることで次元ごとのKLを残す
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)


def to_real_scale(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    return (value * std + mean).cpu().numpy()


def vertex_distance_real(
    vertices: torch.Tensor,
    existence: torch.Tensor,
    vertices_recon: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    # 標準化スケールのMSEとは別に、データ前処理(クリークによる頂点クラスタリング)の再構成誤差と
    # 直接比較できるよう、実座標スケールでの点ごとの距離を求める
    true_real = to_real_scale(vertices, mean, std)
    recon_real = to_real_scale(vertices_recon, mean, std)
    distance = np.linalg.norm(recon_real - true_real, axis=-1)
    return distance[existence.bool().cpu().numpy()]


class RealScaleStrokes(NamedTuple):
    start: np.ndarray  # (N, stroke_count, 2)
    end: np.ndarray  # (N, stroke_count, 2)
    offsets: np.ndarray  # (N, stroke_count, 2)


def true_strokes_real(checkpoint: Checkpoint, batch: Batch) -> RealScaleStrokes:
    vertices_real = to_real_scale(batch.vertices, checkpoint.vertex_mean, checkpoint.vertex_std)
    start_index = batch.stroke_vertex_indices[..., 0:1].cpu().numpy()
    end_index = batch.stroke_vertex_indices[..., 1:2].cpu().numpy()
    offsets_real = to_real_scale(batch.stroke_offsets, checkpoint.stroke_offset_mean, checkpoint.stroke_offset_std)
    return RealScaleStrokes(
        np.take_along_axis(vertices_real, start_index, axis=1),
        np.take_along_axis(vertices_real, end_index, axis=1),
        offsets_real,
    )


def reconstructed_strokes_real(checkpoint: Checkpoint, decoder_output: DecoderOutput) -> RealScaleStrokes:
    start_real = to_real_scale(decoder_output.start_points, checkpoint.vertex_mean, checkpoint.vertex_std)
    end_real = to_real_scale(decoder_output.end_points, checkpoint.vertex_mean, checkpoint.vertex_std)
    offsets_real = to_real_scale(
        decoder_output.stroke_offsets, checkpoint.stroke_offset_mean, checkpoint.stroke_offset_std
    )
    return RealScaleStrokes(start_real, end_real, offsets_real)


def stroke_curves(
    start_points: np.ndarray, end_points: np.ndarray, offsets: np.ndarray, existence_mask: np.ndarray
) -> list[tuple[complex, complex, complex]]:
    # (始点, 終点, オフセット) -> (始点, 制御点, 終点)の2次ベジェ。制御点は弦(始点-終点)の中点をoffsetだけずらした点
    curves = []
    for (start_x, start_y), (end_x, end_y), (offset_x, offset_y), exists in zip(
        start_points, end_points, offsets, existence_mask
    ):
        if not exists:
            continue
        start = complex(start_x, start_y)
        end = complex(end_x, end_y)
        control = (start + end) / 2 + complex(offset_x, offset_y)
        curves.append((start, control, end))
    return curves


def duplicate_slot_pairs(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 1サンプル分の頂点座標(存在するスロットのみ)を受け取り、標準化後の座標間距離が
    # DUPLICATE_POSITION_THRESHOLD未満のペア(i, j)を、positions内でのインデックスの組として返す
    distance = torch.cdist(positions, positions)
    return torch.triu(distance < DUPLICATE_POSITION_THRESHOLD, diagonal=1).nonzero(as_tuple=True)


def count_duplicate_slots(vertex_features: torch.Tensor, existence_mask: torch.Tensor) -> np.ndarray:
    # 各サンプルで、存在すると判定されたスロット同士の座標が極端に近いペアの数を数える
    counts = []
    for sample_idx in range(vertex_features.shape[0]):
        active_indices = existence_mask[sample_idx].nonzero(as_tuple=True)[0]
        if len(active_indices) < 2:
            counts.append(0)
            continue
        positions = vertex_features[sample_idx, active_indices]
        pair_rows, _ = duplicate_slot_pairs(positions)
        counts.append(len(pair_rows))
    return np.array(counts)
