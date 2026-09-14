#!/usr/bin/env python3
# 学習済みモデルの品質診断・複数候補の比較・目視確認のすべてが使う指標計算・データ読み込みをまとめる
from typing import Literal, NamedTuple

import numpy as np
import torch

from vae_checkpoint_v2 import Checkpoint
from vae_data_v2 import Datasets
from vae_model_v2 import flatten_input

ACTIVE_UNIT_THRESHOLD = 0.01  # 潜在次元ごとのKLがこれを下回る場合、その次元は「死んでいる」とみなす


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
