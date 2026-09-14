#!/usr/bin/env python3
# 学習済みモデルの品質診断・複数候補の比較の両方が使う指標計算をまとめる
import numpy as np
import torch

ACTIVE_UNIT_THRESHOLD = 0.01  # 潜在次元ごとのKLがこれを下回る場合、その次元は「死んでいる」とみなす


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # 次元方向へは和を取らず、バッチ平均のみ取ることで次元ごとのKLを残す
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)


def vertex_distance_real(
    vertices: torch.Tensor,
    existence: torch.Tensor,
    vertices_recon: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    # 標準化スケールのMSEとは別に、データ前処理(クリークによる頂点クラスタリング)の再構成誤差と
    # 直接比較できるよう、実座標スケールでの点ごとの距離を求める
    true_real = vertices * std + mean
    recon_real = vertices_recon * std + mean
    distance = (recon_real - true_real).norm(dim=-1)
    return distance[existence.bool()].cpu().numpy()
