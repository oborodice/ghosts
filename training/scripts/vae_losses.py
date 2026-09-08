#!/usr/bin/env python3
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from vae_data import AngleGMMParams

ENDPOINT_LOSS_WEIGHT = 1.0  # 終点座標のMSEに掛ける重み(strokes_lossと同程度のスケールになるよう設計してある)
CONNECTION_LOSS_WEIGHT = 5.0  # 接続点一致損失に掛ける重み。weight sweepの結果、strokes_mseを悪化させずに接続距離を改善できる上限がこの付近だった(10以上ではstrokes_mseが明確に悪化する)
ANGLE_NATURALNESS_LOSS_WEIGHT = 1.0  # 角度の自然さ損失に掛ける重み。weight sweepの結果、strokes_mseの悪化を+11%程度に抑えつつ角度対数密度の改善が最大だった値(2.0以上ではコストが増える一方、密度改善はむしろ弱まった)


def stroke_endpoints(strokes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # cos/sinは実際のラジアン値でないと意味を持たないため、角度・長さ・曲率は一旦実スケールへ戻す
    strokes_destd = strokes * std + mean
    start = strokes_destd[..., 0:2]
    angle = strokes_destd[..., 2]
    curvature = strokes_destd[..., 3]
    length = strokes_destd[..., 4]
    radius = length - curvature
    direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    end = start + radius.unsqueeze(-1) * direction
    # start_x, start_yと同じ統計量(mean/stdの先頭2要素)で標準化し、strokes_lossと比較可能なスケールに揃える
    return (end - mean[0:2]) / std[0:2]


def _compute_endpoint_loss(
    strokes: torch.Tensor,
    strokes_recon: torch.Tensor,
    existence: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    # angle/lengthの誤差は終点位置(特にストロークが長いほど)で増幅されるため、
    # 生パラメータのMSEとは別に終点座標自体のMSEも損失に加える
    end_true = stroke_endpoints(strokes, mean, std)
    end_recon = stroke_endpoints(strokes_recon, mean, std)
    mask = existence.unsqueeze(-1)
    return (((end_recon - end_true) ** 2) * mask).sum(dim=(1, 2)).mean()


def _stroke_points(strokes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # 接続点一致損失用に、各スロットの始点・終点を1つの点列にまとめる。並び順(偶数index=始点、
    # 奇数index=終点)はextract_stroke_features.pyのconnections行列の点indexと対応させている
    batch_size, slot_count, _ = strokes.shape
    start = strokes[..., 0:2]
    end = stroke_endpoints(strokes, mean, std)
    return torch.stack([start, end], dim=2).reshape(batch_size, slot_count * 2, 2)


def _compute_connection_loss(points: torch.Tensor, connections: torch.Tensor) -> torch.Tensor:
    # connectionsは上三角のみが立っているので、立っているペアの座標同士の距離の二乗をそのまま合計すればよい
    diff = points.unsqueeze(2) - points.unsqueeze(1)
    dist_sq = (diff**2).sum(dim=-1)
    return (dist_sq * connections).sum(dim=(1, 2)).mean()


class AngleGMM(NamedTuple):
    means: torch.Tensor  # (K, 2)
    precisions: torch.Tensor  # (K, 2, 2)、共分散行列の逆行列
    log_dets: torch.Tensor  # (K,)、共分散行列の対数行列式
    log_weights: torch.Tensor  # (K,)


def build_angle_gmm(params: AngleGMMParams, device: torch.device) -> AngleGMM:
    precisions = np.linalg.inv(params.covariances)
    _, log_dets = np.linalg.slogdet(params.covariances)
    return AngleGMM(
        torch.tensor(params.means, dtype=torch.float32, device=device),
        torch.tensor(precisions, dtype=torch.float32, device=device),
        torch.tensor(log_dets, dtype=torch.float32, device=device),
        torch.tensor(np.log(params.weights), dtype=torch.float32, device=device),
    )


def _angle_log_density(angle: torch.Tensor, gmm: AngleGMM) -> torch.Tensor:
    # 角度(ラジアン)を(cosθ, sinθ)に変換し、GMM(固定パラメータ)のもとでの対数密度を評価する
    points = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    diff = points.unsqueeze(-2) - gmm.means  # (..., K, 2)
    # コンポーネントごとの2次形式 (x-μ)^T Σ^-1 (x-μ)
    quad = torch.einsum("...ki,kij,...kj->...k", diff, gmm.precisions, diff)
    log_component_density = -0.5 * (quad + gmm.log_dets + 2 * np.log(2 * np.pi))
    return torch.logsumexp(gmm.log_weights + log_component_density, dim=-1)


def _compute_angle_naturalness_loss(
    strokes: torch.Tensor,
    strokes_recon: torch.Tensor,
    existence: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    angle_gmm: AngleGMM,
) -> torch.Tensor:
    # 正解の角度自体が非典型的な場合(実データにも一定数存在する)に再構成をそこから引き離してしまわないよう、
    # 絶対的な自然さではなく正解と比較した相対的な自然さで評価する。正解より不自然になった分だけを
    # 損失にすることで、正解通りに再構成できていれば(正解が非典型的でも)ペナルティが発生しないようにする
    true_angle = strokes[..., 2] * std[2] + mean[2]
    recon_angle = strokes_recon[..., 2] * std[2] + mean[2]
    true_log_density = _angle_log_density(true_angle, angle_gmm)
    recon_log_density = _angle_log_density(recon_angle, angle_gmm)
    penalty = torch.clamp(true_log_density - recon_log_density, min=0.0)
    return (penalty * existence).sum(dim=1).mean()


def compute_loss(
    strokes: torch.Tensor,
    existence: torch.Tensor,
    strokes_recon: torch.Tensor,
    existence_logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    connections: torch.Tensor,
    angle_gmm: AngleGMM,
    beta: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    mask = existence.unsqueeze(-1)  # (B, slot_count, 1) -> strokesの特徴量方向へブロードキャスト
    # 特徴量・スロット方向は和、バッチ方向は平均を取る(sum→batch mean)。
    # 要素方向で平均を取るとKLダイバージェンスに対して再構成損失が相対的に小さくなり、posterior collapseを起こしやすくなるため避ける
    strokes_loss = (((strokes_recon - strokes) ** 2) * mask).sum(dim=(1, 2)).mean()
    existence_loss = (
        F.binary_cross_entropy_with_logits(existence_logits, existence, reduction="none").sum(dim=1).mean()
    )
    kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    endpoint_loss = _compute_endpoint_loss(strokes, strokes_recon, existence, mean, std)

    # endpoint_loss(終点MSE)は各ストロークの終点を独立に正解へ近づけるだけで、接続しているはずの
    # 別スロット同士が再構成後も一致する保証はない。接続点一致損失で再構成後の該当点同士を直接近づける
    points_recon = _stroke_points(strokes_recon, mean, std)
    connection_loss = _compute_connection_loss(points_recon, connections)

    # 実データの角度分布から外れた(非典型的な)角度を再構成するほど損失が大きくなるようにする
    angle_naturalness_loss = _compute_angle_naturalness_loss(
        strokes, strokes_recon, existence, mean, std, angle_gmm
    )

    return (
        strokes_loss
        + existence_loss
        + beta * kl_divergence
        + ENDPOINT_LOSS_WEIGHT * endpoint_loss
        + CONNECTION_LOSS_WEIGHT * connection_loss
        + ANGLE_NATURALNESS_LOSS_WEIGHT * angle_naturalness_loss
    )
