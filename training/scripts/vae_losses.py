#!/usr/bin/env python3
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from vae_data import AngleGMMParams
from vae_model import VAE, ModelShape, unflatten_output

ENDPOINT_LOSS_WEIGHT = 1.0  # 終点座標のMSEに掛ける重み(strokes_lossと同程度のスケールになるよう設計してある)
CONNECTION_LOSS_WEIGHT = 12.0  # 接続点一致損失に掛ける重み。weight sweepの結果、12は接続距離を約1割改善しつつstrokes_mse・重複スロット等への悪影響が候補中最小だった値(16・20はいずれも副作用がより大きく、12→16→20の単調な関係にはなっていない)
CONNECTION_LENGTH_WEIGHT_CAP = 3.0  # 接続ペアの重みの上限倍率(connection_pair_weights参照)。weight sweepの結果、ハネ由来の短いペアの接続距離改善の大部分(3.0で-18.7%、5.0でも-24.8%と伸びが鈍化)をstrokes_mseへの悪化がほぼない(+0.3%)うちに得られる値
CONNECTION_LENGTH_EPSILON = 1e-6  # 0除算回避(existence=0のpaddingスロットは長さ0になるため)
NEARBY_LOSS_WEIGHT = 0.03  # 近傍点間隔一致損失に掛ける重み。weight sweepの結果、duplicate_pairs・spacing_error_meanが単調に改善しstrokes_mseの悪化もない範囲の上限で、0.1以降は両指標とも悪化に転じる(損失の生の値が大きく、重みを上げすぎると学習全体が不安定化するため)
NEARBY_DISTANCE_SCALE = 15.0  # 重みの減衰スケール。実データの非接続ペア距離分布(5〜10%ileが13.66〜17.94)を踏まえ、目・日等の格子状部首の間隔付近だけに重みが乗るよう選んだ
ANGLE_NATURALNESS_LOSS_WEIGHT = 1.0  # 角度の自然さ損失に掛ける重み。weight sweepの結果、strokes_mseの悪化を+11%程度に抑えつつ角度対数密度の改善が最大だった値(2.0以上ではコストが増える一方、密度改善はむしろ弱まった)
CROSSING_GATE_LOW = 0.08  # 交点パラメータt, uがこの範囲内ならストローク内部での交差とみなす(下限)。端点付近の接続点を除外する
CROSSING_GATE_HIGH = 0.92  # 同上(上限)
CROSSING_GATE_SHARPNESS = 40.0  # _interior_gateのsigmoidの急峻さ(大きいほど矩形窓に近づく)。未検証(たたき台)
CROSSING_DENOM_EPSILON = 1e-6  # 平行な弦同士でのゼロ除算回避
CROSSING_LOSS_WEIGHT = 0.3  # weight sweepの結果、0.3〜0.5あたりで斜め関与の交差の改善が頭打ちになりそれ以降はコストだけ伸びるため、knee(境目)に近い値を採用した。本番同様のearly stopping・複数seedでも両seedで斜め関与の改善・strokes_mseの改善が一致した


def stroke_endpoints(strokes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # cos/sinは実際のラジアン値でないと意味を持たないため、角度・長さは一旦実スケールへ戻す。
    # lengthは弦(始点-終点間)の長さそのものなので、制御点オフセット(曲がり)の影響を受けない
    strokes_destd = strokes * std + mean
    start = strokes_destd[..., 0:2]
    angle = strokes_destd[..., 2]
    length = strokes_destd[..., 3]
    direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
    end = start + length.unsqueeze(-1) * direction
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


def stroke_points(strokes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # 接続点一致損失用に、各スロットの始点・終点を1つの点列にまとめる。並び順(偶数index=始点、
    # 奇数index=終点)はextract_stroke_features.pyのconnections行列の点indexと対応させている。
    # evaluate_vae.pyでも接続距離の診断に使う
    batch_size, slot_count, _ = strokes.shape
    start = strokes[..., 0:2]
    end = stroke_endpoints(strokes, mean, std)
    return torch.stack([start, end], dim=2).reshape(batch_size, slot_count * 2, 2)


def connection_pair_weights(strokes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # ハネで分割されたセグメントは元のストロークの一部でしかなく長さが短いため、浮くと「短い孤立した
    # 棒切れ」に見えて視覚的なダメージが大きい。そこで、ペアのうち短い方のセグメント長が平均より
    # 短いほど、その接続ペアの重みを引き上げる(平均的な長さのペアは重み1.0のまま変化しない)
    length = strokes[..., 3] * std[3] + mean[3]  # (B, slot_count)
    point_length = torch.repeat_interleave(length, 2, dim=-1)  # (B, slot_count*2)。始点・終点は同じ長さを共有
    min_length = torch.minimum(point_length.unsqueeze(-1), point_length.unsqueeze(-2))
    min_length = torch.clamp(min_length, min=CONNECTION_LENGTH_EPSILON)
    return torch.clamp(mean[3] / min_length, min=1.0, max=CONNECTION_LENGTH_WEIGHT_CAP)


def _compute_connection_loss(
    points: torch.Tensor, connections: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    # connectionsは上三角のみが立っているので、立っているペアの座標同士の距離の二乗をそのまま合計すればよい
    diff = points.unsqueeze(2) - points.unsqueeze(1)
    dist_sq = (diff**2).sum(dim=-1)
    return (dist_sq * connections * weights).sum(dim=(1, 2)).mean()


def _stroke_points_real(strokes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # stroke_pointsは接続点一致損失用に標準化スケールのまま返すが、近傍点間隔一致損失は
    # NEARBY_DISTANCE_SCALE(実スケールの値)と比較するため、実スケールに戻したものが必要になる
    return stroke_points(strokes, mean, std) * std[0:2] + mean[0:2]


def _nearby_pair_mask(existence: torch.Tensor, connections: torch.Tensor) -> torch.Tensor:
    # 近傍点間隔一致損失の対象ペアを絞り込むマスク。接続点一致損失が対象とする接続点(距離ほぼ0、connection_lossが
    # 既に担当)と、同一ストローク内の始点・終点ペア(strokes_loss側のlength特徴量が既に担当)は対象外にする
    slot_count = existence.shape[1]
    point_count = slot_count * 2
    point_exist = torch.repeat_interleave(existence, 2, dim=-1)
    exist_pair = point_exist.unsqueeze(2) * point_exist.unsqueeze(1)

    slot_index = torch.arange(slot_count, device=existence.device)
    same_point_or_slot = torch.eye(point_count, device=existence.device)
    same_point_or_slot[2 * slot_index, 2 * slot_index + 1] = 1.0
    same_point_or_slot[2 * slot_index + 1, 2 * slot_index] = 1.0
    connections_full = connections + connections.transpose(-1, -2)

    return exist_pair * (1.0 - same_point_or_slot.unsqueeze(0)) * (1.0 - connections_full)


def _nearby_pair_targets(
    strokes: torch.Tensor, existence: torch.Tensor, connections: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # 正解(strokes)側の実スケール距離から、近傍点間隔一致損失の目標距離と重みを求める
    points = _stroke_points_real(strokes, mean, std)
    diff = points.unsqueeze(2) - points.unsqueeze(1)
    real_dist = diff.norm(dim=-1)

    weight = torch.exp(-real_dist / NEARBY_DISTANCE_SCALE) * _nearby_pair_mask(existence, connections)
    weight = torch.triu(weight, diagonal=1)  # connectionsと同じく上三角のみを使い、二重カウントを避ける
    return real_dist, weight


def _compute_nearby_spacing_loss(
    strokes_recon: torch.Tensor,
    real_dist: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    # 再構成後の距離を、正解ペア一律の目標(0)ではなく実データそのものの距離に近づける。これにより
    # 「近いのが正解」のペアを壊さずに、その間隔だけを正確に保つよう学習させる
    points_recon = _stroke_points_real(strokes_recon, mean, std)
    diff = points_recon.unsqueeze(2) - points_recon.unsqueeze(1)
    recon_dist = diff.norm(dim=-1)
    return (weight * (recon_dist - real_dist) ** 2).sum(dim=(1, 2)).mean()


def _pairwise_intersection_params(
    strokes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # 各ストロークを弦(始点->終点)とみなし、全ペア(i, j)の交点パラメータt(iの弦上の位置)・
    # u(jの弦上の位置)をクラメルの公式(2元1次方程式)で解く。t, uはアフィン変換で不変
    # (2次元クロス積の分子・分母に現れるdetが約分される)ため、標準化スケールの座標をそのまま使ってよい
    start = strokes[..., 0:2]
    end = stroke_endpoints(strokes, mean, std)
    direction = end - start  # (B, slot_count, 2)

    d1 = direction.unsqueeze(2)  # (B, slot_count, 1, 2) -- iの方向、jへブロードキャスト
    d2 = direction.unsqueeze(1)  # (B, 1, slot_count, 2) -- jの方向、iへブロードキャスト
    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    # 平行な弦(denom≈0)はt, uが発散するが、_interior_gateで範囲外に押し出されるため実害はない
    denom = torch.where(denom.abs() < CROSSING_DENOM_EPSILON, torch.full_like(denom, CROSSING_DENOM_EPSILON), denom)

    diff = start.unsqueeze(1) - start.unsqueeze(2)  # p3 - p1, (B, slot_count(i), slot_count(j), 2)
    t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / denom
    u = (diff[..., 0] * d1[..., 1] - diff[..., 1] * d1[..., 0]) / denom
    return t, u


def _interior_gate(t: torch.Tensor) -> torch.Tensor:
    # t=CROSSING_GATE_LOW〜HIGHの範囲(ストローク内部)を、2つのsigmoidの積による
    # なめらかな矩形窓で近似する(線分交差判定は本質的に微分不可能なため)
    low = torch.sigmoid(CROSSING_GATE_SHARPNESS * (t - CROSSING_GATE_LOW))
    high = torch.sigmoid(CROSSING_GATE_SHARPNESS * (CROSSING_GATE_HIGH - t))
    return low * high


def off_diagonal_exist_pairs(existence: torch.Tensor) -> torch.Tensor:
    # 存在するスロット同士の全ペア(自分自身を除く)を1、それ以外を0とする上三角マスク。
    # 上三角のみを使うのはペアの二重カウントを避けるため。_crossing_pair_weights(このファイル)・
    # vae_synthetic_losses._compute_synthetic_crossing_lossで共通して使う
    slot_count = existence.shape[1]
    exist_pair = existence.unsqueeze(2) * existence.unsqueeze(1)
    same_slot = torch.eye(slot_count, device=existence.device)
    return torch.triu(exist_pair * (1.0 - same_slot.unsqueeze(0)), diagonal=1)


def _crossing_pair_weights(
    strokes: torch.Tensor, existence: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    # 正解(strokes)側で実際に交差しているペア(才のような正当な交差)は損失の対象外にする。
    # 交差の有無自体は不連続な事実であり勾配は不要なため、正解側の判定はハード閾値で行う。
    # 判定基準を正解側のみにするのは、再構成側は学習途中で崩れている可能性があるため
    with torch.no_grad():
        t, u = _pairwise_intersection_params(strokes, mean, std)
        interior = (t > CROSSING_GATE_LOW) & (t < CROSSING_GATE_HIGH)
        interior &= (u > CROSSING_GATE_LOW) & (u < CROSSING_GATE_HIGH)
        not_crossing = (~interior).float()

    return off_diagonal_exist_pairs(existence) * not_crossing


def weighted_crossing_penalty(
    strokes_recon: torch.Tensor, weights: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    # 再構成側の交差の強さ(0〜1の連続値)を求め、ペアごとの重みを掛けて損失にする。
    # _compute_crossing_loss(このファイル)・vae_synthetic_losses._compute_synthetic_crossing_lossで
    # 共通して使う(weightsの作り方だけが異なる)
    t, u = _pairwise_intersection_params(strokes_recon, mean, std)
    crossing_strength = _interior_gate(t) * _interior_gate(u)
    return (weights * crossing_strength).sum(dim=(1, 2)).mean()


def _compute_crossing_loss(
    strokes: torch.Tensor,
    strokes_recon: torch.Tensor,
    existence: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    # 正解で交差していないペアについて、再構成側の交差の強さにペナルティを与える
    weights = _crossing_pair_weights(strokes, existence, mean, std)
    return weighted_crossing_penalty(strokes_recon, weights, mean, std)


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


def angle_log_density(angle: torch.Tensor, gmm: AngleGMM) -> torch.Tensor:
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
    true_log_density = angle_log_density(true_angle, angle_gmm)
    recon_log_density = angle_log_density(recon_angle, angle_gmm)
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
    points_recon = stroke_points(strokes_recon, mean, std)
    # 重みは正解(strokes)側のセグメント長から決める。再構成側は崩れている可能性があり、
    # どのペアがハネ由来かの判定基準として使うべきではないため
    connection_weights = connection_pair_weights(strokes, mean, std)
    connection_loss = _compute_connection_loss(points_recon, connections, connection_weights)

    # 目・日等の格子状の部首は、実データで近い(しかし接続はしていない)ストローク同士の間隔が
    # 再構成で保てず束になって潰れやすい(重複)。実データの間隔そのものを再構成の目標にする
    nearby_dist, nearby_weight = _nearby_pair_targets(strokes, existence, connections, mean, std)
    nearby_spacing_loss = _compute_nearby_spacing_loss(strokes_recon, nearby_dist, nearby_weight, mean, std)

    # 実データの角度分布から外れた(非典型的な)角度を再構成するほど損失が大きくなるようにする
    angle_naturalness_loss = _compute_angle_naturalness_loss(
        strokes, strokes_recon, existence, mean, std, angle_gmm
    )

    # 正解で交差していないストロークペアが、再構成で交差してしまう(貫き)ことを抑制する
    crossing_loss = _compute_crossing_loss(strokes, strokes_recon, existence, mean, std)

    return (
        strokes_loss
        + existence_loss
        + beta * kl_divergence
        + ENDPOINT_LOSS_WEIGHT * endpoint_loss
        + CONNECTION_LOSS_WEIGHT * connection_loss
        + NEARBY_LOSS_WEIGHT * nearby_spacing_loss
        + ANGLE_NATURALNESS_LOSS_WEIGHT * angle_naturalness_loss
        + CROSSING_LOSS_WEIGHT * crossing_loss
    )
