#!/usr/bin/env python3
from typing import NamedTuple

import torch
import torch.nn.functional as F

from vae_losses import AngleGMM, angle_log_density, off_diagonal_exist_pairs
from vae_model_v2 import DecoderOutput


class LossComponents(NamedTuple):
    # 合計だけでなく内訳を残すのは、特定の損失項目だけが暴走するケース(例: 勾配爆発)を、
    # 合計損失の値だけを見ていても気づけないため。呼び出し元はtotalをbackward()に使う
    total: torch.Tensor
    vertex_loss: torch.Tensor
    vertex_existence_loss: torch.Tensor
    start_pointer_loss: torch.Tensor
    end_pointer_loss: torch.Tensor
    stroke_offset_loss: torch.Tensor
    stroke_existence_loss: torch.Tensor
    kl_divergence: torch.Tensor
    crossing_loss: torch.Tensor
    angle_naturalness_loss: torch.Tensor


CROSSING_GATE_LOW = 0.08  # 交点パラメータt, uがこの範囲内ならストローク内部での交差とみなす(下限)。端点付近の接続点を除外する
CROSSING_GATE_HIGH = 0.92  # 同上(上限)
CROSSING_GATE_SHARPNESS = 40.0  # _interior_gateのsigmoidの急峻さ(大きいほど矩形窓に近づく)
CROSSING_DENOM_EPSILON = 1e-6  # 平行な弦同士でのゼロ除算回避
CROSSING_LOSS_WEIGHT = 0.3  # 独立ストローク表現でのweight sweepの結果を暫定採用する。頂点参照ベースでは
# 損失全体のスケールが変わるため最適値がそのまま通用する保証はないが、まず動かして傾向を見るための暫定値とする
ANGLE_NATURALNESS_LOSS_WEIGHT = 1.0  # 同上
MIN_DIRECTION_NORM = 1.0  # atan2(角度)の勾配は方向ベクトルの実スケールでのノルムに反比例して発散する
# (ノルム1e-4ではノルム1.0の場合の1万倍)。atan2に渡す前にノルムを底上げしても、atan2はスケール不変
# (同じ方向なら大きさによらず同じ角度)なため、合成した関数の勾配は元のまま変わらず解決にならない
# (検証済み)。そのため、ノルムがこの値未満のストロークは「角度が実質定義できない退化ケース」とみなし、
# atan2に渡す前に勾配的に無関係な固定方向へdetachして置き換え、角度自然さ損失の対象からも除外する
# (実データの最短ストロークより十分小さく、崩壊した頂点座標のような病的なケースのみを除外する値)


def pointer_loss(logits: torch.Tensor, target_index: torch.Tensor, stroke_existence: torch.Tensor) -> torch.Tensor:
    # logits: (B, stroke_count, vertex_count) -> cross_entropyのクラス次元(vertex_count)を
    # dim=1に持ってくるためtransposeする
    per_stroke_loss = F.cross_entropy(logits.transpose(1, 2), target_index, reduction="none")
    return (per_stroke_loss * stroke_existence).sum(dim=1).mean()


def _true_stroke_points(
    vertices: torch.Tensor, stroke_vertex_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # 正解の頂点インデックスから、標準化スケールのまま始点・終点座標を引く(交差判定はアフィン変換で
    # 不変なため標準化スケールのままでよく、再構成側の始点・終点座標と同じスケールで扱える)
    feature_dim = vertices.shape[-1]
    start_index = stroke_vertex_indices[..., 0:1].expand(-1, -1, feature_dim)
    end_index = stroke_vertex_indices[..., 1:2].expand(-1, -1, feature_dim)
    start = torch.gather(vertices, 1, start_index)
    end = torch.gather(vertices, 1, end_index)
    return start, end


def _pairwise_intersection_params(start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 各ストロークを弦(始点->終点)とみなし、全ペア(i, j)の交点パラメータt(iの弦上の位置)・
    # u(jの弦上の位置)をクラメルの公式(2元1次方程式)で解く。t, uはアフィン変換で不変
    # (2次元クロス積の分子・分母に現れるdetが約分される)ため、標準化スケールの座標をそのまま使ってよい
    direction = end - start  # (B, stroke_count, 2)
    d1 = direction.unsqueeze(2)  # (B, stroke_count, 1, 2) -- iの方向、jへブロードキャスト
    d2 = direction.unsqueeze(1)  # (B, 1, stroke_count, 2) -- jの方向、iへブロードキャスト
    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    # 平行な弦(denom≈0)はt, uが発散するが、_interior_gateで範囲外に押し出されるため実害はない
    denom = torch.where(denom.abs() < CROSSING_DENOM_EPSILON, torch.full_like(denom, CROSSING_DENOM_EPSILON), denom)

    diff = start.unsqueeze(1) - start.unsqueeze(2)  # p3 - p1, (B, stroke_count(i), stroke_count(j), 2)
    t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / denom
    u = (diff[..., 0] * d1[..., 1] - diff[..., 1] * d1[..., 0]) / denom
    return t, u


def _interior_gate(t: torch.Tensor) -> torch.Tensor:
    # t=CROSSING_GATE_LOW〜HIGHの範囲(ストローク内部)を、2つのsigmoidの積による
    # なめらかな矩形窓で近似する(線分交差判定は本質的に微分不可能なため)
    low = torch.sigmoid(CROSSING_GATE_SHARPNESS * (t - CROSSING_GATE_LOW))
    high = torch.sigmoid(CROSSING_GATE_SHARPNESS * (CROSSING_GATE_HIGH - t))
    return low * high


def _weighted_crossing_penalty(start: torch.Tensor, end: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # 再構成側の交差の強さ(0〜1の連続値)を求め、ペアごとの重みを掛けて損失にする
    t, u = _pairwise_intersection_params(start, end)
    crossing_strength = _interior_gate(t) * _interior_gate(u)
    return (weights * crossing_strength).sum(dim=(1, 2)).mean()


def _crossing_pair_weights(
    true_start: torch.Tensor, true_end: torch.Tensor, existence: torch.Tensor
) -> torch.Tensor:
    # 正解データで実際に交差しているペア(才のような正当な交差)は損失の対象外にする。交差の有無自体は
    # 不連続な事実であり勾配は不要なため、正解側の判定はハード閾値で行う。判定基準を正解側のみにするのは、
    # 再構成側は学習途中で崩れている可能性があるため
    with torch.no_grad():
        t, u = _pairwise_intersection_params(true_start, true_end)
        interior = (t > CROSSING_GATE_LOW) & (t < CROSSING_GATE_HIGH)
        interior &= (u > CROSSING_GATE_LOW) & (u < CROSSING_GATE_HIGH)
        not_crossing = (~interior).float()
    return off_diagonal_exist_pairs(existence) * not_crossing


def _compute_crossing_loss(
    true_start: torch.Tensor,
    true_end: torch.Tensor,
    recon_start: torch.Tensor,
    recon_end: torch.Tensor,
    existence: torch.Tensor,
) -> torch.Tensor:
    # 正解で交差していないペアについて、再構成側の交差の強さにペナルティを与える
    weights = _crossing_pair_weights(true_start, true_end, existence)
    return _weighted_crossing_penalty(recon_start, recon_end, weights)


def _direction_real(
    start: torch.Tensor, end: torch.Tensor, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> torch.Tensor:
    # atan2は実際のラジアン値でないと意味を持たないため、標準化された座標のまま角度を求めると、
    # x軸・y軸で標準化の標準偏差が異なることにより縦横比が歪み、実際の見た目の角度とは異なる値になる。
    # 実スケールに戻してから計算する(交差判定のアフィン不変性とは異なり、角度はアフィン変換で不変ではない)
    start_real = start * vertex_std + vertex_mean
    end_real = end * vertex_std + vertex_mean
    return end_real - start_real


def _safe_angle(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # ノルムがMIN_DIRECTION_NORM未満の退化した方向ベクトルは、atan2に渡す前に勾配的に無関係な
    # 固定方向(1, 0)へ置き換える(torch.whereのbackwardでは選ばれなかった側の勾配は伝播しないため、
    # 元のdirectionへは勾配が流れない)。戻り値のwell_definedで、そのストロークを損失計算からも除外する
    well_defined = direction.norm(dim=-1) >= MIN_DIRECTION_NORM
    safe_direction = torch.where(well_defined.unsqueeze(-1), direction, direction.new_tensor([1.0, 0.0]))
    angle = torch.atan2(safe_direction[..., 1], safe_direction[..., 0])
    return angle, well_defined.float()


def _compute_angle_naturalness_loss(
    true_start: torch.Tensor,
    true_end: torch.Tensor,
    recon_start: torch.Tensor,
    recon_end: torch.Tensor,
    existence: torch.Tensor,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
    angle_gmm: AngleGMM,
) -> torch.Tensor:
    # 正解の角度自体が非典型的な場合(実データにも一定数存在する)に再構成をそこから引き離してしまわないよう、
    # 絶対的な自然さではなく正解と比較した相対的な自然さで評価する。正解通りに再構成できていれば
    # (正解が非典型的でも)ペナルティが発生しないようにする
    true_angle, true_well_defined = _safe_angle(_direction_real(true_start, true_end, vertex_mean, vertex_std))
    recon_angle, recon_well_defined = _safe_angle(_direction_real(recon_start, recon_end, vertex_mean, vertex_std))
    true_log_density = angle_log_density(true_angle, angle_gmm)
    recon_log_density = angle_log_density(recon_angle, angle_gmm)
    penalty = torch.clamp(true_log_density - recon_log_density, min=0.0)
    mask = existence * true_well_defined * recon_well_defined
    return (penalty * mask).sum(dim=1).mean()


def compute_loss(
    vertices: torch.Tensor,
    vertex_existence: torch.Tensor,
    stroke_vertex_indices: torch.Tensor,
    stroke_offsets: torch.Tensor,
    stroke_existence: torch.Tensor,
    decoder_output: DecoderOutput,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
    angle_gmm: AngleGMM,
) -> LossComponents:
    # 特徴量・スロット方向は和、バッチ方向は平均を取る(sum→batch mean)。
    # 要素方向で平均を取るとKLダイバージェンスに対して再構成損失が相対的に小さくなり、posterior collapseを起こしやすくなるため避ける
    vertex_mask = vertex_existence.unsqueeze(-1)
    vertex_loss = (((decoder_output.vertex_features - vertices) ** 2) * vertex_mask).sum(dim=(1, 2)).mean()
    vertex_existence_loss = (
        F.binary_cross_entropy_with_logits(decoder_output.vertex_existence_logits, vertex_existence, reduction="none")
        .sum(dim=1)
        .mean()
    )

    # ポインタの参照先はcross entropyで直接教師する(座標自体へのMSEは加えない)。座標の正しさは、
    # ポインタが正解頂点に集中しさえすれば頂点座標MSEを通じて間接的に保証される設計のため
    start_pointer_loss = pointer_loss(
        decoder_output.start_pointer_logits, stroke_vertex_indices[..., 0], stroke_existence
    )
    end_pointer_loss = pointer_loss(
        decoder_output.end_pointer_logits, stroke_vertex_indices[..., 1], stroke_existence
    )

    stroke_offset_mask = stroke_existence.unsqueeze(-1)
    stroke_offset_loss = (
        ((decoder_output.stroke_offsets - stroke_offsets) ** 2) * stroke_offset_mask
    ).sum(dim=(1, 2)).mean()
    stroke_existence_loss = (
        F.binary_cross_entropy_with_logits(
            decoder_output.stroke_existence_logits, stroke_existence, reduction="none"
        )
        .sum(dim=1)
        .mean()
    )

    kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    true_start, true_end = _true_stroke_points(vertices, stroke_vertex_indices)
    crossing_loss = _compute_crossing_loss(
        true_start, true_end, decoder_output.start_points, decoder_output.end_points, stroke_existence
    )
    angle_naturalness_loss = _compute_angle_naturalness_loss(
        true_start, true_end, decoder_output.start_points, decoder_output.end_points, stroke_existence,
        vertex_mean, vertex_std, angle_gmm,
    )

    total = (
        vertex_loss
        + vertex_existence_loss
        + start_pointer_loss
        + end_pointer_loss
        + stroke_offset_loss
        + stroke_existence_loss
        + beta * kl_divergence
        + CROSSING_LOSS_WEIGHT * crossing_loss
        + ANGLE_NATURALNESS_LOSS_WEIGHT * angle_naturalness_loss
    )
    # kl_divergence同様、crossing_loss・angle_naturalness_lossも重み乗算前の生の値を残す
    # (重み(CROSSING_LOSS_WEIGHT等)は呼び出し元ではなくこのファイル内で完結する値のため)
    return LossComponents(
        total, vertex_loss, vertex_existence_loss, start_pointer_loss, end_pointer_loss,
        stroke_offset_loss, stroke_existence_loss, kl_divergence, crossing_loss, angle_naturalness_loss,
    )
