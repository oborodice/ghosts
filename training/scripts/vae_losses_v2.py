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
    min_length_loss: torch.Tensor


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
MIN_LENGTH_LOSS_WEIGHT = 1.0  # 交差抑制損失・角度自然さ損失はいずれも、MIN_DIRECTION_NORM未満のペア/
# ストロークを損失の計算対象から除外する(退化ケースでの数値破綻を避けるため)。この除外は副作用として、
# ストロークを縮めて対象から外れること自体が損失を下げる手段になってしまう(交差抑制損失で実測済み)。
# この抜け道は個々の損失の除外条件を直そうとするより、ストローク自体の縮小に独立してペナルティを
# 与える方が両方の損失に共通して効く。既存コードに対応物が存在しない新規の損失のため暫定値とする


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


def _direction_real(
    start: torch.Tensor, end: torch.Tensor, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> torch.Tensor:
    # atan2は実際のラジアン値でないと意味を持たないため、標準化された座標のまま角度を求めると、
    # x軸・y軸で標準化の標準偏差が異なることにより縦横比が歪み、実際の見た目の角度とは異なる値になる。
    # 実スケールに戻してから計算する(交差判定のアフィン不変性とは異なり、角度はアフィン変換で不変ではない)
    start_real = start * vertex_std + vertex_mean
    end_real = end * vertex_std + vertex_mean
    return end_real - start_real


def _well_defined_mask(direction: torch.Tensor) -> torch.Tensor:
    # 実スケールでの方向ベクトルのノルムがMIN_DIRECTION_NORM未満のストロークは、始点・終点が実質同じ
    # 場所にある退化したケースとみなす。角度(atan2の勾配爆発)だけでなく交差判定(崩壊したストローク集団は
    # 実際の見た目によらずcrossing_strengthが小さい一定値になる、実測で確認済みの抜け道)も、この閾値で
    # 退化ストロークを除外する必要がある
    return (direction.norm(dim=-1) >= MIN_DIRECTION_NORM).float()


def _pairwise_intersection_params(start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 各ストロークを弦(始点->終点)とみなし、全ペア(i, j)の交点パラメータt(iの弦上の位置)・
    # u(jの弦上の位置)をクラメルの公式(2元1次方程式)で解く。t, uはアフィン変換で不変
    # (2次元クロス積の分子・分母に現れるdetが約分される)ため、標準化スケールの座標をそのまま使ってよい
    # 注意: 実際に描画される曲線の曲がり(offset_x/offset_y)はここでは一切考慮されない。そのため
    # 交差抑制損失が交差を避けるために動かせる連続的なレバーは始点・終点の座標(頂点位置)のみであり、
    # 「曲げて避ける」という選択肢は与えられていない。この非対称性により、本来無関係な2頂点を
    # 近づけて弦同士を交差させないようにする(結果として過剰接続を助長する)という副作用が起こりうる
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


def _weighted_crossing_penalty(
    start: torch.Tensor, end: torch.Tensor, weights: torch.Tensor, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> torch.Tensor:
    # 再構成側の交差の強さ(0〜1の連続値)を求め、ペアごとの重みを掛けて損失にする。ペアの片方でも
    # 方向ベクトルが退化していれば、そのペアは交差判定自体が意味をなさないため対象から除外する
    t, u = _pairwise_intersection_params(start, end)
    crossing_strength = _interior_gate(t) * _interior_gate(u)
    well_defined = _well_defined_mask(_direction_real(start, end, vertex_mean, vertex_std))
    pair_well_defined = well_defined.unsqueeze(2) * well_defined.unsqueeze(1)
    return (weights * pair_well_defined * crossing_strength).sum(dim=(1, 2)).mean()


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
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
) -> torch.Tensor:
    # 正解で交差していないペアについて、再構成側の交差の強さにペナルティを与える
    weights = _crossing_pair_weights(true_start, true_end, existence)
    return _weighted_crossing_penalty(recon_start, recon_end, weights, vertex_mean, vertex_std)


def _safe_angle(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # ノルムがMIN_DIRECTION_NORM未満の退化した方向ベクトルは、atan2に渡す前に勾配的に無関係な
    # 固定方向(1, 0)へ置き換える(torch.whereのbackwardでは選ばれなかった側の勾配は伝播しないため、
    # 元のdirectionへは勾配が流れない)。戻り値のwell_definedで、そのストロークを損失計算からも除外する
    well_defined = _well_defined_mask(direction)
    safe_direction = torch.where(well_defined.bool().unsqueeze(-1), direction, direction.new_tensor([1.0, 0.0]))
    angle = torch.atan2(safe_direction[..., 1], safe_direction[..., 0])
    return angle, well_defined


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


def _compute_min_length_loss(
    start: torch.Tensor,
    end: torch.Tensor,
    existence: torch.Tensor,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
) -> torch.Tensor:
    # ストロークの方向ベクトルの実スケールでのノルムがMIN_DIRECTION_NORM未満になるほどペナルティを
    # 与える。MIN_DIRECTION_NORM以上では常に0(実データの最短ストロークより十分小さい値のため、
    # 正当な短いストロークには影響しない)。退化域では他の損失(交差抑制・角度自然さ)がこのストロークを
    # 対象から除外しており押し戻す力を持たないため、この損失だけがノルムを増やす方向に勾配を持つ
    norm = _direction_real(start, end, vertex_mean, vertex_std).norm(dim=-1)
    penalty = torch.clamp(MIN_DIRECTION_NORM - norm, min=0.0)
    return (penalty * existence).sum(dim=1).mean()


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
    #
    # lossesは重み乗算前の生の値(LossComponentsにそのまま格納する値)。個々の損失を追加するたびに
    # (a)このdictへの1行(b)重みが1.0以外ならweightsへの1行、の2箇所を触るだけで済むようにするため、
    # totalの合計式・LossComponentsへの詰め替えを手書きしない構成にしている
    losses: dict[str, torch.Tensor] = {}

    vertex_mask = vertex_existence.unsqueeze(-1)
    losses["vertex_loss"] = (((decoder_output.vertex_features - vertices) ** 2) * vertex_mask).sum(dim=(1, 2)).mean()
    losses["vertex_existence_loss"] = (
        F.binary_cross_entropy_with_logits(decoder_output.vertex_existence_logits, vertex_existence, reduction="none")
        .sum(dim=1)
        .mean()
    )

    # ポインタの参照先はcross entropyで直接教師する(座標自体へのMSEは加えない)。座標の正しさは、
    # ポインタが正解頂点に集中しさえすれば頂点座標MSEを通じて間接的に保証される設計のため
    losses["start_pointer_loss"] = pointer_loss(
        decoder_output.start_pointer_logits, stroke_vertex_indices[..., 0], stroke_existence
    )
    losses["end_pointer_loss"] = pointer_loss(
        decoder_output.end_pointer_logits, stroke_vertex_indices[..., 1], stroke_existence
    )

    stroke_offset_mask = stroke_existence.unsqueeze(-1)
    losses["stroke_offset_loss"] = (
        ((decoder_output.stroke_offsets - stroke_offsets) ** 2) * stroke_offset_mask
    ).sum(dim=(1, 2)).mean()
    losses["stroke_existence_loss"] = (
        F.binary_cross_entropy_with_logits(
            decoder_output.stroke_existence_logits, stroke_existence, reduction="none"
        )
        .sum(dim=1)
        .mean()
    )

    losses["kl_divergence"] = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    true_start, true_end = _true_stroke_points(vertices, stroke_vertex_indices)
    losses["crossing_loss"] = _compute_crossing_loss(
        true_start, true_end, decoder_output.start_points, decoder_output.end_points, stroke_existence,
        vertex_mean, vertex_std,
    )
    losses["angle_naturalness_loss"] = _compute_angle_naturalness_loss(
        true_start, true_end, decoder_output.start_points, decoder_output.end_points, stroke_existence,
        vertex_mean, vertex_std, angle_gmm,
    )
    losses["min_length_loss"] = _compute_min_length_loss(
        decoder_output.start_points, decoder_output.end_points, stroke_existence, vertex_mean, vertex_std
    )

    # ここに列挙のない損失(vertex_loss等)は暗黙的に重み1.0として扱う
    weights = {
        "kl_divergence": beta,
        "crossing_loss": CROSSING_LOSS_WEIGHT,
        "angle_naturalness_loss": ANGLE_NATURALNESS_LOSS_WEIGHT,
        "min_length_loss": MIN_LENGTH_LOSS_WEIGHT,
    }
    total = sum(weights.get(name, 1.0) * value for name, value in losses.items())
    return LossComponents(total=total, **losses)
