#!/usr/bin/env python3
from typing import NamedTuple

import torch
import torch.nn.functional as F

from vae_crossing_geometry_v2 import (
    CROSSING_GATE_HIGH,
    CROSSING_GATE_LOW,
    interior_gate,
    pairwise_segment_intersection_params,
)
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
    vertex_repulsion_loss: torch.Tensor


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

VERTEX_REPULSION_THRESHOLD = 4.0  # 端点接続判定の閾値と同じ(extract_stroke_features_v2.CONNECTION_THRESHOLD)
VERTEX_REPULSION_LOSS_WEIGHT = 1.0  # 交差抑制損失が「無関係な頂点同士を寄せる」ことで過剰接続
# (3本以上合流)を悪化させる副作用への対応として追加。ablationで、頂点参照数ベースの代替案
# (多重参照抑制損失)より明確に効果が高いことを確認済み(index単位ではなく座標単位でペナルティを
# 与えるため、根本原因(座標単位での頂点の密集)に直接効く)。既存コードに対応物が存在しない
# 新規の損失のため暫定値とする


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


def _fold_point(
    start: torch.Tensor,
    end: torch.Tensor,
    offset: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
) -> torch.Tensor:
    # vae_eval_common_v2.stroke_curvesが定義するベジェ制御点(弦の中点をoffset(実スケール)だけ
    # ずらした点)を、start/endと同じ標準化頂点座標空間で求める。offsetはvertexとは別の正規化
    # (stroke_offset_mean/std)を持つため、実スケールに変換してからvertex_stdで割ることで
    # 頂点座標と同じアフィン変換後の空間に揃える(この空間内であれば交差判定のアフィン不変性が成り立つ)
    offset_real = offset * stroke_offset_std + stroke_offset_mean
    return (start + end) / 2 + offset_real / vertex_std


def _segment_global_position(local_param: torch.Tensor, segment_index: int) -> torch.Tensor:
    # 折れ線の線分(0: 始点->制御点、1: 制御点->終点)上のローカルなt(0〜1)を、ストローク全体を
    # 弦一本とみなした場合と同じ0〜1スケールでの位置に変換する。制御点(曲線内部の点)がローカル
    # パラメータの端(区間0のt=1、区間1のt=0)に来るため、変換せずinterior_gateにそのまま渡すと
    # 曲線内部の点が本来の始点・終点と誤って同じ扱いで除外されてしまう
    return (segment_index + local_param) / 2


def _folded_segment_global_positions(
    start: torch.Tensor, end: torch.Tensor, mid: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # ストロークを2線分(0: 始点->制御点、1: 制御点->終点)の折れ線として近似し、4通りの線分の
    # 組み合わせ(前半-前半、前半-後半、後半-前半、後半-後半)それぞれについて、交点パラメータを
    # ストローク全体スケールの位置(t_global, u_global)に変換して返す
    segments = [(start, mid), (mid, end)]
    positions = []
    for seg_i_index, (seg_i_start, seg_i_end) in enumerate(segments):
        for seg_j_index, (seg_j_start, seg_j_end) in enumerate(segments):
            t, u = pairwise_segment_intersection_params(seg_i_start, seg_i_end, seg_j_start, seg_j_end)
            positions.append(
                (_segment_global_position(t, seg_i_index), _segment_global_position(u, seg_j_index))
            )
    return positions


def _folded_crossing_mask(start: torch.Tensor, end: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    # 4通りの線分組み合わせを、ハード閾値・論理OR(いずれか1組でも交差していれば全体として
    # 交差しているとみなす、勾配不要)で統合する。正解側の判定用
    crossing = None
    for t_global, u_global in _folded_segment_global_positions(start, end, mid):
        interior = (
            (t_global > CROSSING_GATE_LOW) & (t_global < CROSSING_GATE_HIGH)
            & (u_global > CROSSING_GATE_LOW) & (u_global < CROSSING_GATE_HIGH)
        )
        crossing = interior if crossing is None else (crossing | interior)
    return crossing


def _folded_crossing_strength(start: torch.Tensor, end: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    # _folded_crossing_maskと同じ4通りの組み合わせを、交差強度(0〜1の連続値)の確率的OR
    # (ハード閾値・論理ORの微分可能な近似)で統合する。再構成側(勾配が必要)用
    not_crossing = 1.0
    for t_global, u_global in _folded_segment_global_positions(start, end, mid):
        not_crossing = not_crossing * (1 - interior_gate(t_global) * interior_gate(u_global))
    return 1 - not_crossing


def _crossing_pair_weights(
    true_start: torch.Tensor,
    true_end: torch.Tensor,
    true_offset: torch.Tensor,
    existence: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
) -> torch.Tensor:
    # 正解データで実際に交差しているペア(才のような正当な交差)は損失の対象外にする。交差の有無自体は
    # 不連続な事実であり勾配は不要なため、正解側の判定はハード閾値で行う。判定基準を正解側のみにするのは、
    # 再構成側は学習途中で崩れている可能性があるため
    with torch.no_grad():
        mid = _fold_point(true_start, true_end, true_offset, vertex_std, stroke_offset_mean, stroke_offset_std)
        crossing = _folded_crossing_mask(true_start, true_end, mid)
        not_crossing = (~crossing).float()
    return off_diagonal_exist_pairs(existence) * not_crossing


def _weighted_crossing_penalty(
    start: torch.Tensor,
    end: torch.Tensor,
    offset: torch.Tensor,
    weights: torch.Tensor,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
) -> torch.Tensor:
    # 再構成側の交差の強さ(0〜1の連続値)を求め、ペアごとの重みを掛けて損失にする。ペアの片方でも
    # 方向ベクトルが退化していれば、そのペアは交差判定自体が意味をなさないため対象から除外する
    mid = _fold_point(start, end, offset, vertex_std, stroke_offset_mean, stroke_offset_std)
    crossing_strength = _folded_crossing_strength(start, end, mid)
    well_defined = _well_defined_mask(_direction_real(start, end, vertex_mean, vertex_std))
    pair_well_defined = well_defined.unsqueeze(2) * well_defined.unsqueeze(1)
    return (weights * pair_well_defined * crossing_strength).sum(dim=(1, 2)).mean()


def _compute_crossing_loss(
    true_start: torch.Tensor,
    true_end: torch.Tensor,
    true_offset: torch.Tensor,
    recon_start: torch.Tensor,
    recon_end: torch.Tensor,
    recon_offset: torch.Tensor,
    existence: torch.Tensor,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
) -> torch.Tensor:
    # 正解で交差していないペアについて、再構成側の交差の強さにペナルティを与える
    weights = _crossing_pair_weights(
        true_start, true_end, true_offset, existence, vertex_std, stroke_offset_mean, stroke_offset_std
    )
    return _weighted_crossing_penalty(
        recon_start, recon_end, recon_offset, weights, vertex_mean, vertex_std, stroke_offset_mean, stroke_offset_std
    )


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


def _pairwise_vertex_distance_real(
    vertices: torch.Tensor, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> torch.Tensor:
    # 全頂点ペアの実スケールでのユークリッド距離。(B, vertex_count, vertex_count)
    vertices_real = vertices * vertex_std + vertex_mean
    diff = vertices_real.unsqueeze(2) - vertices_real.unsqueeze(1)
    return diff.norm(dim=-1)


def _vertex_repulsion_pair_weights(
    true_vertices: torch.Tensor, vertex_existence: torch.Tensor, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> torch.Tensor:
    # 正解データで実際に離れている頂点ペアのみを対象にする(_crossing_pair_weightsと同じ設計:
    # 正解側の判定は勾配不要のハード閾値で行う)。目・日のような格子状の部首で正解上も近い頂点ペアは
    # ここで対象から除外されるため、正当な近さを壊すリスクを個別の閾値調整に頼らず構造的に避けられる
    with torch.no_grad():
        true_distance = _pairwise_vertex_distance_real(true_vertices, vertex_mean, vertex_std)
        far_in_truth = (true_distance > VERTEX_REPULSION_THRESHOLD).float()
    return off_diagonal_exist_pairs(vertex_existence) * far_in_truth


def _compute_vertex_repulsion_loss(
    true_vertices: torch.Tensor,
    recon_vertices: torch.Tensor,
    vertex_existence: torch.Tensor,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
) -> torch.Tensor:
    # 正解で離れているはずの頂点ペアが、再構成でVERTEX_REPULSION_THRESHOLD未満まで近づいたら
    # ペナルティを与える。crossing/angleと異なりストロークの端点(points)ではなく頂点スロット
    # (vertex_features)自体を直接動かす損失であり、ポインタの選択を経由しないためdetachは不要
    weights = _vertex_repulsion_pair_weights(true_vertices, vertex_existence, vertex_mean, vertex_std)
    recon_distance = _pairwise_vertex_distance_real(recon_vertices, vertex_mean, vertex_std)
    penalty = torch.clamp(VERTEX_REPULSION_THRESHOLD - recon_distance, min=0.0)
    return (weights * penalty).sum(dim=(1, 2)).mean()


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
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
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
        true_start, true_end, stroke_offsets,
        decoder_output.start_points, decoder_output.end_points, decoder_output.stroke_offsets,
        stroke_existence, vertex_mean, vertex_std, stroke_offset_mean, stroke_offset_std,
    )
    losses["angle_naturalness_loss"] = _compute_angle_naturalness_loss(
        true_start, true_end, decoder_output.start_points, decoder_output.end_points, stroke_existence,
        vertex_mean, vertex_std, angle_gmm,
    )
    losses["min_length_loss"] = _compute_min_length_loss(
        decoder_output.start_points, decoder_output.end_points, stroke_existence, vertex_mean, vertex_std
    )
    losses["vertex_repulsion_loss"] = _compute_vertex_repulsion_loss(
        vertices, decoder_output.vertex_features, vertex_existence, vertex_mean, vertex_std
    )

    # ここに列挙のない損失(vertex_loss等)は暗黙的に重み1.0として扱う
    weights = {
        "kl_divergence": beta,
        "crossing_loss": CROSSING_LOSS_WEIGHT,
        "angle_naturalness_loss": ANGLE_NATURALNESS_LOSS_WEIGHT,
        "min_length_loss": MIN_LENGTH_LOSS_WEIGHT,
        "vertex_repulsion_loss": VERTEX_REPULSION_LOSS_WEIGHT,
    }
    total = sum(weights.get(name, 1.0) * value for name, value in losses.items())
    return LossComponents(total=total, **losses)
