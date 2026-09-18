#!/usr/bin/env python3
# ストロークを線分として扱う交差判定の幾何計算。再構成側(vae_losses_v2.py)・合成z側
# (vae_synthetic_losses_v2.py)は2線分の折れ線近似、診断側(report_generation_stats_v2.py)は
# より精密な12分割ポリライン近似で、それぞれ同じ線分交差判定の核(pairwise_segment_intersection_params・
# _hard_crossing_interior)を使うため、共有モジュールに切り出している
import torch

CROSSING_GATE_LOW = 0.08  # 交点パラメータt, uがこの範囲内ならストローク内部での交差とみなす(下限)。端点付近の接続点を除外する
CROSSING_GATE_HIGH = 0.92  # 同上(上限)
CROSSING_GATE_SHARPNESS = 40.0  # interior_gateのsigmoidの急峻さ(大きいほど矩形窓に近づく)
CROSSING_DENOM_EPSILON = 1e-6  # 平行な線分同士でのゼロ除算回避


def pairwise_segment_intersection_params(
    start1: torch.Tensor, end1: torch.Tensor, start2: torch.Tensor, end2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 線分1(start1->end1、ストロークiの区間)と線分2(start2->end2、ストロークjの区間)の
    # 全ペア(i, j)について、交点パラメータt(線分1上の位置)・u(線分2上の位置)をクラメルの公式
    # (2元1次方程式)で解く。t, uはアフィン変換で不変(2次元クロス積の分子・分母に現れるdetが
    # 約分される)ため、start1/end1とstart2/end2が同じアフィン変換後の空間(標準化頂点座標空間)に
    # あれば、標準化スケールの座標をそのまま使ってよい
    d1 = (end1 - start1).unsqueeze(2)  # (B, stroke_count, 1, 2) -- iの方向、jへブロードキャスト
    d2 = (end2 - start2).unsqueeze(1)  # (B, 1, stroke_count, 2) -- jの方向、iへブロードキャスト
    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    # 平行な線分(denom≈0)は、単に別々の場所にあるだけならt, uが確実に発散しinterior_gateで
    # 範囲外に押し出されるため実害はない。しかし2線分が完全に同一直線上へ重なる場合は分子側の量も
    # 同時にゼロへ近づく0/0に近い不定形になり、epsilonへの置き換えだけでは丸め誤差次第で
    # 実質ランダムな値になってしまう(実測で、この不定形が多数のペアで同時発生し偽の交差を
    # 大量発生させ、学習が破壊されることを確認済み)。degenerateを呼び出し側に返し、
    # このt, uの値そのものを判定に使わせないことで安全性を保証する
    degenerate = denom.abs() < CROSSING_DENOM_EPSILON
    denom = torch.where(degenerate, torch.full_like(denom, CROSSING_DENOM_EPSILON), denom)

    diff = start2.unsqueeze(1) - start1.unsqueeze(2)  # p3 - p1, (B, stroke_count(i), stroke_count(j), 2)
    t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / denom
    u = (diff[..., 0] * d1[..., 1] - diff[..., 1] * d1[..., 0]) / denom
    return t, u, degenerate


def _hard_crossing_interior(
    t: torch.Tensor, u: torch.Tensor, t_global: torch.Tensor, u_global: torch.Tensor, degenerate: torch.Tensor
) -> torch.Tensor:
    # ハード閾値の交差判定(勾配不要)。t_global, u_globalがストローク内部(端点付近の接続点を除く)に
    # あり、かつt, uが実際の線分の範囲[0, 1]内にある(_segment_membership_gateの説明を参照、
    # 折れ線・ポリラインで範囲外のtがグローバル位置だけの判定をすり抜けるのを防ぐ)ことを要求する。
    # degenerateなペア(2線分がほぼ同一直線上に重なる、t, uの値が信頼できない)は判定不能として除外する
    return (
        (t_global > CROSSING_GATE_LOW) & (t_global < CROSSING_GATE_HIGH)
        & (u_global > CROSSING_GATE_LOW) & (u_global < CROSSING_GATE_HIGH)
        & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        & ~degenerate
    )


def interior_gate(t: torch.Tensor) -> torch.Tensor:
    # t=CROSSING_GATE_LOW〜HIGHの範囲(ストローク内部)を、2つのsigmoidの積による
    # なめらかな矩形窓で近似する(線分交差判定は本質的に微分不可能なため)
    low = torch.sigmoid(CROSSING_GATE_SHARPNESS * (t - CROSSING_GATE_LOW))
    high = torch.sigmoid(CROSSING_GATE_SHARPNESS * (CROSSING_GATE_HIGH - t))
    return low * high


def _segment_membership_gate(t: torch.Tensor) -> torch.Tensor:
    # tが実際の線分の範囲[0, 1]内にあるかどうかを、interior_gateと同じ発想のsigmoidの積で
    # なめらかに近似する(0〜1の範囲そのものが対象なのでinterior_gateのCROSSING_GATE_LOW/HIGHの
    # ような余白は取らない)。pairwise_segment_intersection_paramsが返すt, uは2直線を無限に
    # 延長した場合の交点パラメータであり、線分の範囲外(t<0またはt>1)でも数値としては返ってくる。
    # 複数線分の折れ線では、範囲外のtをそのままストローク全体のグローバル位置に変換すると、
    # 別の区間の内部位置にたまたま折り返されてしまい、実際には交わっていない線分ペアを交差ありと
    # 誤判定することがある(実測で確認済み)。単一線分(弦のみ)の判定ではグローバル位置=ローカル
    # 位置そのものでありCROSSING_GATE_LOW/HIGH自体が範囲チェックを兼ねるため問題にならないが、
    # 複数線分ではグローバル位置の判定だけでは不十分で、このローカルな範囲チェックが別途必要になる
    low = torch.sigmoid(CROSSING_GATE_SHARPNESS * t)
    high = torch.sigmoid(CROSSING_GATE_SHARPNESS * (1 - t))
    return low * high


def fold_point(
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
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    # ストロークを2線分(0: 始点->制御点、1: 制御点->終点)の折れ線として近似し、4通りの線分の
    # 組み合わせ(前半-前半、前半-後半、後半-前半、後半-後半)それぞれについて、ローカルな交点
    # パラメータ(t, u)と、それをストローク全体スケールの位置に変換した値(t_global, u_global)の
    # 両方を返す。呼び出し側は両方を使う必要がある(t, uが線分の範囲[0, 1]内にあるかの判定と、
    # t_global, u_globalがストローク内部(端点付近の接続点を除く)にあるかの判定は別物、
    # _segment_membership_gate参照)。degenerateはpairwise_segment_intersection_params参照
    # (2線分がほぼ同一直線上に重なる退化ケース)
    segments = [(start, mid), (mid, end)]
    positions = []
    for seg_i_index, (seg_i_start, seg_i_end) in enumerate(segments):
        for seg_j_index, (seg_j_start, seg_j_end) in enumerate(segments):
            t, u, degenerate = pairwise_segment_intersection_params(seg_i_start, seg_i_end, seg_j_start, seg_j_end)
            positions.append(
                (t, u, _segment_global_position(t, seg_i_index), _segment_global_position(u, seg_j_index), degenerate)
            )
    return positions


def folded_crossing_mask(start: torch.Tensor, end: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    # 4通りの線分組み合わせを、ハード閾値・論理OR(いずれか1組でも交差していれば全体として
    # 交差しているとみなす、勾配不要)で統合する。正解側の判定用。degenerate(2線分が同一直線上に
    # 重なる退化ケース、t, uの値が信頼できない)なペアは判定不能として交差から除外する
    crossing = None
    for t, u, t_global, u_global, degenerate in _folded_segment_global_positions(start, end, mid):
        interior = _hard_crossing_interior(t, u, t_global, u_global, degenerate)
        crossing = interior if crossing is None else (crossing | interior)
    return crossing


def folded_crossing_strength(start: torch.Tensor, end: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    # folded_crossing_maskと同じ4通りの組み合わせを、交差強度(0〜1の連続値)の確率的OR
    # (ハード閾値・論理ORの微分可能な近似)で統合する。勾配が必要な側(再構成・合成z側の両方)用。
    # degenerateなペアは交差強度を強制的に0にする(folded_crossing_maskと同じ理由)
    not_crossing = 1.0
    for t, u, t_global, u_global, degenerate in _folded_segment_global_positions(start, end, mid):
        on_segment = _segment_membership_gate(t) * _segment_membership_gate(u)
        pair_strength = interior_gate(t_global) * interior_gate(u_global) * on_segment * (~degenerate).float()
        not_crossing = not_crossing * (1 - pair_strength)
    return 1 - not_crossing


def bezier_polyline_points(
    start: torch.Tensor, end: torch.Tensor, control: torch.Tensor, segment_count: int
) -> torch.Tensor:
    # 2次ベジェ曲線B(t) = (1-t)^2*start + 2t(1-t)*control + t^2*endを、t=0, 1/n, ..., 1の
    # (segment_count+1)点でサンプリングする。vae_eval_common.bezier_polylineと同じ式(diagnostic側の
    # 精密な曲線近似)をバッチ処理向けにベクトル化したもの。fold_pointが返す「中点」は制御点そのもの
    # (曲線上の点ではない)であり、ここでのcontrolも同様に制御点を渡す必要がある
    t = torch.linspace(0, 1, segment_count + 1, device=start.device, dtype=start.dtype)
    t = t.view(1, 1, -1, 1)  # (1, 1, segment_count+1, 1) -- バッチ・ストローク次元へブロードキャスト
    start_ = start.unsqueeze(2)  # (B, stroke_count, 1, 2)
    end_ = end.unsqueeze(2)
    control_ = control.unsqueeze(2)
    return (1 - t) ** 2 * start_ + 2 * t * (1 - t) * control_ + t**2 * end_


def polyline_crossing_points(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # vae_eval_common.classify_crossingsの交差判定(ハード閾値、勾配不要)を、pointsで表される
    # 全ストロークのポリライン(bezier_polyline_pointsが返す(B, stroke_count, segment_count+1, 2))に
    # 対してベクトル化して計算する。classify_crossingsは1サンプル・1ペアごとにPythonのfor文で
    # segment_count×segment_count通りの線分交差を判定しているが、これをバッチ全体・全ストローク
    # ペアに対して一度に計算する。診断・報告専用(学習ループのcrossing_lossは_folded_*系列を使う)。
    # 交差の実座標(vae_eval_common._find_curve_intersectionsが返す点と同じもの、3本以上合流の
    # クラスタリングに使う)もあわせて返す。1ペアにつき最初に見つかった(a, b)の組み合わせの交点を
    # 採用する(_find_curve_intersectionsの「found=True; break」と同じ、a→bの順で最初に見つかった
    # ものを採用するセマンティクスに合わせている)
    segment_count = points.shape[2] - 1
    batch_size, stroke_count = points.shape[0], points.shape[1]
    crossing = torch.zeros(batch_size, stroke_count, stroke_count, dtype=torch.bool, device=points.device)
    point = torch.zeros(batch_size, stroke_count, stroke_count, 2, dtype=points.dtype, device=points.device)
    for a in range(segment_count):
        seg_i_start, seg_i_end = points[:, :, a], points[:, :, a + 1]
        for b in range(segment_count):
            seg_j_start, seg_j_end = points[:, :, b], points[:, :, b + 1]
            t, u, degenerate = pairwise_segment_intersection_params(seg_i_start, seg_i_end, seg_j_start, seg_j_end)
            t_global = (a + t) / segment_count
            u_global = (b + u) / segment_count
            interior = _hard_crossing_interior(t, u, t_global, u_global, degenerate)
            newly_found = interior & ~crossing
            segment_i_direction = (seg_i_end - seg_i_start).unsqueeze(2)  # (B, stroke_count, 1, 2)
            candidate_point = seg_i_start.unsqueeze(2) + t.unsqueeze(-1) * segment_i_direction
            point = torch.where(newly_found.unsqueeze(-1), candidate_point, point)
            crossing = crossing | interior
    return crossing, point


def polyline_diagonal_involved_mask(
    crossing: torch.Tensor, angle: torch.Tensor, axis_tolerance_deg: float
) -> torch.Tensor:
    # vae_eval_common.classify_crossingsの"diagonal_involved"(交差ペアのうち、少なくとも一方が
    # 軸方向(水平・垂直)でないもの)をベクトル化する。vae_eval_common.is_axis_alignedと同じ判定式
    deg = torch.rad2deg(angle) % 90
    axis_aligned = (deg < axis_tolerance_deg) | (deg > (90 - axis_tolerance_deg))
    both_axis = axis_aligned.unsqueeze(2) & axis_aligned.unsqueeze(1)
    return crossing & ~both_axis


def folded_crossing_per_sample_total(
    start: torch.Tensor,
    end: torch.Tensor,
    offset: torch.Tensor,
    existence: torch.Tensor,
    well_defined: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
) -> torch.Tensor:
    # ストロークをoffset付きの折れ線として近似した交差数を、サンプルごとに合計する。上三角(i<j)のみを
    # 対象にして各ペアを1回だけ数え、両ストロークが存在し(existence)かつ両方が退化していない
    # (well_defined)ペアに限定する。crossingの頻度を集計統計として扱う損失・目標値の計算
    # (vae_data_v2._compute_crossing_targets、vae_synthetic_losses_v2._compute_synthetic_crossing_loss)
    # で共通して使う
    mid = fold_point(start, end, offset, vertex_std, stroke_offset_mean, stroke_offset_std)
    strength = folded_crossing_strength(start, end, mid)

    stroke_count = strength.shape[1]
    upper_triangle = torch.triu(
        torch.ones(stroke_count, stroke_count, dtype=torch.bool, device=strength.device), diagonal=1
    ).float()
    pair_exists = existence.unsqueeze(2) * existence.unsqueeze(1)
    pair_well_defined = well_defined.unsqueeze(2) * well_defined.unsqueeze(1)
    mask = pair_exists * pair_well_defined * upper_triangle.unsqueeze(0)
    return (strength * mask).sum(dim=(1, 2))
