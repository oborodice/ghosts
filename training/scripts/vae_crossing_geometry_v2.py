#!/usr/bin/env python3
# ストロークを線分として扱う交差判定の幾何計算。再構成側(vae_losses_v2.py、折れ線近似)と
# 合成z側(vae_synthetic_losses_v2.py、弦のみ)の両方から使うため、共有モジュールに切り出している
import torch

CROSSING_GATE_LOW = 0.08  # 交点パラメータt, uがこの範囲内ならストローク内部での交差とみなす(下限)。端点付近の接続点を除外する
CROSSING_GATE_HIGH = 0.92  # 同上(上限)
CROSSING_GATE_SHARPNESS = 40.0  # interior_gateのsigmoidの急峻さ(大きいほど矩形窓に近づく)
CROSSING_DENOM_EPSILON = 1e-6  # 平行な線分同士でのゼロ除算回避


def pairwise_segment_intersection_params(
    start1: torch.Tensor, end1: torch.Tensor, start2: torch.Tensor, end2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # 線分1(start1->end1、ストロークiの区間)と線分2(start2->end2、ストロークjの区間)の
    # 全ペア(i, j)について、交点パラメータt(線分1上の位置)・u(線分2上の位置)をクラメルの公式
    # (2元1次方程式)で解く。t, uはアフィン変換で不変(2次元クロス積の分子・分母に現れるdetが
    # 約分される)ため、start1/end1とstart2/end2が同じアフィン変換後の空間(標準化頂点座標空間)に
    # あれば、標準化スケールの座標をそのまま使ってよい
    d1 = (end1 - start1).unsqueeze(2)  # (B, stroke_count, 1, 2) -- iの方向、jへブロードキャスト
    d2 = (end2 - start2).unsqueeze(1)  # (B, 1, stroke_count, 2) -- jの方向、iへブロードキャスト
    denom = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    # 平行な線分(denom≈0)はt, uが発散するが、interior_gateで範囲外に押し出されるため実害はない
    denom = torch.where(denom.abs() < CROSSING_DENOM_EPSILON, torch.full_like(denom, CROSSING_DENOM_EPSILON), denom)

    diff = start2.unsqueeze(1) - start1.unsqueeze(2)  # p3 - p1, (B, stroke_count(i), stroke_count(j), 2)
    t = (diff[..., 0] * d2[..., 1] - diff[..., 1] * d2[..., 0]) / denom
    u = (diff[..., 0] * d1[..., 1] - diff[..., 1] * d1[..., 0]) / denom
    return t, u


def interior_gate(t: torch.Tensor) -> torch.Tensor:
    # t=CROSSING_GATE_LOW〜HIGHの範囲(ストローク内部)を、2つのsigmoidの積による
    # なめらかな矩形窓で近似する(線分交差判定は本質的に微分不可能なため)
    low = torch.sigmoid(CROSSING_GATE_SHARPNESS * (t - CROSSING_GATE_LOW))
    high = torch.sigmoid(CROSSING_GATE_SHARPNESS * (CROSSING_GATE_HIGH - t))
    return low * high


def chord_crossing_per_sample_total(
    start: torch.Tensor, end: torch.Tensor, existence: torch.Tensor, well_defined: torch.Tensor
) -> torch.Tensor:
    # ストロークを始点・終点を結ぶ弦とみなした交差数を、サンプルごとに合計する。上三角(i<j)のみを
    # 対象にして各ペアを1回だけ数え、両ストロークが存在し(existence)かつ両方が退化していない
    # (well_defined)ペアに限定する。crossingの頻度を集計統計として扱う損失・目標値の計算
    # (vae_data_v2._compute_crossing_targets、vae_synthetic_losses_v2._compute_synthetic_crossing_loss)
    # で共通して使う
    t, u = pairwise_segment_intersection_params(start, end, start, end)
    strength = interior_gate(t) * interior_gate(u)

    stroke_count = strength.shape[1]
    upper_triangle = torch.triu(
        torch.ones(stroke_count, stroke_count, dtype=torch.bool, device=strength.device), diagonal=1
    ).float()
    pair_exists = existence.unsqueeze(2) * existence.unsqueeze(1)
    pair_well_defined = well_defined.unsqueeze(2) * well_defined.unsqueeze(1)
    mask = pair_exists * pair_well_defined * upper_triangle.unsqueeze(0)
    return (strength * mask).sum(dim=(1, 2))
