#!/usr/bin/env python3
# 生成結果の各軸(ストローク数・長さ・曲がり具合・面積・軸方向率・孤立率・交差・3本以上合流・
# 同方向ストロークの束)を、実データと揃えた方法で数値化する診断ツール。VAEの学習・推論(生成)には
# 一切組み込まれない、独立した事後診断用のスクリプト
import numpy as np
import torch

from vae_checkpoint import load_checkpoint
from vae_eval_common import (
    SEGMENTS_PER_CURVE,
    attract_to_latent_prior,
    bezier_polyline,
    crossings_and_triple_junctions,
    encode,
    existence_mask_from_logits,
    find_crossing_pairs,
    is_axis_aligned,
    load_train_data,
    stroke_endpoints_array,
    strokes_to_curves,
)
from vae_model import select_device, unflatten_output
from vae_synthetic_losses import masked_mean_std

SAMPLE_COUNT = 2000
CONNECTION_THRESHOLD = 4.0  # extract_stroke_features.CONNECTION_THRESHOLDと同じ
BUNDLED_ANGLE_THRESHOLD_DEG = 15.0  # vae_eval_common.AXIS_TOLERANCE_DEGと同じ考え方(同方向とみなす角度差)


def _bbox_area(strokes: np.ndarray, existence: np.ndarray) -> np.ndarray:
    midpoints = stroke_endpoints_array(strokes).mean(axis=-2)
    n = strokes.shape[0]
    areas = np.zeros(n)
    for i in range(n):
        active = np.where(existence[i])[0]
        if len(active) < 2:
            continue
        points = midpoints[i, active]
        areas[i] = (points[:, 0].max() - points[:, 0].min()) * (points[:, 1].max() - points[:, 1].min())
    return areas


def _isolated_stroke_rate(strokes: np.ndarray, existence: np.ndarray) -> float:
    endpoints = stroke_endpoints_array(strokes)
    total_isolated = total_strokes = 0
    for i in range(len(strokes)):
        active = np.where(existence[i])[0]
        if len(active) == 0:
            continue
        points = endpoints[i, active].reshape(len(active) * 2, 2)
        dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        for k in range(len(active)):
            dist[2 * k, 2 * k + 1] = np.inf
            dist[2 * k + 1, 2 * k] = np.inf
        np.fill_diagonal(dist, np.inf)
        has_conn = (dist.reshape(len(active), 2, len(active) * 2) < CONNECTION_THRESHOLD).any(axis=(1, 2))
        total_isolated += int((~has_conn).sum())
        total_strokes += len(active)
    return 100 * total_isolated / total_strokes if total_strokes else float("nan")


def _circular_angle_diff_deg(a: float, b: float) -> float:
    # 向きの違いだけを見る(180度反対向きは同じ直線とみなす)ため、角度をmod 180で比較する
    diff = np.degrees(a - b) % 180
    return np.minimum(diff, 180 - diff)


def _is_bundled_pair(polyline_a: np.ndarray, polyline_b: np.ndarray, angle_a: float, angle_b: float) -> bool:
    # 端点同士が近い(=接続)ペアは対象外。それ以外で、同方向(角度差15度未満)かつ線分同士が
    # CONNECTION_THRESHOLD未満まで近接していれば「束」とみなす。交差しているかどうかの判定は
    # 呼び出し側(ペアごとに1回で済み、ここで毎回計算する必要がないため)で行う
    endpoints_a = np.array([polyline_a[0], polyline_a[-1]])
    endpoints_b = np.array([polyline_b[0], polyline_b[-1]])
    endpoint_dist = np.linalg.norm(endpoints_a[:, None, :] - endpoints_b[None, :, :], axis=-1).min()
    if endpoint_dist < CONNECTION_THRESHOLD:
        return False
    if _circular_angle_diff_deg(angle_a, angle_b) >= BUNDLED_ANGLE_THRESHOLD_DEG:
        return False
    diff = polyline_a[:, None, :] - polyline_b[None, :, :]
    min_dist = np.sqrt((diff**2).sum(axis=-1)).min()
    return min_dist < CONNECTION_THRESHOLD


def _bundled_stroke_rate(strokes: np.ndarray, existence: np.ndarray) -> float:
    # 交差も端点接続もしていないのに同方向・近接なペア(_is_bundled_pair参照)を「束」とみなし、
    # 字あたりの平均個数を返す。生成結果で同方向のストロークが近接して並走する現象を数値化する
    total_bundled = 0
    n = strokes.shape[0]
    for i in range(n):
        mask = existence[i]
        active = np.where(mask)[0]
        if len(active) < 2:
            continue
        curves = strokes_to_curves(strokes[i], mask)
        crossing_pairs = set(find_crossing_pairs(curves))
        polylines = [bezier_polyline(s, c, e, SEGMENTS_PER_CURVE) for s, c, e in curves]
        angles = strokes[i, active, 2]
        m = len(active)
        for a in range(m):
            for b in range(a + 1, m):
                if (a, b) in crossing_pairs or (b, a) in crossing_pairs:
                    continue
                if _is_bundled_pair(polylines[a], polylines[b], angles[a], angles[b]):
                    total_bundled += 1
    return total_bundled / n


def _report(label: str, strokes: np.ndarray, existence: np.ndarray) -> None:
    existence_bool = existence.astype(bool)
    stroke_count = existence_bool.sum(axis=1)
    length_mean, length_std, length_count = masked_mean_std(
        torch.from_numpy(strokes[..., 3]), torch.from_numpy(existence_bool.astype(np.float32))
    )
    curviness = np.linalg.norm(strokes[..., 4:6], axis=-1)
    curviness_mean, curviness_std, _ = masked_mean_std(
        torch.from_numpy(curviness), torch.from_numpy(existence_bool.astype(np.float32))
    )
    valid_char = length_count >= 2  # ストローク数1以下の字は標準偏差が定義できない(常に0になる)ため除外する
    crossings, diag, triple = crossings_and_triple_junctions(strokes, existence_bool)

    print(f"--- {label} (n={len(strokes)}) ---")
    print(f"stroke_count mean(std) = {stroke_count.mean():.2f} ({stroke_count.std():.2f})")
    print(f"stroke_length mean(std) = {length_mean[valid_char].mean():.2f} ({length_std[valid_char].mean():.2f})")
    print(f"curviness mean(std) = {curviness_mean[valid_char].mean():.2f} ({curviness_std[valid_char].mean():.2f})")
    print(f"bbox_area mean = {_bbox_area(strokes, existence_bool).mean():.1f}")
    print(f"axis_aligned_rate = {100 * np.mean([is_axis_aligned(a) for a in strokes[..., 2][existence_bool]]):.1f}%")
    print(f"isolated_stroke_rate = {_isolated_stroke_rate(strokes, existence_bool):.1f}%")
    print(f"crossings mean = {crossings.mean():.3f} (diagonal-involved = {diag.mean():.3f})")
    print(f"triple_junctions mean = {triple.mean():.3f}")
    print(f"bundled_stroke_pairs mean = {_bundled_stroke_rate(strokes, existence_bool):.4f}")


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    train_data = load_train_data(checkpoint, device)
    mu_real, _ = encode(checkpoint, train_data.strokes_standardized, train_data.existence_tensor)

    _report("real data", train_data.strokes, train_data.existence)

    torch.manual_seed(0)
    z_raw = torch.randn(SAMPLE_COUNT, checkpoint.latent_dim, device=device)
    with torch.no_grad():
        z = attract_to_latent_prior(z_raw, mu_real)
        recon = checkpoint.model.decode(z)
        strokes_recon, existence_logits = unflatten_output(recon, checkpoint.shape)
        strokes_recon_real = (strokes_recon * checkpoint.std + checkpoint.mean).cpu().numpy()
        existence_recon = existence_mask_from_logits(existence_logits)

    _report("generated (current production checkpoint)", strokes_recon_real, existence_recon)


if __name__ == "__main__":
    main()
