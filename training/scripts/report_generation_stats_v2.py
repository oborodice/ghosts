#!/usr/bin/env python3
# 混ぜ合わせ生成(attract_to_latent_prior)時の孤立率・3本以上合流・交差を、
# report_generation_stats.pyと揃えた方法で数値化する診断ツール。VAEの学習・推論(生成)には
# 一切組み込まれない、独立した事後診断用のスクリプト。あわせて、重複スロットが3本以上合流の
# カウントを狂わせていないかも確認する
import numpy as np
import torch

from vae_checkpoint_v2 import load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common import (
    JUNCTION_CLUSTER_RADIUS,
    attract_to_latent_prior,
    classify_crossings,
    count_triple_junctions,
    existence_mask_from_logits,
)
from vae_eval_common_v2 import (
    duplicate_slot_pairs,
    encode_batch,
    load_batch,
    reconstructed_strokes_real,
    stroke_curves,
    to_real_scale,
    true_strokes_real,
)
from vae_model_v2 import select_device

SAMPLE_COUNT = 2000  # report_generation_stats.pyと同じ値(歴史的な比較のため)
SEED = 0
CONNECTION_THRESHOLD = 4.0  # extract_stroke_features_v2.CONNECTION_THRESHOLDと同じ


def _isolated_stroke_rate(start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray) -> float:
    # report_generation_stats.pyの_isolated_stroke_rateと同じ定義(端点間距離のみを見て、
    # 曲線の交差は考慮しない)を、始点・終点配列に対して適用する
    endpoints = np.stack([start_points, end_points], axis=2)  # (N, stroke_count, 2[始点/終点], 2[x, y])
    total_isolated = total_strokes = 0
    for i in range(len(start_points)):
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


def _active_angles(start_points: np.ndarray, end_points: np.ndarray, existence_mask: np.ndarray) -> np.ndarray:
    # classify_crossingsに渡す角度は、stroke_curvesと同じ順序(存在するストロークのみ、元の並び順)で揃える必要がある
    delta = end_points[existence_mask] - start_points[existence_mask]
    return np.arctan2(delta[:, 1], delta[:, 0])


def _crossings_and_triple_junctions(
    start_points: np.ndarray, end_points: np.ndarray, offsets: np.ndarray, existence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # classify_crossings/count_triple_junctionsはいずれも曲線タプルのみを受け取り、頂点+辺構造か
    # 独立スロット表現かに依存しないため、vae_eval_common.pyからそのまま流用する。2つの指標を
    # 1つの関数にまとめているのは、サンプルごとのcurves構築(交差判定を含み重い)を1回で
    # 両方に使い回すため(呼び出し元は_reportのみで、ファイル間の重複排除が目的ではない)
    n = len(start_points)
    total = np.zeros(n)
    diag = np.zeros(n)
    triple = np.zeros(n)
    for i in range(n):
        mask = existence[i].astype(bool)
        curves = stroke_curves(start_points[i], end_points[i], offsets[i], mask)
        angles = _active_angles(start_points[i], end_points[i], mask)
        cls = classify_crossings(curves, angles)
        total[i] = cls["total"]
        diag[i] = cls["diagonal_involved"]
        triple[i] = count_triple_junctions(curves)
    return total, diag, triple


def _report(
    label: str, start_points: np.ndarray, end_points: np.ndarray, offsets: np.ndarray, existence: np.ndarray
) -> None:
    crossings, diag, triple = _crossings_and_triple_junctions(start_points, end_points, offsets, existence)
    print(f"--- {label} (n={len(start_points)}) ---")
    print(f"isolated_stroke_rate = {_isolated_stroke_rate(start_points, end_points, existence):.2f}%")
    print(f"crossings mean = {crossings.mean():.3f} (diagonal-involved = {diag.mean():.3f})")
    print(f"triple_junctions mean = {triple.mean():.3f}")
    print()


def _duplicate_pair_real_distances(
    vertex_features: torch.Tensor, existence_mask: torch.Tensor, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> np.ndarray:
    # 重複スロットと判定されたペアに絞って、実スケールでの距離を集める
    # (標準化後の判定閾値DUPLICATE_POSITION_THRESHOLDでは、3本以上合流のクラスタリング半径
    # JUNCTION_CLUSTER_RADIUSと直接比較できないため)
    distances = []
    for sample_idx in range(vertex_features.shape[0]):
        active_indices = existence_mask[sample_idx].nonzero(as_tuple=True)[0]
        if len(active_indices) < 2:
            continue
        positions = vertex_features[sample_idx, active_indices]
        pair_rows, pair_cols = duplicate_slot_pairs(positions)
        if len(pair_rows) == 0:
            continue
        real_positions = to_real_scale(positions, vertex_mean, vertex_std)
        rows, cols = pair_rows.cpu().numpy(), pair_cols.cpu().numpy()
        real_distance = np.linalg.norm(real_positions[rows] - real_positions[cols], axis=-1)
        distances.extend(real_distance.tolist())
    return np.array(distances)


def _print_duplicate_slot_check(
    vertex_features: torch.Tensor,
    vertex_existence_logits: torch.Tensor,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
) -> None:
    # 重複スロット(異なる頂点スロットがほぼ同じ座標を指す現象)が、3本以上合流の空間クラスタリングに
    # 見逃されず飲み込まれているか(=カウントを狂わせていないか)を確認する
    print("--- duplicate slot pairs vs. triple-junction clustering radius ---")
    existence_mask = torch.from_numpy(existence_mask_from_logits(vertex_existence_logits))
    distances = _duplicate_pair_real_distances(vertex_features, existence_mask, vertex_mean, vertex_std)
    if len(distances) == 0:
        print("No duplicate slot pairs found in this sample.")
        return
    within_radius = (distances < JUNCTION_CLUSTER_RADIUS).mean() * 100
    print(
        f"real-scale distance: mean={distances.mean():.3f} max={distances.max():.3f} "
        f"(n={len(distances)}, junction clustering radius={JUNCTION_CLUSTER_RADIUS})"
    )
    print(f"fraction within clustering radius (= not missed by triple-junction counting): {within_radius:.1f}%")


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    train_batch = load_batch(prepare_datasets(), "train", device)

    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, train_batch)

    true_strokes = true_strokes_real(checkpoint, train_batch)
    true_existence = train_batch.stroke_existence.cpu().numpy().astype(bool)
    _report("real data", true_strokes.start, true_strokes.end, true_strokes.offsets, true_existence)

    torch.manual_seed(SEED)
    z_raw = torch.randn(SAMPLE_COUNT, checkpoint.latent_dim, device=device)
    with torch.no_grad():
        z = attract_to_latent_prior(z_raw, mu_real)
        decoder_output = checkpoint.model.decode(z)

    generated_strokes = reconstructed_strokes_real(checkpoint, decoder_output)
    generated_existence = existence_mask_from_logits(decoder_output.stroke_existence_logits)
    _report(
        "generated (current production checkpoint)",
        generated_strokes.start, generated_strokes.end, generated_strokes.offsets, generated_existence,
    )

    _print_duplicate_slot_check(
        decoder_output.vertex_features, decoder_output.vertex_existence_logits,
        checkpoint.vertex_mean, checkpoint.vertex_std,
    )


if __name__ == "__main__":
    main()
