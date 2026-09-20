#!/usr/bin/env python3
# 混ぜ合わせ生成(attract_to_latent_prior)時の孤立率・3本以上合流・交差を、過去の実データ・
# 生成結果の測定値と比較できる方法で数値化する診断ツール。VAEの学習・推論(生成)には
# 一切組み込まれない、独立した事後診断用のスクリプト。あわせて、重複スロットが3本以上合流の
# カウントを狂わせていないか、およびポインタの構造的な破綻(自己ループ・幽霊参照)の頻度も確認する
import numpy as np
import torch

from extract_stroke_features_v2 import CONNECTION_THRESHOLD
from vae_checkpoint_v2 import load_checkpoint
from vae_crossing_geometry_v2 import bezier_polyline_points, polyline_crossing_points, polyline_diagonal_involved_mask
from vae_data_v2 import prepare_datasets
from vae_eval_common import (
    AXIS_TOLERANCE_DEG,
    JUNCTION_CLUSTER_RADIUS,
    SEGMENTS_PER_CURVE,
    existence_mask_from_logits,
)
from vae_eval_common_v2 import (
    attract_to_latent_prior,
    duplicate_slot_pairs,
    encode_batch,
    load_batch,
    reconstructed_strokes_real,
    to_real_scale,
    true_strokes_real,
)
from vae_losses import AngleGMM, angle_log_density, build_angle_gmm
from vae_losses_v2 import MIN_DIRECTION_NORM
from vae_model_v2 import select_device
from vae_synthetic_losses import masked_mean_std

SAMPLE_COUNT = 2000  # 過去の実データ・生成結果の測定と同じ値(歴史的な比較のため)
SEED = 0


def _isolated_stroke_rate(start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray) -> float:
    # 端点間の実スケール距離のみで孤立を判定する(曲線同士の交差は考慮しない)。
    # 過去の実データ・生成結果の測定値と比較できるよう、この定義(距離ベース、交差非考慮)を維持する
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


def _triple_junction_count(
    crossing: np.ndarray, crossing_point: np.ndarray, start_points: np.ndarray, end_points: np.ndarray, mask: np.ndarray
) -> int:
    # 3本以上合流の検出。交差点(crossing/crossing_pointとしてベクトル化済み)と端点同士の近接
    # (ここではPythonのまま、ストローク数に対してO(n^2)だが定数が小さく軽い)を同じ「交点」として
    # 扱い、近接する交点同士をUnion-Findで1つのクラスタにまとめた上で、そのクラスタに関与する
    # ストローク数を数える(1つのクラスタに3本以上のストロークが絡んでいれば3本以上合流とみなす)
    active = np.where(mask)[0]
    interaction_points = []
    interaction_strokes = []
    for ii in range(len(active)):
        a = active[ii]
        for jj in range(ii + 1, len(active)):
            b = active[jj]
            if crossing[a, b]:
                interaction_points.append(crossing_point[a, b])
                interaction_strokes.append(frozenset((a, b)))
            for pa in (start_points[a], end_points[a]):
                for pb in (start_points[b], end_points[b]):
                    if np.linalg.norm(pa - pb) < CONNECTION_THRESHOLD:
                        interaction_points.append((pa + pb) / 2)
                        interaction_strokes.append(frozenset((a, b)))

    n_points = len(interaction_points)
    if n_points == 0:
        return 0
    parent = list(range(n_points))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in range(n_points):
        for q in range(p + 1, n_points):
            if np.linalg.norm(interaction_points[p] - interaction_points[q]) < JUNCTION_CLUSTER_RADIUS:
                union(p, q)

    clusters: dict[int, set[int]] = {}
    for idx in range(n_points):
        clusters.setdefault(find(idx), set()).update(interaction_strokes[idx])
    return sum(1 for strokes in clusters.values() if len(strokes) >= 3)


def _crossings_and_triple_junctions(
    start_points: np.ndarray, end_points: np.ndarray, offsets: np.ndarray, existence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 線分交差探索(最も重い部分、ストロークペア×12線分×12線分の判定)だけベクトル化して求め、
    # 3本以上合流のUnion-Findクラスタリング(元々軽い)はそのままPythonループで行う。この
    # ベクトル化版は、実データ全件において素朴なループ実装と完全に同じ結果を返すことを検証済み
    start = torch.from_numpy(start_points).float()
    end = torch.from_numpy(end_points).float()
    offset = torch.from_numpy(offsets).float()
    existence_t = torch.from_numpy(existence).float()

    control = (start + end) / 2 + offset
    points = bezier_polyline_points(start, end, control, SEGMENTS_PER_CURVE)
    crossing, crossing_point = polyline_crossing_points(points)

    delta = end - start
    angle = torch.atan2(delta[..., 1], delta[..., 0])
    diagonal = polyline_diagonal_involved_mask(crossing, angle, AXIS_TOLERANCE_DEG)

    stroke_count = existence.shape[1]
    upper_triangle = torch.triu(torch.ones(stroke_count, stroke_count, dtype=torch.bool), diagonal=1)
    pair_exists = (existence_t.unsqueeze(2) * existence_t.unsqueeze(1)).bool()
    mask = pair_exists & upper_triangle.unsqueeze(0)

    total = (crossing & mask).sum(dim=(1, 2)).numpy().astype(float)
    diag = (diagonal & mask).sum(dim=(1, 2)).numpy().astype(float)

    crossing_np = crossing.numpy()
    crossing_point_np = crossing_point.numpy()
    existence_np = existence.astype(bool)
    n = len(start_points)
    triple = np.zeros(n)
    for i in range(n):
        triple[i] = _triple_junction_count(
            crossing_np[i], crossing_point_np[i], start_points[i], end_points[i], existence_np[i]
        )
    return total, diag, triple


def _angle_naturalness_log_density(
    start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray, angle_gmm: AngleGMM
) -> float:
    # 正解データが無い生成経路でも比較できるよう、正解との相対評価ではなく、GMMの対数密度をそのまま
    # 使う絶対評価にする(値が大きいほど自然)。退化した(始点・終点がほぼ同じ)ストロークは、
    # 方向ベクトルのノルムがMIN_DIRECTION_NORM未満のものとして除外する
    direction = end_points - start_points
    well_defined = np.linalg.norm(direction, axis=-1) >= MIN_DIRECTION_NORM
    mask = existence.astype(bool) & well_defined
    angle = np.arctan2(direction[..., 1], direction[..., 0])
    angle_tensor = torch.from_numpy(angle[mask]).float().to(angle_gmm.means.device)
    log_density = angle_log_density(angle_tensor, angle_gmm)
    return log_density.mean().item()


def _offset_std(offsets: np.ndarray, existence: np.ndarray) -> float:
    # ストロークoffset(曲がり具合)の標準偏差。生成側でこの分散が実データ比で大きく潰れやすいため、
    # 都度スクリプトで個別に測るのではなく3点比較の正式な指標にする
    return offsets[existence.astype(bool)].std()


def _stroke_length(start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray) -> float:
    # ストローク長(実スケールでの始点・終点間距離)の平均。生成側でこれが実データ比で大きく縮みやすいため、
    # 都度スクリプトで個別に測るのではなく3点比較の正式な指標にする。ストローク数1以下の字は平均が
    # 単一ストロークの値に支配されてしまうため、count<2の字を除外した上で字ごとの平均を平均する
    # (全ストロークをフラットにプールした平均ではない)
    length = np.linalg.norm(end_points - start_points, axis=-1)
    mean, _, count = masked_mean_std(
        torch.from_numpy(length).float(), torch.from_numpy(existence.astype("float32"))
    )
    valid = count >= 2
    return mean[valid].mean().item()


def _report(
    label: str,
    start_points: np.ndarray,
    end_points: np.ndarray,
    offsets: np.ndarray,
    existence: np.ndarray,
    angle_gmm: AngleGMM,
) -> None:
    crossings, diag, triple = _crossings_and_triple_junctions(start_points, end_points, offsets, existence)
    print(f"--- {label} (n={len(start_points)}) ---")
    print(f"isolated_stroke_rate = {_isolated_stroke_rate(start_points, end_points, existence):.2f}%")
    print(f"crossings mean = {crossings.mean():.3f} (diagonal-involved = {diag.mean():.3f})")
    print(f"triple_junctions mean = {triple.mean():.3f}")
    print(f"angle_naturalness (log density, higher = more natural) = "
          f"{_angle_naturalness_log_density(start_points, end_points, existence, angle_gmm):.3f}")
    print(f"offset_std = {_offset_std(offsets, existence):.4f}")
    print(f"stroke_length mean = {_stroke_length(start_points, end_points, existence):.3f}")
    print()


def _pointer_targets(logits: torch.Tensor) -> np.ndarray:
    return logits.argmax(dim=-1).cpu().numpy()


def _self_loop_rate(start_index: np.ndarray, end_index: np.ndarray, existence: np.ndarray) -> float:
    # 始点・終点の頂点ポインタを独立に離散選択する機構固有の失敗モード。正解データでは
    # 始点・終点は必ず異なる頂点を指すため、両者が同じ頂点を指すことは長さ0の自己参照ストロークを意味する
    self_loop = (start_index == end_index) & existence
    return 100 * self_loop.sum() / existence.sum() if existence.sum() else float("nan")


def _phantom_reference_rate(
    pointer_index: np.ndarray, vertex_existence_mask: np.ndarray, existence: np.ndarray
) -> float:
    # ストロークのポインタが指す先の頂点が、その頂点自身のexistenceでは「存在しない」と
    # 判定されているケース(幽霊参照)の頻度
    target_exists = np.take_along_axis(vertex_existence_mask, pointer_index, axis=1)
    phantom = ~target_exists & existence
    return 100 * phantom.sum() / existence.sum() if existence.sum() else float("nan")


def _report_pointer_integrity(
    label: str,
    start_index: np.ndarray,
    end_index: np.ndarray,
    vertex_existence_mask: np.ndarray,
    existence: np.ndarray,
) -> None:
    print(f"--- {label}: pointer integrity ---")
    print(f"self_loop_rate = {_self_loop_rate(start_index, end_index, existence):.2f}%")
    print(
        f"phantom_reference_rate (start/end) = "
        f"{_phantom_reference_rate(start_index, vertex_existence_mask, existence):.2f}% / "
        f"{_phantom_reference_rate(end_index, vertex_existence_mask, existence):.2f}%"
    )
    print()


def _duplicate_pair_real_distances(
    vertex_features: torch.Tensor, existence_mask: torch.Tensor, vertex_mean: torch.Tensor, vertex_std: torch.Tensor
) -> np.ndarray:
    # 重複スロットと判定されたペアに絞って、実スケールでの距離を集める。重複判定自体は標準化後の
    # 座標空間の閾値で行われているため、実スケールのクラスタリング半径(JUNCTION_CLUSTER_RADIUS)と
    # 直接比較するには、ここで距離を実スケールに変換し直す必要がある
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
    datasets = prepare_datasets()
    angle_gmm = build_angle_gmm(datasets.angle_gmm_params, device)
    train_batch = load_batch(datasets, "train", device)

    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, train_batch)

    true_strokes = true_strokes_real(checkpoint, train_batch)
    true_existence = train_batch.stroke_existence.cpu().numpy().astype(bool)
    true_vertex_existence_mask = train_batch.vertex_existence.cpu().numpy().astype(bool)
    true_start_index = train_batch.stroke_vertex_indices[..., 0].cpu().numpy()
    true_end_index = train_batch.stroke_vertex_indices[..., 1].cpu().numpy()
    _report("real data", true_strokes.start, true_strokes.end, true_strokes.offsets, true_existence, angle_gmm)
    _report_pointer_integrity(
        "real data", true_start_index, true_end_index, true_vertex_existence_mask, true_existence
    )

    with torch.no_grad():
        recon_output = checkpoint.model.decode(mu_real)
    recon_strokes = reconstructed_strokes_real(checkpoint, recon_output)
    recon_existence = existence_mask_from_logits(recon_output.stroke_existence_logits)
    _report(
        "reconstruction (encode -> decode(mu))",
        recon_strokes.start, recon_strokes.end, recon_strokes.offsets, recon_existence, angle_gmm,
    )
    _report_pointer_integrity(
        "reconstruction (encode -> decode(mu))",
        _pointer_targets(recon_output.start_pointer_logits),
        _pointer_targets(recon_output.end_pointer_logits),
        existence_mask_from_logits(recon_output.vertex_existence_logits),
        recon_existence,
    )

    torch.manual_seed(SEED)
    z_raw = torch.randn(SAMPLE_COUNT, checkpoint.latent_dim, device=device)
    with torch.no_grad():
        z = attract_to_latent_prior(z_raw, mu_real)
        decoder_output = checkpoint.model.decode(z)

    generated_strokes = reconstructed_strokes_real(checkpoint, decoder_output)
    generated_existence = existence_mask_from_logits(decoder_output.stroke_existence_logits)
    _report(
        "generated (current production checkpoint)",
        generated_strokes.start, generated_strokes.end, generated_strokes.offsets, generated_existence, angle_gmm,
    )
    _report_pointer_integrity(
        "generated (current production checkpoint)",
        _pointer_targets(decoder_output.start_pointer_logits),
        _pointer_targets(decoder_output.end_pointer_logits),
        existence_mask_from_logits(decoder_output.vertex_existence_logits),
        generated_existence,
    )

    _print_duplicate_slot_check(
        decoder_output.vertex_features, decoder_output.vertex_existence_logits,
        checkpoint.vertex_mean, checkpoint.vertex_std,
    )


if __name__ == "__main__":
    main()
