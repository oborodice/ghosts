#!/usr/bin/env python3
# 実データ・再構成(encode→decode(mu))・生成(LatentSampler、ソフトデコード)の3つを、
# 同じ方法で測って並べる診断ツール。孤立率・3本以上合流・交差・角度の自然さ・offsetのばらつき・
# ストローク長・ストローク数・キャンバス占有率と、ポインタの構造的な破綻(自己ループ・幽霊参照)を測る。
# あわせて、生成側の混ぜ合わせの診断(near_dup_rate・effective_k)と、重複スロットが3本以上合流の
# カウントを狂わせていないかを確認する。VAEの学習・推論(生成)には一切組み込まれない、独立した事後診断用のスクリプト
import argparse
from pathlib import Path

import numpy as np
import torch

from extract_stroke_features_v2 import CONNECTION_THRESHOLD
from vae_checkpoint_v2 import Checkpoint, load_checkpoint
from vae_crossing_geometry_v2 import bezier_polyline_points, polyline_crossing_points, polyline_diagonal_involved_mask
from vae_data_v2 import prepare_datasets
from vae_eval_common import (
    AXIS_TOLERANCE_DEG,
    JUNCTION_CLUSTER_RADIUS,
    SEGMENTS_PER_CURVE,
    existence_mask_from_logits,
)
from vae_classifier_dataset_v2 import NEAREST_REAL_FILTER_THRESHOLD, VIEWBOX_SIZE
from vae_eval_common_v2 import (
    decode_in_chunks,
    duplicate_slot_pairs,
    encode_batch,
    load_batch,
    reconstructed_strokes_real,
    to_real_scale,
    true_strokes_real,
    vertex_distance_real,
)
from vae_generation_v2 import GENERATION_SOFT_TEMPERATURE, LatentSampler
from vae_losses import AngleGMM, angle_log_density, build_angle_gmm
from vae_losses_v2 import MIN_DIRECTION_NORM
from vae_model_v2 import CHECKPOINT_PATH, DecoderOutput, select_device
from vae_synthetic_losses import masked_mean_std

SAMPLE_COUNT = 2000  # 他の生成の診断(多様性のレポートなど)と同じ点数にし、結果を直接比べられるようにする
SEED = 0


def isolated_stroke_counts(
    start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 端点間の実スケール距離のみで孤立を判定する(曲線同士の交差は考慮しない)。
    # 過去の実データ・生成結果の測定値と比較できるよう、この定義(距離ベース、交差非考慮)を維持する。
    # サンプル(1字)ごとの(孤立ストローク数, 総ストローク数)を返す。母集団全体の割合(_isolated_stroke_rate)・
    # サンプルごとの割合(analyze_classifier_scores_v2.pyの既知指標)の両方をこの2つの値から計算できる
    endpoints = np.stack([start_points, end_points], axis=2)  # (N, stroke_count, 2[始点/終点], 2[x, y])
    n = len(start_points)
    isolated_count = np.zeros(n)
    total_count = np.zeros(n)
    for i in range(n):
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
        isolated_count[i] = int((~has_conn).sum())
        total_count[i] = len(active)
    return isolated_count, total_count


def _isolated_stroke_rate(start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray) -> float:
    isolated_count, total_count = isolated_stroke_counts(start_points, end_points, existence)
    total = total_count.sum()
    return 100 * isolated_count.sum() / total if total else float("nan")


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


def crossings_and_triple_junctions(
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


def angle_naturalness_log_density_sums(
    start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray, angle_gmm: AngleGMM
) -> tuple[np.ndarray, np.ndarray]:
    # 正解データが無い生成経路でも比較できるよう、正解との相対評価ではなく、GMMの対数密度をそのまま
    # 使う絶対評価にする(値が大きいほど自然)。退化した(始点・終点がほぼ同じ)ストロークは、
    # 方向ベクトルのノルムがMIN_DIRECTION_NORM未満のものとして除外する。
    # サンプル(1字)ごとの(対数密度の合計, 有効ストローク数)を返す。マスクする前に(N, stroke_count)
    # 全体で対数密度を計算してから合計側でマスクするため、マスク後にまとめて平均する場合(母集団全体、
    # _angle_naturalness_log_density)と全く同じ値になる
    direction = end_points - start_points
    well_defined = np.linalg.norm(direction, axis=-1) >= MIN_DIRECTION_NORM
    mask = existence.astype(bool) & well_defined
    angle = np.arctan2(direction[..., 1], direction[..., 0])
    angle_tensor = torch.from_numpy(angle).float().to(angle_gmm.means.device)
    log_density = angle_log_density(angle_tensor, angle_gmm).cpu().numpy()
    mask_f = mask.astype(np.float64)
    return (log_density * mask_f).sum(axis=1), mask_f.sum(axis=1)


def _angle_naturalness_log_density(
    start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray, angle_gmm: AngleGMM
) -> float:
    sums, counts = angle_naturalness_log_density_sums(start_points, end_points, existence, angle_gmm)
    total_count = counts.sum()
    return sums.sum() / total_count if total_count else float("nan")


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


def _stroke_count(existence: np.ndarray) -> float:
    # 1字あたりの実在ストローク数の平均
    return existence.astype(bool).sum(axis=1).mean()


def _canvas_occupancy(vertex_positions: np.ndarray, vertex_existence: np.ndarray) -> float:
    # 実在頂点の外接矩形が、キャンバス全体(KanjiVGのviewBox、VIEWBOX_SIZE四方)に対してどの程度の
    # 面積を占めるか。頂点が1個以下の字は外接矩形の面積が0になり
    # 占有率の意味を持たないため除外する(_stroke_lengthのcount<2除外と同じ考え方)
    mask = vertex_existence.astype(bool)
    ratios = []
    for i in range(len(vertex_positions)):
        active = vertex_positions[i, mask[i]]
        if len(active) < 2:
            continue
        width = active[:, 0].max() - active[:, 0].min()
        height = active[:, 1].max() - active[:, 1].min()
        ratios.append((width * height) / (VIEWBOX_SIZE**2))
    return float(np.mean(ratios)) if ratios else float("nan")


def _report(
    label: str,
    start_points: np.ndarray,
    end_points: np.ndarray,
    offsets: np.ndarray,
    existence: np.ndarray,
    angle_gmm: AngleGMM,
    vertex_positions: np.ndarray,
    vertex_existence: np.ndarray,
) -> None:
    crossings, diag, triple = crossings_and_triple_junctions(start_points, end_points, offsets, existence)
    print(f"--- {label} (n={len(start_points)}) ---")
    print(f"isolated_stroke_rate = {_isolated_stroke_rate(start_points, end_points, existence):.2f}%")
    print(f"crossings mean = {crossings.mean():.3f} (diagonal-involved = {diag.mean():.3f})")
    print(f"triple_junctions mean = {triple.mean():.3f}")
    print(f"angle_naturalness (log density, higher = more natural) = "
          f"{_angle_naturalness_log_density(start_points, end_points, existence, angle_gmm):.3f}")
    print(f"offset_std = {_offset_std(offsets, existence):.4f}")
    print(f"stroke_length mean = {_stroke_length(start_points, end_points, existence):.3f}")
    print(f"stroke_count mean = {_stroke_count(existence):.3f}")
    print(f"canvas_occupancy mean = {_canvas_occupancy(vertex_positions, vertex_existence) * 100:.1f}%")
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


def _evaluate_and_report(
    label: str, checkpoint: Checkpoint, decoder_output: DecoderOutput, angle_gmm: AngleGMM
) -> None:
    # decode結果(reconstruction・generatedのどちらも同じDecoderOutput形状)から、_report・
    # _report_pointer_integrityに渡す実スケールのストローク・頂点データを組み立てる共通処理。
    # 「real data」はDecoderOutputではなくデータセット由来のテンソルを直接使う(ポインタも
    # argmaxではなく正解のstroke_vertex_indices)ため、ここには含めない
    strokes = reconstructed_strokes_real(checkpoint, decoder_output)
    existence = existence_mask_from_logits(decoder_output.stroke_existence_logits)
    vertex_existence_mask = existence_mask_from_logits(decoder_output.vertex_existence_logits)
    vertex_positions = to_real_scale(decoder_output.vertex_features, checkpoint.vertex_mean, checkpoint.vertex_std)
    _report(
        label, strokes.start, strokes.end, strokes.offsets, existence, angle_gmm,
        vertex_positions, vertex_existence_mask,
    )
    _report_pointer_integrity(
        label,
        _pointer_targets(decoder_output.start_pointer_logits),
        _pointer_targets(decoder_output.end_pointer_logits),
        vertex_existence_mask,
        existence,
    )


def _print_vertex_reconstruction_error(
    vertices: torch.Tensor,
    existence: torch.Tensor,
    vertices_recon: torch.Tensor,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
) -> None:
    # 実スケールでの頂点再構成誤差。正解が存在しない合成z側(generated)には適用できないため、
    # reconstruction専用の別枠にする
    distances = vertex_distance_real(vertices, existence, vertices_recon, vertex_mean, vertex_std)
    print("--- reconstruction (encode -> decode(mu)): vertex accuracy ---")
    print(f"vertex reconstruction error (real-scale) mean = {distances.mean():.4f}")
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


def _print_mixing_diagnostics(z: torch.Tensor, weights: torch.Tensor, mu_real: torch.Tensor) -> None:
    # 合成z領域に新しい損失を試す際、目的の指標(isolated_stroke_rate等)だけでなくこの2指標も
    # 必ず確認する。損失が設計上encoderに直接勾配を流さなくても、decoderの適応を介した間接的な
    # 結合学習でencoderの表現(実在字の潜在空間上の配置)自体が変わりうることが実測で確認されており、
    # 目的の指標の改善が「decoderが混ぜ合わせをうまく扱えるようになった」のではなく「encoderが
    # 実在字を詰め込んだことの副産物」である可能性を、この2指標で切り分ける。定義は
    # vae_classifier_dataset_v2.pyのnear_dup_rate判定と揃える
    distances = torch.cdist(z, mu_real)
    nearest = distances.min(dim=1).values
    near_dup_rate = (nearest < NEAREST_REAL_FILTER_THRESHOLD).float().mean().item() * 100
    effective_k = 1.0 / (weights**2).sum(dim=1)
    print("--- generated: mixing diagnostics (near_dup_rate / effective_k) ---")
    print(f"near_dup_rate = {near_dup_rate:.2f}%")
    print(f"effective_k mean = {effective_k.mean():.2f}, median = {effective_k.median():.2f}")
    print()


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to the VAE checkpoint to evaluate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = select_device()
    checkpoint = load_checkpoint(device, checkpoint_path=args.checkpoint)
    datasets = prepare_datasets()
    angle_gmm = build_angle_gmm(datasets.angle_gmm_params, device)
    train_batch = load_batch(datasets, "train", device)

    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, train_batch)

    true_strokes = true_strokes_real(checkpoint, train_batch)
    true_existence = train_batch.stroke_existence.cpu().numpy().astype(bool)
    true_vertex_existence_mask = train_batch.vertex_existence.cpu().numpy().astype(bool)
    true_vertex_positions = to_real_scale(train_batch.vertices, checkpoint.vertex_mean, checkpoint.vertex_std)
    true_start_index = train_batch.stroke_vertex_indices[..., 0].cpu().numpy()
    true_end_index = train_batch.stroke_vertex_indices[..., 1].cpu().numpy()
    _report(
        "real data", true_strokes.start, true_strokes.end, true_strokes.offsets, true_existence, angle_gmm,
        true_vertex_positions, true_vertex_existence_mask,
    )
    _report_pointer_integrity(
        "real data", true_start_index, true_end_index, true_vertex_existence_mask, true_existence
    )

    with torch.no_grad():
        recon_output = decode_in_chunks(checkpoint.model, mu_real)
    _evaluate_and_report("reconstruction (encode -> decode(mu))", checkpoint, recon_output, angle_gmm)
    _print_vertex_reconstruction_error(
        train_batch.vertices, train_batch.vertex_existence, recon_output.vertex_features,
        checkpoint.vertex_mean, checkpoint.vertex_std,
    )

    torch.manual_seed(SEED)
    z_raw = torch.randn(SAMPLE_COUNT, checkpoint.latent_dim, device=device)
    with torch.no_grad():
        z, weights = LatentSampler(mu_real).sample(z_raw)
        decoder_output = decode_in_chunks(checkpoint.model, z, soft_temperature=GENERATION_SOFT_TEMPERATURE)

    _evaluate_and_report("generated", checkpoint, decoder_output, angle_gmm)
    _print_mixing_diagnostics(z, weights, mu_real)

    _print_duplicate_slot_check(
        decoder_output.vertex_features, decoder_output.vertex_existence_logits,
        checkpoint.vertex_mean, checkpoint.vertex_std,
    )


if __name__ == "__main__":
    main()
