#!/usr/bin/env python3
# 生成した字の、ストローク数の分布・字の大きさ・実在字との近さ・構造の破綻・なめらかさを、実在字と比べて
# 数値化する診断ツール。生成は、本番と同じLatentSampler(引き寄せと、実在字の分布に合わせる補正)と
# ソフトデコードで行う。VAEの学習・推論(生成)には一切組み込まれない、独立した事後診断用のスクリプト
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

import report_morph_smoothness_v2 as morph
from report_generation_stats_v2 import (
    angle_naturalness_log_density_sums,
    crossings_and_triple_junctions,
    isolated_stroke_counts,
)
from vae_checkpoint_v2 import Checkpoint, load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common import existence_mask_from_logits
from vae_eval_common_v2 import (
    RealScaleStrokes,
    decode_in_chunks,
    encode_batch,
    load_batch,
    reconstructed_strokes_real,
    to_real_scale,
    true_strokes_real,
)
from vae_generation_v2 import GENERATION_SOFT_TEMPERATURE, LatentSampler
from vae_losses import AngleGMM, build_angle_gmm
from vae_model_v2 import CHECKPOINT_PATH, DecoderOutput

SAMPLE_COUNT = 2000
SEED = 0
TRAJECTORY_PERIOD_SECONDS = 30.0
NEAR_COPY_DISTANCES = (0.24, 0.5)  # 実在字どうしの最寄り距離の下位5%・25%。これを下回る生成字は、実在字のコピーに近い
PASS_PERCENTILES = (2, 98)  # 1字ごとの合格率の基準にする、実在字の分布の範囲
MANY_STROKES = 20  # この本数以上の字の割合を、多い側の多様性の目安として出す(実在字の上位16%程度)
REPORTED_COUNT_QUANTILES = (0.05, 0.5, 0.95)


def _character_features(
    start: np.ndarray, end: np.ndarray, existence: np.ndarray, vertex_positions: np.ndarray, vertex_existence: np.ndarray
) -> dict[str, np.ndarray]:
    # 字ごとの、ストローク数・ストロークの平均の長さ・外接の幅と高さ
    length = np.linalg.norm(end - start, axis=-1)
    count = existence.sum(axis=1).astype(float)
    mean_length = np.array([length[i][existence[i]].mean() if existence[i].any() else 0.0 for i in range(len(count))])
    extent = lambda axis: np.array(
        [np.ptp(vertex_positions[i][vertex_existence[i], axis]) if vertex_existence[i].any() else 0.0 for i in range(len(count))]
    )
    return {"count": count, "mean_length": mean_length, "width": extent(0), "height": extent(1)}


def _output_vectors(vertices: torch.Tensor, vertex_existence: torch.Tensor) -> torch.Tensor:
    # 出力空間での距離に使う特徴(標準化スケールの頂点座標、存在しない頂点は0)
    return (vertices * vertex_existence.unsqueeze(-1)).reshape(len(vertices), -1)


def _pass_rate_and_thresholds(
    strokes: RealScaleStrokes,
    existence: np.ndarray,
    angle_gmm: AngleGMM,
    thresholds: dict[str, np.ndarray] | None = None,
    has_self_loop: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # 1字ごとの合格率(参考値)。3本以上合流・交差数・孤立率・角度・ストローク数が、実在字の分布の範囲に収まり、
    # 自己ループ(has_self_loop)がない字の割合。thresholdsを渡さなければ、この字の集合(=実在字)から範囲を求める
    crossings, _, triple = crossings_and_triple_junctions(strokes.start, strokes.end, strokes.offsets, existence)
    isolated, total = isolated_stroke_counts(strokes.start, strokes.end, existence)
    angle_sum, angle_count = angle_naturalness_log_density_sums(strokes.start, strokes.end, existence, angle_gmm)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_sample = {
            "triple": triple,
            "crossings": crossings,
            "isolated": isolated / np.where(total > 0, total, np.nan),
            "angle": angle_sum / np.where(angle_count > 0, angle_count, np.nan),
            "count": existence.sum(axis=1).astype(float),
        }
    if thresholds is None:
        thresholds = {k: np.nanpercentile(v, PASS_PERCENTILES) for k, v in per_sample.items()}
    inside = np.ones(len(existence), dtype=bool) if has_self_loop is None else ~has_self_loop
    for key, (low, high) in thresholds.items():
        inside &= (per_sample[key] >= low) & (per_sample[key] <= high)
    return inside, thresholds


def _pointer_failures(output: DecoderOutput, existence: np.ndarray) -> tuple[float, float, np.ndarray]:
    # 自己ループ(始点・終点が同じ頂点を指す)と、存在しない頂点への参照(始点・終点のどちらか)の、
    # 存在するストロークに対する割合(%)。3つ目は、字ごとに、自己ループがあるか
    start_index = output.start_pointer_logits.argmax(-1).numpy()
    end_index = output.end_pointer_logits.argmax(-1).numpy()
    vertex_exists = existence_mask_from_logits(output.vertex_existence_logits)
    total = existence.sum()
    self_loop = (start_index == end_index) & existence
    refers_missing = ~np.take_along_axis(vertex_exists, start_index, axis=1) | ~np.take_along_axis(
        vertex_exists, end_index, axis=1
    )
    phantom = (existence & refers_missing).sum() / total
    return 100 * float(self_loop.sum() / total), 100 * float(phantom), self_loop.any(axis=1)


@torch.no_grad()
def _print_static_metrics(
    checkpoint: Checkpoint,
    z: torch.Tensor,
    real_features: dict[str, np.ndarray],
    real_vectors: torch.Tensor,
    real_thresholds: dict[str, np.ndarray],
    angle_gmm: AngleGMM,
) -> None:
    soft = decode_in_chunks(checkpoint.model, z, soft_temperature=GENERATION_SOFT_TEMPERATURE)
    soft_strokes = reconstructed_strokes_real(checkpoint, soft)
    soft_existence = existence_mask_from_logits(soft.stroke_existence_logits)
    vertex_exists = existence_mask_from_logits(soft.vertex_existence_logits)
    features = _character_features(
        soft_strokes.start, soft_strokes.end, soft_existence,
        to_real_scale(soft.vertex_features, checkpoint.vertex_mean, checkpoint.vertex_std), vertex_exists,
    )
    count = features["count"]
    quantile_labels = "/".join(f"{100 * q:g}" for q in REPORTED_COUNT_QUANTILES)
    quantile_values = [int(np.quantile(count, q)) for q in REPORTED_COUNT_QUANTILES]
    print(
        f"  stroke count: mean={count.mean():.2f} std={count.std():.2f} "
        f"quantiles({quantile_labels}%)={quantile_values} "
        f">={MANY_STROKES} strokes={100 * (count >= MANY_STROKES).mean():.1f}% empty={100 * (count == 0).mean():.2f}% "
        f"(real: mean={real_features['count'].mean():.2f} std={real_features['count'].std():.2f} "
        f">={MANY_STROKES} strokes={100 * (real_features['count'] >= MANY_STROKES).mean():.1f}%)"
    )
    distances = " ".join(f"{name}={wasserstein_distance(features[key], real_features[key]):.2f}" for name, key in (
        ("count", "count"), ("mean_length", "mean_length"), ("width", "width"), ("height", "height")))
    print(f"  Wasserstein distance to real (smaller is closer): {distances}")

    vectors = _output_vectors(soft.vertex_features, torch.from_numpy(vertex_exists).float())
    nearest = torch.cdist(vectors, real_vectors).min(1).values
    fractions = " ".join(f"<{d:g}: {100 * float((nearest < d).float().mean()):.1f}%" for d in NEAR_COPY_DISTANCES)
    print(f"  distance to the nearest real character (output space): median={float(nearest.median()):.2f} {fractions}")

    hard = decode_in_chunks(checkpoint.model, z)
    hard_strokes = reconstructed_strokes_real(checkpoint, hard)
    hard_existence = existence_mask_from_logits(hard.stroke_existence_logits)
    self_loop, phantom, has_self_loop = _pointer_failures(hard, hard_existence)
    inside, _ = _pass_rate_and_thresholds(hard_strokes, hard_existence, angle_gmm, real_thresholds, has_self_loop)
    print(
        f"  reference (not a gate; hard decode): self-loop={self_loop:.2f}% phantom reference={phantom:.2f}% "
        f"per-character pass rate={100 * inside.mean():.1f}%"
    )


def _print_trajectory_metrics(checkpoint: Checkpoint, sampler: LatentSampler) -> None:
    walks = morph.generate_walks(
        morph.TRAJECTORY_COUNT, morph.FRAMES_PER_TRAJECTORY, checkpoint.latent_dim, TRAJECTORY_PERIOD_SECONDS, morph.SEED
    )
    z = morph.prepare_latents(walks, sampler)
    frames = morph.decode_frames(checkpoint, z, GENERATION_SOFT_TEMPERATURE)
    print(f"  period {TRAJECTORY_PERIOD_SECONDS:g}s, {morph.TRAJECTORY_COUNT} trajectories x {morph.FRAMES_PER_TRAJECTORY} frames")
    morph.print_smoothness_metrics(frames, z, morph.TRAJECTORY_COUNT)
    counts = frames.existence.sum(axis=1).reshape(morph.TRAJECTORY_COUNT, morph.FRAMES_PER_TRAJECTORY)
    print(
        f"  stroke count over time: mean={counts.mean():.1f} std within a trajectory={counts.std(axis=1).mean():.2f} "
        f"std overall={counts.std():.2f} empty frames={100 * (counts == 0).mean():.2f}%"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure the diversity, structural failures and smoothness of generated characters against real ones.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to the VAE checkpoint to evaluate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")  # モデルが小さく、CPUで十分な速さのため
    checkpoint = load_checkpoint(device, checkpoint_path=args.checkpoint)
    datasets = prepare_datasets()
    train_batch = load_batch(datasets, "train", device)
    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, train_batch)

    print(f"checkpoint={args.checkpoint}")

    real_strokes = true_strokes_real(checkpoint, train_batch)
    real_existence = train_batch.stroke_existence.numpy().astype(bool)
    real_vertex_exists = train_batch.vertex_existence.numpy().astype(bool)
    real_features = _character_features(
        real_strokes.start, real_strokes.end, real_existence,
        to_real_scale(train_batch.vertices, checkpoint.vertex_mean, checkpoint.vertex_std), real_vertex_exists,
    )
    angle_gmm = build_angle_gmm(datasets.angle_gmm_params, device)
    _, real_thresholds = _pass_rate_and_thresholds(real_strokes, real_existence, angle_gmm)
    real_vectors = _output_vectors(train_batch.vertices, train_batch.vertex_existence)

    sampler = LatentSampler(mu_real)
    torch.manual_seed(SEED)
    z, _ = sampler.sample(torch.randn(SAMPLE_COUNT, checkpoint.latent_dim))

    print("=== static samples (LatentSampler, soft decode)")
    _print_static_metrics(checkpoint, z, real_features, real_vectors, real_thresholds, angle_gmm)
    print("=== trajectories")
    _print_trajectory_metrics(checkpoint, sampler)


if __name__ == "__main__":
    main()
