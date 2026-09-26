#!/usr/bin/env python3
# 潜在変数をsimplex noiseで動かしたときの、生成結果のなめらかさ(モーフィング中に、字が瞬間的に
# 飛んだり、ストロークがパッと現れたりしないか)を数値化する診断ツール。VAEの学習・推論(生成)には
# 一切組み込まれない、独立した事後診断用のスクリプト。
#
# 軌跡は、フロントエンドと同じ作り方(次元ごとに独立したノイズ、時刻 x 速さを入力、倍率を掛ける)で
# 生成する。ノイズの実装はフロントエンドのライブラリとは別(opensimplex)のため、同じ軌跡にはならないが、
# 統計的な性質(値の範囲・変化の速さ)は合わせてある
import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from opensimplex import OpenSimplex

from vae_checkpoint_v2 import Checkpoint, load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common import existence_mask_from_logits
from vae_eval_common_v2 import (
    GENERATION_SOFT_TEMPERATURE,
    attract_to_latent_prior,
    decode_in_chunks,
    encode_batch,
    load_batch,
    reconstructed_strokes_real,
)
from vae_model_v2 import CHECKPOINT_PATH

FPS = 30.0
Z_SCALE = 2.0  # フロントエンドがsimplex noiseの出力に掛ける倍率
FRAMES_PER_TRAJECTORY = 300
TRAJECTORY_COUNT = 64
PERIODS_SECONDS = (30.0, 60.0)  # フロントエンドのSPEED(=1/周期)に対応する周期の秒数。大きいほどゆっくり動く
SEED = 0
JUMP_THRESHOLD = 10.0  # 連続する2フレームの間で、端点がこれ以上(キャンバスは109単位)動いたらジャンプとみなす
ATTRACT_CHUNK_SIZE = 3000  # 全フレームを1回でattractすると、実在字との距離行列が大きくなりすぎるため分割する
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 0  # 信頼区間を毎回同じ値にするため固定する
CONFIDENCE_PERCENTILES = (2.5, 97.5)  # 95%信頼区間

# opensimplexの出力は、フロントエンドが使うsimplex-noiseより、値の振れ幅が小さく、変化がゆっくりしている。
# 値の標準偏差(0.439)と、連続するフレーム間の移動量(0.0221)が、フロントエンドの出力と一致するように、
# 実測で合わせた係数
NOISE_AMPLITUDE = 1.2
NOISE_TIME_SCALE = 1.9
DIMENSION_SPACING = 10.0  # 次元ごとに、ノイズ場のy方向にこれだけ離した行を使い、次元どうしを無関係にする
NOISE_START_RANGE = 256.0  # 軌跡ごとの開始位置を選ぶ範囲


class DecodedFrames(NamedTuple):
    # 先頭の軸は、全軌跡・全フレームを連結した、フレーム数の軸
    start: np.ndarray  # (frames, stroke_count, 2)
    end: np.ndarray  # (frames, stroke_count, 2)
    offsets: np.ndarray  # (frames, stroke_count, 2)
    existence: np.ndarray  # (frames, stroke_count) 存在すると判定されたスロット


def generate_walks(
    trajectory_count: int, frame_count: int, latent_dim: int, period_seconds: float, seed: int
) -> np.ndarray:
    # 返り値は(trajectory_count, frame_count, latent_dim)。軌跡ごとに、ノイズ場のseedと開始位置を変える
    rng = np.random.default_rng(seed)
    noise_times = NOISE_TIME_SCALE * np.arange(frame_count) / FPS / period_seconds
    rows = np.arange(latent_dim) * DIMENSION_SPACING
    walks = np.empty((trajectory_count, frame_count, latent_dim), dtype=np.float32)
    for trajectory in range(trajectory_count):
        noise = OpenSimplex(int(rng.integers(0, 2**31)))
        start_time = rng.uniform(0, NOISE_START_RANGE)
        walks[trajectory] = noise.noise2array(start_time + noise_times, rows).T * NOISE_AMPLITUDE * Z_SCALE
    return walks


def prepare_latents(walks: np.ndarray, mu_real: torch.Tensor) -> torch.Tensor:
    # walks: (trajectory_count, frame_count, latent_dim)を、attract_to_latent_priorに通す。
    # 返り値は、全軌跡・全フレームを連結した(frames, latent_dim)
    z = torch.from_numpy(walks.reshape(-1, walks.shape[-1]))
    with torch.no_grad():
        return torch.cat(
            [
                attract_to_latent_prior(z[i : i + ATTRACT_CHUNK_SIZE], mu_real)[0]
                for i in range(0, len(z), ATTRACT_CHUNK_SIZE)
            ]
        )


def decode_frames(checkpoint: Checkpoint, z: torch.Tensor, soft_temperature: float) -> DecodedFrames:
    with torch.no_grad():
        output = decode_in_chunks(checkpoint.model, z, soft_temperature=soft_temperature)
    strokes = reconstructed_strokes_real(checkpoint, output)
    return DecodedFrames(
        strokes.start,
        strokes.end,
        strokes.offsets,
        existence_mask_from_logits(output.stroke_existence_logits),
    )


def _by_trajectory(values: np.ndarray, trajectory_count: int) -> np.ndarray:
    return values.reshape(trajectory_count, -1, *values.shape[1:])


def _slot_displacement(frames: DecodedFrames, trajectory_count: int) -> np.ndarray:
    # 連続する2フレームの間の、同じスロットの端点の移動量(始点・終点の大きいほう)の、スロット間の最大値。
    # 両方のフレームで存在するスロットだけを見る。返り値は(trajectory_count, 軌跡あたりのフレーム数 - 1)
    start, end, existence = (_by_trajectory(a, trajectory_count) for a in (frames.start, frames.end, frames.existence))
    exists_in_both = existence[:, 1:] & existence[:, :-1]
    moved = np.maximum(
        np.linalg.norm(start[:, 1:] - start[:, :-1], axis=-1), np.linalg.norm(end[:, 1:] - end[:, :-1], axis=-1)
    )
    return np.where(exists_in_both, moved, 0).max(axis=-1)


def _bootstrap_confidence_interval(per_trajectory_values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(per_trajectory_values)
    means = [per_trajectory_values[rng.integers(0, count, count)].mean() for _ in range(BOOTSTRAP_ROUNDS)]
    low, high = np.percentile(means, CONFIDENCE_PERCENTILES)
    return float(low), float(high)


def print_smoothness_metrics(frames: DecodedFrames, z: torch.Tensor, trajectory_count: int) -> None:
    # frames・zは、trajectory_count本の同じ長さの軌跡を、フレームの軸で連結したもの
    frames_per_trajectory = len(z) // trajectory_count
    displacement = _slot_displacement(frames, trajectory_count)
    z_path = z.reshape(trajectory_count, frames_per_trajectory, -1)
    z_step_lengths = (z_path[:, 1:] - z_path[:, :-1]).norm(dim=2).numpy()
    is_jump = displacement > JUMP_THRESHOLD
    jump_rate_per_trajectory = is_jump.mean(axis=1)
    low, high = _bootstrap_confidence_interval(jump_rate_per_trajectory)
    existence = _by_trajectory(frames.existence, trajectory_count)
    flip_count = (existence[:, 1:] != existence[:, :-1]).sum()
    duration = (frames_per_trajectory - 1) / FPS

    print(
        f"  frame transitions with a slot jump (> {JUMP_THRESHOLD:g} units): {100 * jump_rate_per_trajectory.mean():.2f}%"
        f" (95% CI {100 * low:.2f}-{100 * high:.2f})"
    )
    print(f"  jumps per unit of z path length (speed-independent): {is_jump.sum() / z_step_lengths.sum():.2f}")
    print(f"  stroke pop-in/out: {flip_count / trajectory_count / duration:.2f} flips per second per trajectory")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantify how smoothly generated characters change while z follows simplex noise.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to the VAE checkpoint to evaluate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")  # モデルが小さく、CPUで十分な速さのため
    checkpoint = load_checkpoint(device, checkpoint_path=args.checkpoint)
    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, load_batch(prepare_datasets(), "train", device))

    for period in PERIODS_SECONDS:
        walks = generate_walks(TRAJECTORY_COUNT, FRAMES_PER_TRAJECTORY, checkpoint.latent_dim, period, SEED)
        z = prepare_latents(walks, mu_real)
        print(f"=== period {period:g}s ({TRAJECTORY_COUNT} trajectories x {FRAMES_PER_TRAJECTORY} frames, {FPS:g} fps)")
        print_smoothness_metrics(decode_frames(checkpoint, z, GENERATION_SOFT_TEMPERATURE), z, TRAJECTORY_COUNT)


if __name__ == "__main__":
    main()
