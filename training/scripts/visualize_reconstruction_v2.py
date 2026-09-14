#!/usr/bin/env python3
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from vae_checkpoint_v2 import Checkpoint, load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common import draw_curves
from vae_eval_common_v2 import Batch, encode_batch, load_batch, to_real_scale
from vae_model_v2 import DecoderOutput, select_device

TYPICAL_INDICES = [0, 1, 2, 3]  # 典型的なサンプルとして固定で表示する検証データの先頭4件


class RealScaleStrokes(NamedTuple):
    start: np.ndarray  # (N, stroke_count, 2)
    end: np.ndarray  # (N, stroke_count, 2)
    offsets: np.ndarray  # (N, stroke_count, 2)


def _decode(checkpoint: Checkpoint, batch: Batch) -> DecoderOutput:
    with torch.no_grad():
        # 再パラメータ化のサンプリングεによるランダム性を排除するため、muをそのままdecodeする
        mu, _ = encode_batch(checkpoint, batch)
        return checkpoint.model.decode(mu)


def _true_strokes_real(checkpoint: Checkpoint, batch: Batch) -> RealScaleStrokes:
    vertices_real = to_real_scale(batch.vertices, checkpoint.vertex_mean, checkpoint.vertex_std)
    start_index = batch.stroke_vertex_indices[..., 0:1].cpu().numpy()
    end_index = batch.stroke_vertex_indices[..., 1:2].cpu().numpy()
    offsets_real = to_real_scale(batch.stroke_offsets, checkpoint.stroke_offset_mean, checkpoint.stroke_offset_std)
    return RealScaleStrokes(
        np.take_along_axis(vertices_real, start_index, axis=1),
        np.take_along_axis(vertices_real, end_index, axis=1),
        offsets_real,
    )


def _reconstructed_strokes_real(checkpoint: Checkpoint, decoder_output: DecoderOutput) -> RealScaleStrokes:
    start_real = to_real_scale(decoder_output.start_points, checkpoint.vertex_mean, checkpoint.vertex_std)
    end_real = to_real_scale(decoder_output.end_points, checkpoint.vertex_mean, checkpoint.vertex_std)
    offsets_real = to_real_scale(
        decoder_output.stroke_offsets, checkpoint.stroke_offset_mean, checkpoint.stroke_offset_std
    )
    return RealScaleStrokes(start_real, end_real, offsets_real)


def _sample_mean_vertex_distance(true_real: np.ndarray, existence: np.ndarray, recon_real: np.ndarray) -> np.ndarray:
    # サンプルごとの頂点の平均距離(存在する頂点のみ)。ワースト字を選ぶために使う
    distance = np.linalg.norm(recon_real - true_real, axis=-1)
    return (distance * existence).sum(axis=1) / existence.sum(axis=1)


def _worst_vertex_index(checkpoint: Checkpoint, batch: Batch, decoder_output: DecoderOutput) -> int:
    true_real = to_real_scale(batch.vertices, checkpoint.vertex_mean, checkpoint.vertex_std)
    recon_real = to_real_scale(decoder_output.vertex_features, checkpoint.vertex_mean, checkpoint.vertex_std)
    existence = batch.vertex_existence.cpu().numpy().astype(bool)
    return int(np.argmax(_sample_mean_vertex_distance(true_real, existence, recon_real)))


def _stroke_curves(
    start_points: np.ndarray, end_points: np.ndarray, offsets: np.ndarray, existence_mask: np.ndarray
) -> list[tuple[complex, complex, complex]]:
    # (始点, 終点, オフセット) -> (始点, 制御点, 終点)の2次ベジェ。制御点は弦(始点-終点)の中点をoffsetだけずらした点
    curves = []
    for (start_x, start_y), (end_x, end_y), (offset_x, offset_y), exists in zip(
        start_points, end_points, offsets, existence_mask
    ):
        if not exists:
            continue
        start = complex(start_x, start_y)
        end = complex(end_x, end_y)
        control = (start + end) / 2 + complex(offset_x, offset_y)
        curves.append((start, control, end))
    return curves


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    batch = load_batch(prepare_datasets(), "val", device)

    decoder_output = _decode(checkpoint, batch)
    true_strokes = _true_strokes_real(checkpoint, batch)
    reconstructed_strokes = _reconstructed_strokes_real(checkpoint, decoder_output)
    stroke_existence_np = batch.stroke_existence.cpu().numpy().astype(bool)

    worst_index = _worst_vertex_index(checkpoint, batch, decoder_output)
    indices = [*TYPICAL_INDICES, worst_index]

    _, axes = plt.subplots(nrows=2, ncols=len(indices))
    for col, index in enumerate(indices):
        original_curves = _stroke_curves(
            true_strokes.start[index], true_strokes.end[index], true_strokes.offsets[index],
            stroke_existence_np[index],
        )
        reconstructed_curves = _stroke_curves(
            reconstructed_strokes.start[index], reconstructed_strokes.end[index], reconstructed_strokes.offsets[index],
            stroke_existence_np[index],
        )
        draw_curves(axes[0, col], original_curves)
        title = f"idx {index}" + (" (worst)" if index == worst_index else "")
        axes[0, col].set_title(title)
        draw_curves(axes[1, col], reconstructed_curves)

    plt.show()


if __name__ == "__main__":
    main()
