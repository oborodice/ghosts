#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch

from vae_checkpoint_v2 import Checkpoint, load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common import draw_curves
from vae_eval_common_v2 import (
    Batch,
    encode_batch,
    load_batch,
    reconstructed_strokes_real,
    stroke_curves,
    to_real_scale,
    true_strokes_real,
)
from vae_model_v2 import DecoderOutput, select_device

TYPICAL_INDICES = [0, 1, 2, 3]  # 典型的なサンプルとして固定で表示する検証データの先頭4件


def _decode(checkpoint: Checkpoint, batch: Batch) -> DecoderOutput:
    with torch.no_grad():
        # 再パラメータ化のサンプリングεによるランダム性を排除するため、muをそのままdecodeする
        mu, _ = encode_batch(checkpoint, batch)
        return checkpoint.model.decode(mu)


def _sample_mean_vertex_distance(true_real: np.ndarray, existence: np.ndarray, recon_real: np.ndarray) -> np.ndarray:
    # サンプルごとの頂点の平均距離(存在する頂点のみ)。ワースト字を選ぶために使う
    distance = np.linalg.norm(recon_real - true_real, axis=-1)
    return (distance * existence).sum(axis=1) / existence.sum(axis=1)


def _worst_vertex_index(checkpoint: Checkpoint, batch: Batch, decoder_output: DecoderOutput) -> int:
    true_real = to_real_scale(batch.vertices, checkpoint.vertex_mean, checkpoint.vertex_std)
    recon_real = to_real_scale(decoder_output.vertex_features, checkpoint.vertex_mean, checkpoint.vertex_std)
    existence = batch.vertex_existence.cpu().numpy().astype(bool)
    return int(np.argmax(_sample_mean_vertex_distance(true_real, existence, recon_real)))


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    batch = load_batch(prepare_datasets(), "val", device)

    decoder_output = _decode(checkpoint, batch)
    true_strokes = true_strokes_real(checkpoint, batch)
    reconstructed_strokes = reconstructed_strokes_real(checkpoint, decoder_output)
    stroke_existence_np = batch.stroke_existence.cpu().numpy().astype(bool)

    worst_index = _worst_vertex_index(checkpoint, batch, decoder_output)
    indices = [*TYPICAL_INDICES, worst_index]

    _, axes = plt.subplots(nrows=2, ncols=len(indices))
    for col, index in enumerate(indices):
        original_curves = stroke_curves(
            true_strokes.start[index], true_strokes.end[index], true_strokes.offsets[index],
            stroke_existence_np[index],
        )
        reconstructed_curves = stroke_curves(
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
