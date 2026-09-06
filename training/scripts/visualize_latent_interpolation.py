#!/usr/bin/env python3
import matplotlib.pyplot as plt
import torch

from vae_checkpoint import load_checkpoint
from vae_eval_common import decode_to_segments, draw_segments, encode, load_validation_data
from vae_model import select_device

# 補間の両端に使うvalidationサンプルのインデックス
SAMPLE_INDEX_A = 0
SAMPLE_INDEX_B = 1
STEP_COUNT = 5


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    data = load_validation_data(checkpoint, device)

    indices = [SAMPLE_INDEX_A, SAMPLE_INDEX_B]
    strokes = data.strokes_standardized[indices]
    existence = data.existence_tensor[indices]
    mu, _ = encode(checkpoint, strokes, existence)
    mu_a, mu_b = mu[0], mu[1]

    # tを0〜1に等間隔分割し、両端(t=0, t=1)がそれぞれmu_a, mu_bと一致する直線補間を行う
    t = torch.linspace(0, 1, STEP_COUNT, device=device).unsqueeze(1)
    z = mu_a * (1 - t) + mu_b * t
    segments = decode_to_segments(checkpoint, z)

    _, axes = plt.subplots(nrows=1, ncols=STEP_COUNT)
    for ax, sample_segments in zip(axes, segments):
        draw_segments(ax, sample_segments)
    axes[0].set_title(f"idx {SAMPLE_INDEX_A}")
    axes[-1].set_title(f"idx {SAMPLE_INDEX_B}")
    plt.show()


if __name__ == "__main__":
    main()
