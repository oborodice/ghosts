#!/usr/bin/env python3
import matplotlib.pyplot as plt
import torch

from vae_checkpoint import load_checkpoint
from vae_eval_common import decode_to_segments, draw_segments
from vae_model import select_device

SAMPLE_COUNT = 6


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)

    # 生成時に実際に使う経路(事前分布からのサンプリング)そのものを確認するため、実在データはエンコードしない
    z = torch.randn(SAMPLE_COUNT, checkpoint.latent_dim, device=device)
    segments = decode_to_segments(checkpoint, z)

    _, axes = plt.subplots(nrows=1, ncols=SAMPLE_COUNT)
    for ax, sample_segments in zip(axes, segments):
        draw_segments(ax, sample_segments)
    plt.show()


if __name__ == "__main__":
    main()
