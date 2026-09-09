#!/usr/bin/env python3
import matplotlib.pyplot as plt
import torch

from vae_checkpoint import load_checkpoint
from vae_eval_common import attract_to_latent_prior, decode_to_curves, draw_curves, encode, load_train_data
from vae_model import select_device

SAMPLE_COUNT = 6


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    train_data = load_train_data(checkpoint, device)
    mu_real, _ = encode(checkpoint, train_data.strokes_standardized, train_data.existence_tensor)

    # 生成時に実際に使う経路(事前分布からのサンプリング+実データへのカーネル重み付け)そのものを確認する
    # (export_onnx.pyがエクスポートするグラフと同じ処理。カーネル重み付けを行わないと、
    # 生成時に実際には出現しない座標を見ることになる)
    z_raw = torch.randn(SAMPLE_COUNT, checkpoint.latent_dim, device=device)
    z = attract_to_latent_prior(z_raw, mu_real)
    curves = decode_to_curves(checkpoint, z)

    _, axes = plt.subplots(nrows=1, ncols=SAMPLE_COUNT)
    for ax, sample_curves in zip(axes, curves):
        draw_curves(ax, sample_curves)
    plt.show()


if __name__ == "__main__":
    main()
