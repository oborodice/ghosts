#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch

from vae_checkpoint import load_checkpoint
from vae_eval_common import load_train_data
from vae_model import flatten_input, select_device

# webがfetchして読み込む配置場所(export_onnx.pyのvae_phase1.onnxと同じ配置方針)
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "web" / "public" / "latent_prior.bin"


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    train_data = load_train_data(checkpoint, device)

    with torch.no_grad():
        mu, _ = checkpoint.model.encode(
            flatten_input(train_data.strokes_standardized, train_data.existence_tensor)
        )

    # web側はfloat32のバイナリをそのままFloat32Arrayとして読み込むため、JSONではなく生バイナリで保存する
    # (点数×潜在次元数を1次元に平坦化した並び。1点分がlatent_dim個ずつ連続する)
    mu_flat = mu.cpu().numpy().astype(np.float32).reshape(-1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mu_flat.tofile(OUTPUT_PATH)
    print(f"Saved {mu.shape[0]} points x {mu.shape[1]} dims to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
