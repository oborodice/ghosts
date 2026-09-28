#!/usr/bin/env python3
# 学習したGAN(train_glyph_gan.py のチェックポイント)の生成器を、ONNXに書き出す。
# 入力はsimplex noiseの値(潜在の次元の数)で、正規分布への変換 → 写像ネットワーク → 生成器 → インクの画像(0=紙〜1=インク、1チャンネル)。
# - 推論には移動平均の版の重みを使う
# - ノイズの画像(ノイズの注入)は、フレームごとに変えると字がちらつくため、定数として埋め込む
# 書き出したあと、同じ入力でPyTorchとONNX Runtimeの出力を比べる
import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from glyph_inference import GlyphGenerator, load_glyph_generator
from glyph_walk import simplex_scattered

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# PyTorchの既定(20)より古い版にして、表示側のONNX Runtimeが古い版(1.16以降)でも読めるようにする
OPSET_VERSION = 18


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "onnx" / "glyph_generator.onnx")
    return parser.parse_args()


def _compare_with_pytorch(model: GlyphGenerator, path: Path, simplex_values: torch.Tensor) -> None:
    with torch.no_grad():
        torch_ink = model(simplex_values).numpy()
    onnx_ink = onnxruntime.InferenceSession(str(path)).run(None, {"simplex_values": simplex_values.numpy()})[0]
    print(f"max abs diff between PyTorch and ONNX Runtime {np.abs(onnx_ink - torch_ink).max():.2e}")


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = load_glyph_generator(args.checkpoint)
    simplex_values = torch.from_numpy(simplex_scattered(model.latent_dim, 1, seed=0))  # 書き出しに使う例の入力(1フレーム分)
    # 重みも1つのファイルにまとめる(別ファイルに分けると、表示側へ .onnx だけを持っていったときに重みがなくて動かない)
    torch.onnx.export(model, (simplex_values,), args.output, input_names=["simplex_values"], output_names=["ink"], opset_version=OPSET_VERSION,
                      external_data=False)
    _compare_with_pytorch(model, args.output, simplex_values)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
