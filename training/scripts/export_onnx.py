#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from torch.export import Dim

from vae_checkpoint import load_checkpoint
from vae_model import VAE, ModelShape, unflatten_output

# webがfetchして読み込む配置場所(training/dataではなくweb/publicに置く)
ONNX_PATH = Path(__file__).resolve().parent.parent.parent / "web" / "public" / "vae_phase1.onnx"

OPSET_VERSION = 18  # 使用する演算(Linear, ReLU, Sigmoidなど)はいずれも古くから存在し、特定opsetを要求する要素はないため、比較的新しく安定している値を選んだ
VERIFICATION_BATCH_SIZE = 4  # エクスポート時のダミー入力(バッチサイズ1)とは異なるサイズで、可変バッチが実際に機能するか確認する


class _GenerationModel(nn.Module):
    # decode + 標準化の逆変換 + existenceのSigmoidまで含めることで、
    # web側はモデル固有の後処理(mean/std, sigmoid)を再実装せず、生スケールのストローク特徴量と
    # 存在確率をそのまま受け取れるようにする
    def __init__(self, model: VAE, shape: ModelShape, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.shape = shape
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon = self.model.decode(z)
        strokes, existence_logits = unflatten_output(recon, self.shape)
        strokes = strokes * self.std + self.mean
        return strokes, torch.sigmoid(existence_logits)


def _verify_export(generation_model: _GenerationModel, z: torch.Tensor) -> None:
    # トレースベースのエクスポートが実際に同じ計算を再現できているか、PyTorch側の出力と突き合わせて確認する
    with torch.no_grad():
        expected_strokes, expected_existence_prob = generation_model(z)

    session = onnxruntime.InferenceSession(str(ONNX_PATH))
    actual_strokes, actual_existence_prob = session.run(None, {"z": z.numpy()})

    strokes_diff = np.abs(actual_strokes - expected_strokes.numpy()).max()
    existence_diff = np.abs(actual_existence_prob - expected_existence_prob.numpy()).max()
    print(f"Max abs diff: strokes={strokes_diff:.2e} existence={existence_diff:.2e}")


def main() -> None:
    # エクスポートはCPU上で行う(推論性能は問題にならず、デバイス依存の挙動差を避けるため)
    device = torch.device("cpu")
    checkpoint = load_checkpoint(device)
    generation_model = _GenerationModel(
        checkpoint.model, checkpoint.shape, checkpoint.mean, checkpoint.std
    )
    generation_model.eval()

    # トレースはグラフの形状・構造を記録するだけで値自体は結果に影響しないため、値・バッチサイズは何でもよい
    dummy_z = torch.zeros(1, checkpoint.latent_dim)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    # バッチ軸を固定サイズにせず、推論時に任意のバッチサイズを受け付けられるようにする
    batch = Dim("batch")
    torch.onnx.export(
        generation_model,
        (dummy_z,),
        str(ONNX_PATH),
        input_names=["z"],
        output_names=["strokes", "existence_prob"],
        dynamic_shapes=({0: batch},),
        opset_version=OPSET_VERSION,
        # モデルが小さく(数百KB)、外部データファイルに分ける利点がないため単一ファイルにまとめる
        external_data=False,
    )
    print(f"Saved ONNX model to {ONNX_PATH}")

    verification_z = torch.randn(VERIFICATION_BATCH_SIZE, checkpoint.latent_dim)
    _verify_export(generation_model, verification_z)


if __name__ == "__main__":
    main()
