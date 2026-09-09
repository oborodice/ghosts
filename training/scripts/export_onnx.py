#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn as nn

from vae_checkpoint import load_checkpoint
from vae_eval_common import attract_to_latent_prior, encode, load_train_data
from vae_model import VAE, ModelShape, unflatten_output

# webがfetchして読み込む配置場所(training/dataではなくweb/publicに置く)
ONNX_PATH = Path(__file__).resolve().parent.parent.parent / "web" / "public" / "vae.onnx"

OPSET_VERSION = 18  # 使用する演算(Linear, ReLU, Sigmoidなど)はいずれも古くから存在し、特定opsetを要求する要素はないため、比較的新しく安定している値を選んだ
# web側は1フレームにつき1文字しか生成しないため、バッチサイズは常に1で固定する。decoderに
# nn.TransformerEncoderを導入した際、torch.onnx.exportの可変バッチ(dynamic_shapes)が
# 効かなくなることを確認したため、元々不要だった可変バッチ対応自体を廃止した
BATCH_SIZE = 1


class _GenerationModel(nn.Module):
    # 生成用のz_rawを実データ(mu_real)へ引き寄せるカーネル重み付け(attract_to_latent_prior)+
    # decode + 標準化の逆変換 + existenceのSigmoidまで含めることで、
    # web側は学習データの統計的な性質を一切知らず、生成用のzをそのまま渡すだけでよくなる
    def __init__(
        self, model: VAE, shape: ModelShape, mean: torch.Tensor, std: torch.Tensor, mu_real: torch.Tensor
    ) -> None:
        super().__init__()
        self.model = model
        self.shape = shape
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mu_real", mu_real)

    def forward(self, z_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = attract_to_latent_prior(z_raw, self.mu_real)
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
    train_data = load_train_data(checkpoint, device)
    mu_real, _ = encode(checkpoint, train_data.strokes_standardized, train_data.existence_tensor)

    generation_model = _GenerationModel(
        checkpoint.model, checkpoint.shape, checkpoint.mean, checkpoint.std, mu_real
    )
    generation_model.eval()

    # トレースはグラフの形状・構造を記録するだけで値自体は結果に影響しないため、値は何でもよい
    dummy_z = torch.zeros(BATCH_SIZE, checkpoint.latent_dim)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        generation_model,
        (dummy_z,),
        str(ONNX_PATH),
        input_names=["z"],
        output_names=["strokes", "existence_prob"],
        opset_version=OPSET_VERSION,
        # mu_real(学習データ全件のencode結果)を含めても数MB程度で、外部データファイルに
        # 分ける利点がないため単一ファイルにまとめる
        external_data=False,
    )
    print(f"Saved ONNX model to {ONNX_PATH}")

    verification_z = torch.randn(BATCH_SIZE, checkpoint.latent_dim)
    _verify_export(generation_model, verification_z)


if __name__ == "__main__":
    main()
