#!/usr/bin/env python3
from typing import NamedTuple

import torch

from vae_model import CHECKPOINT_PATH, VAE, ModelShape, SlotAttentionConfig


class Checkpoint(NamedTuple):
    model: VAE
    shape: ModelShape
    hidden_dims: tuple[int, int]
    slot_attention_config: SlotAttentionConfig
    mean: torch.Tensor
    std: torch.Tensor
    latent_dim: int


def load_checkpoint(device: torch.device) -> Checkpoint:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    shape = ModelShape(checkpoint["slot_count"], checkpoint["feature_dim"])
    slot_attention_config = SlotAttentionConfig(*checkpoint["slot_attention_config"])
    model = VAE(shape, checkpoint["hidden_dims"], checkpoint["latent_dim"], slot_attention_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return Checkpoint(
        model,
        shape,
        checkpoint["hidden_dims"],
        slot_attention_config,
        checkpoint["mean"].to(device),
        checkpoint["std"].to(device),
        checkpoint["latent_dim"],
    )


def save_checkpoint(
    model: VAE,
    shape: ModelShape,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    # 生成(推論)に必要な情報のみを保存する(学習再開用のoptimizer状態・epoch数などは含まない)。
    # hidden_dims・latent_dimはvae_model側の現在のグローバル定数ではなく、呼び出し元(モデルの実際の
    # 構築元)から明示的に受け取る。読み込んだチェックポイントをそのまま再保存するケース
    # (finetune_stroke_count.py)で、モデル構築時と異なるハイパーパラメータへ変わらないようにするため
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            # torch.loadのweights_only=True(デフォルト)はNamedTupleサブクラスを許可しないため、
            # プレーンなtupleとして保存する
            "slot_attention_config": tuple(slot_attention_config),
            "slot_count": shape.slot_count,
            "feature_dim": shape.feature_dim,
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
        },
        CHECKPOINT_PATH,
    )
