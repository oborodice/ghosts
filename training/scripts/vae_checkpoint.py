#!/usr/bin/env python3
from typing import NamedTuple

import torch

from vae_model import CHECKPOINT_PATH, VAE, ModelShape, SlotAttentionConfig


class Checkpoint(NamedTuple):
    model: VAE
    shape: ModelShape
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
        checkpoint["mean"].to(device),
        checkpoint["std"].to(device),
        checkpoint["latent_dim"],
    )
