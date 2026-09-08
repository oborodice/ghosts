#!/usr/bin/env python3
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

HIDDEN_DIMS: tuple[int, int] = (768, 384)
LATENT_DIM = 32

# 学習時の保存先であると同時に、将来の推論/生成スクリプトの読み込み先でもある
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "vae.pt"


class ModelShape(NamedTuple):
    slot_count: int
    feature_dim: int

    @property
    def input_dim(self) -> int:
        # ストローク特徴量(slot_count × feature_dim)に、スロットごとのexistenceフラグ分を加える
        return self.slot_count * self.feature_dim + self.slot_count


class VAE(nn.Module):
    def __init__(self, shape: ModelShape, hidden_dims: tuple[int, int], latent_dim: int) -> None:
        super().__init__()
        hidden1, hidden2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(shape.input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, shape.input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + std * epsという決定的な式にすることで、サンプリングを微分可能にする(再パラメータ化トリック)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def flatten_input(strokes: torch.Tensor, existence: torch.Tensor) -> torch.Tensor:
    # strokesを平坦化してからexistenceを連結する。この並び順はunflatten_outputと対応させる必要がある
    return torch.cat([strokes.flatten(1), existence], dim=1)


def unflatten_output(flat: torch.Tensor, shape: ModelShape) -> tuple[torch.Tensor, torch.Tensor]:
    # flatten_inputと同じ並び順(strokes→existence)を前提に分割する
    strokes_dim = shape.slot_count * shape.feature_dim
    strokes_flat, existence_logits = flat[:, :strokes_dim], flat[:, strokes_dim:]
    return strokes_flat.view(-1, shape.slot_count, shape.feature_dim), existence_logits


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
