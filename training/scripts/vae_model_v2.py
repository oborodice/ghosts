#!/usr/bin/env python3
# 頂点+ストロークのデコーダは、今後(頂点のみ→ソフトポインタ→ハードポインタの順で)クエリ数や
# 出力ヘッドの構成が段階的に増えていく前提のため、既存の学習パイプラインとは独立した専用モジュールとして
# 定義する。現時点(頂点のみ)の内容自体は、スロット数・特徴量次元に依存しない既存のSelf-Attention
# ベースのデコーダ設計をそのまま踏襲している
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn

# 頂点のみのこの段階では未チューニングで据え置いている値。再構成精度に問題があれば見直す
HIDDEN_DIMS: tuple[int, int] = (1024, 512)
LATENT_DIM = 48

SLOT_DIM = 256
SLOT_ATTENTION_HEADS = 4
SLOT_ATTENTION_LAYERS = 2
SLOT_ATTENTION_FFN_DIM = 512

# 学習時の保存先であると同時に、将来の推論/生成スクリプトの読み込み先でもある
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "vae_v2.pt"


class ModelShape(NamedTuple):
    slot_count: int
    feature_dim: int

    @property
    def input_dim(self) -> int:
        # 特徴量(slot_count × feature_dim)に、スロットごとのexistenceフラグ分を加える
        return self.slot_count * self.feature_dim + self.slot_count


class SlotAttentionConfig(NamedTuple):
    slot_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int


class SlotAttentionDecoder(nn.Module):
    # 各スロットが他のスロットの出力を参照しながら特徴量を決められるよう、DETRの学習可能なobject queryに
    # 近い発想で、スロットごとの埋め込み+zの文脈をSelf-Attentionで相互参照させてから、スロットごとに
    # 読み出す。出力はunflatten_outputとの互換のため従来通りのflat(features→existenceの順)な
    # ベクトルにして返す
    def __init__(self, shape: ModelShape, latent_dim: int, config: SlotAttentionConfig) -> None:
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(shape.slot_count, config.slot_dim))
        self.z_to_context = nn.Linear(latent_dim, config.slot_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.slot_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.feature_head = nn.Linear(config.slot_dim, shape.feature_dim)
        self.existence_head = nn.Linear(config.slot_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        context = self.z_to_context(z).unsqueeze(1)  # (B, 1, SLOT_DIM)
        # slot_queries: (1, slot_count, SLOT_DIM) + context: (B, 1, SLOT_DIM) はbroadcastで
        # (B, slot_count, SLOT_DIM)になる(バッチ方向の明示的なexpandは不要)
        slots = self.slot_queries.unsqueeze(0) + context
        slots = self.transformer(slots)
        features_flat = self.feature_head(slots).flatten(1)
        existence_logits = self.existence_head(slots).squeeze(-1)
        return torch.cat([features_flat, existence_logits], dim=1)


class VAE(nn.Module):
    def __init__(
        self,
        shape: ModelShape,
        hidden_dims: tuple[int, int],
        latent_dim: int,
        slot_attention_config: SlotAttentionConfig,
    ) -> None:
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
        self.decoder = SlotAttentionDecoder(shape, latent_dim, slot_attention_config)

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


def flatten_input(features: torch.Tensor, existence: torch.Tensor) -> torch.Tensor:
    # この並び順はunflatten_outputと対応させる必要がある
    return torch.cat([features.flatten(1), existence], dim=1)


def unflatten_output(flat: torch.Tensor, shape: ModelShape) -> tuple[torch.Tensor, torch.Tensor]:
    # flatten_inputと同じ並び順(features→existence)を前提に分割する
    features_dim = shape.slot_count * shape.feature_dim
    features_flat, existence_logits = flat[:, :features_dim], flat[:, features_dim:]
    return features_flat.view(-1, shape.slot_count, shape.feature_dim), existence_logits


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
