#!/usr/bin/env python3
# 頂点+ストロークのデコーダは、今後(頂点のみ→ソフトポインタ→ハードポインタの順で)クエリ数や
# 出力ヘッドの構成が段階的に増えていく前提のため、既存の学習パイプラインとは独立した専用モジュールとして
# 定義する。現時点(ハードポインタ)は、頂点トークン(座標+existence)とストロークトークン
# (参照先頂点へのポインタ+オフセット+existence)の2種類を同じSelf-Attentionスタックに通す。
# ポインタは学習時のみStraight-Through Gumbel-Softmaxで離散化し、推論時(model.eval())はノイズ無しの
# argmaxにする(ONNX変換後のargmax+gatherと一致させるため)
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 未チューニングで据え置いている値。再構成精度に問題があれば見直す
HIDDEN_DIMS: tuple[int, int] = (1024, 512)
LATENT_DIM = 48

SLOT_DIM = 256
SLOT_ATTENTION_HEADS = 4
SLOT_ATTENTION_LAYERS = 2
SLOT_ATTENTION_FFN_DIM = 512

VERTEX_TOKEN_TYPE = 0
STROKE_TOKEN_TYPE = 1

# 学習時の保存先であると同時に、将来の推論/生成スクリプトの読み込み先でもある
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "vae_v2.pt"


class ModelShape(NamedTuple):
    vertex_count: int
    vertex_feature_dim: int  # 座標(x, y)の次元数
    stroke_count: int
    stroke_feature_dim: int  # オフセット(offset_x, offset_y)の次元数。参照先頂点はポインタが担うため含めない

    @property
    def token_count(self) -> int:
        return self.vertex_count + self.stroke_count

    @property
    def input_dim(self) -> int:
        # この内訳(頂点の座標+existence、ストロークのone-hot×2+offset+existence)はflatten_inputの
        # 連結順序と対応させる必要がある
        vertex_dim = self.vertex_count * self.vertex_feature_dim + self.vertex_count
        stroke_dim = self.stroke_count * (2 * self.vertex_count + self.stroke_feature_dim) + self.stroke_count
        return vertex_dim + stroke_dim


class SlotAttentionConfig(NamedTuple):
    slot_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int


class DecoderOutput(NamedTuple):
    vertex_features: torch.Tensor  # (B, vertex_count, vertex_feature_dim)
    vertex_existence_logits: torch.Tensor  # (B, vertex_count)
    stroke_offsets: torch.Tensor  # (B, stroke_count, stroke_feature_dim)
    stroke_existence_logits: torch.Tensor  # (B, stroke_count)
    start_pointer_logits: torch.Tensor  # (B, stroke_count, vertex_count)
    end_pointer_logits: torch.Tensor  # (B, stroke_count, vertex_count)
    start_points: torch.Tensor  # (B, stroke_count, vertex_feature_dim) -- ポインタが選んだ頂点の座標(one-hot選択)
    end_points: torch.Tensor  # (B, stroke_count, vertex_feature_dim) -- 同上、終点座標


class SlotAttentionDecoder(nn.Module):
    # 各スロットが他のスロットの出力を参照しながら特徴量を決められるよう、DETRの学習可能なobject queryに
    # 近い発想で、スロットごとの埋め込み+zの文脈をSelf-Attentionで相互参照させてから、スロットごとに
    # 読み出す。頂点・ストロークの2種類のトークンを区別できるよう、スロット埋め込みにタイプ埋め込みを足す
    def __init__(
        self, shape: ModelShape, latent_dim: int, config: SlotAttentionConfig, gumbel_temperature: float
    ) -> None:
        super().__init__()
        self.shape = shape
        self.gumbel_temperature = gumbel_temperature
        self.slot_queries = nn.Parameter(torch.randn(shape.token_count, config.slot_dim))
        self.type_embedding = nn.Embedding(2, config.slot_dim)  # 2種類 = 頂点(VERTEX_TOKEN_TYPE)・ストローク(STROKE_TOKEN_TYPE)
        token_types = [VERTEX_TOKEN_TYPE] * shape.vertex_count + [STROKE_TOKEN_TYPE] * shape.stroke_count
        self.register_buffer("token_types", torch.tensor(token_types))  # 学習対象ではないためbufferとして持つ
        self.z_to_context = nn.Linear(latent_dim, config.slot_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.slot_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)

        self.vertex_feature_head = nn.Linear(config.slot_dim, shape.vertex_feature_dim)
        self.vertex_existence_head = nn.Linear(config.slot_dim, 1)
        self.stroke_offset_head = nn.Linear(config.slot_dim, shape.stroke_feature_dim)
        self.stroke_existence_head = nn.Linear(config.slot_dim, 1)

        # ストロークの始点・終点それぞれについて、頂点トークンへの注意(スケール済み内積)で参照先を選ぶ
        self.pointer_key_head = nn.Linear(config.slot_dim, config.slot_dim)
        self.pointer_start_query_head = nn.Linear(config.slot_dim, config.slot_dim)
        self.pointer_end_query_head = nn.Linear(config.slot_dim, config.slot_dim)

    def _pointer(
        self, query_head: nn.Linear, stroke_slots: torch.Tensor, key: torch.Tensor, vertex_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # スケール済みドット積アテンション(Transformerの標準的な手法)で、ストロークトークンから
        # 頂点トークンへの注意スコアを求める。scaleが無いと次元数が大きいほど内積の分散が増え、
        # softmax後の分布が極端に鋭くなりすぎて勾配が消える
        query = query_head(stroke_slots)  # (B, stroke_count, SLOT_DIM)
        scale = self.pointer_key_head.out_features**0.5
        logits = torch.bmm(query, key.transpose(1, 2)) / scale

        if self.training:
            # Straight-Through Gumbel-Softmax: 前向き計算はone-hot(ハード)だが、逆伝播は
            # ハード化前のsoftmax分布を通じて勾配が流れる
            selection = F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=True, dim=-1)
        else:
            # 推論時はGumbelノイズを乗せず、決定的なargmaxで選ぶ(ONNX変換後のargmax+gatherと一致させるため)
            selection = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(logits.dtype)

        points = torch.bmm(selection, vertex_features)
        return logits, points

    def forward(self, z: torch.Tensor) -> DecoderOutput:
        context = self.z_to_context(z).unsqueeze(1)  # (B, 1, SLOT_DIM)
        # slot_queries/type_embedding: (1, token_count, SLOT_DIM) + context: (B, 1, SLOT_DIM) はbroadcastで
        # (B, token_count, SLOT_DIM)になる(バッチ方向の明示的なexpandは不要)
        type_embeddings = self.type_embedding(self.token_types)
        slots = self.slot_queries.unsqueeze(0) + type_embeddings.unsqueeze(0) + context
        slots = self.transformer(slots)
        vertex_slots, stroke_slots = slots[:, : self.shape.vertex_count], slots[:, self.shape.vertex_count :]

        vertex_features = self.vertex_feature_head(vertex_slots)
        vertex_existence_logits = self.vertex_existence_head(vertex_slots).squeeze(-1)
        stroke_offsets = self.stroke_offset_head(stroke_slots)
        stroke_existence_logits = self.stroke_existence_head(stroke_slots).squeeze(-1)

        # 始点・終点で同じkey(頂点トークン)投影を共有する。頂点トークン自体はどちらのポインタから見ても同じであるため
        pointer_key = self.pointer_key_head(vertex_slots)
        start_logits, start_points = self._pointer(
            self.pointer_start_query_head, stroke_slots, pointer_key, vertex_features
        )
        end_logits, end_points = self._pointer(
            self.pointer_end_query_head, stroke_slots, pointer_key, vertex_features
        )

        return DecoderOutput(
            vertex_features,
            vertex_existence_logits,
            stroke_offsets,
            stroke_existence_logits,
            start_logits,
            end_logits,
            start_points,
            end_points,
        )


class VAE(nn.Module):
    def __init__(
        self,
        shape: ModelShape,
        hidden_dims: tuple[int, int],
        latent_dim: int,
        slot_attention_config: SlotAttentionConfig,
        gumbel_temperature: float,
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
        self.decoder = SlotAttentionDecoder(shape, latent_dim, slot_attention_config, gumbel_temperature)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + std * epsという決定的な式にすることで、サンプリングを微分可能にする(再パラメータ化トリック)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> DecoderOutput:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[DecoderOutput, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def flatten_input(
    vertices: torch.Tensor,
    vertex_existence: torch.Tensor,
    stroke_vertex_indices: torch.Tensor,
    stroke_offsets: torch.Tensor,
    stroke_existence: torch.Tensor,
    shape: ModelShape,
) -> torch.Tensor:
    # ストロークの参照先頂点(始点・終点)はカテゴリ変数のため、大小関係を暗示しないone-hotに展開する
    start_onehot = F.one_hot(stroke_vertex_indices[..., 0], num_classes=shape.vertex_count).float()
    end_onehot = F.one_hot(stroke_vertex_indices[..., 1], num_classes=shape.vertex_count).float()
    return torch.cat(
        [
            vertices.flatten(1),
            vertex_existence,
            start_onehot.flatten(1),
            end_onehot.flatten(1),
            stroke_offsets.flatten(1),
            stroke_existence,
        ],
        dim=1,
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
