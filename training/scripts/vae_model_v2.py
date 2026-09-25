#!/usr/bin/env python3
# 頂点+ストロークのデコーダは、今後(頂点のみ→ソフトポインタ→ハードポインタの順で)クエリ数や
# 出力ヘッドの構成が段階的に増えていく前提のため、既存の学習パイプラインとは独立した専用モジュールとして
# 定義する。現時点(ハードポインタ)は、頂点トークン(座標+existence)とストロークトークン
# (参照先頂点へのポインタ+オフセット+existence)の2種類を同じSelf-Attentionスタックに通す。
# ポインタは学習時のみStraight-Through Gumbel-Softmaxで離散化し、推論時(model.eval())はノイズ無しの
# argmaxにする(ONNX変換後のargmax+gatherと一致させるため)。生成時は、decode(soft_temperature=...)で、
# この選択を確率加重平均に置き換えられる
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 容量・repulsion・βの組み合わせsweepで、他の7候補を上回った値。ただしbandwidth
# (vae_eval_common_v2.KERNEL_BANDWIDTH)はこのsweep全体でLATENT_DIM=48向けに較正した値を一律流用
# したままで、96次元向けに個別較正すればさらに良くなる可能性は未検証。なお、保存済みチェックポイントは
# 学習時点のhidden_dims/latent_dim/slot_attention_configを保持しており、このファイルの現在値には追従しない
HIDDEN_DIMS: tuple[int, int] = (2048, 1024)
LATENT_DIM = 96

SLOT_DIM = 512
SLOT_ATTENTION_HEADS = 4
SLOT_ATTENTION_LAYERS = 2
SLOT_ATTENTION_FFN_DIM = 512

VERTEX_TOKEN_TYPE = 0
STROKE_TOKEN_TYPE = 1

# 頂点座標を回帰でなく、標準化空間[-VERTEX_BIN_RANGE, VERTEX_BIN_RANGE]を
# VERTEX_BIN_COUNT分割したビンの分類として扱う(MSEの「平均への回帰」を、cross-entropy+argmaxの
# 「最頻値への収束」に置き換える設計)。範囲・分割数は実データの標準化後の座標分布(実測で概ね±2.1)に
# 余裕を持たせつつ、ビン幅が実スケール換算で約2.5単位(現行のMSEベース頂点再構成誤差13〜21より
# 十分小さい)になるよう較正した値
VERTEX_BIN_COUNT = 50
VERTEX_BIN_RANGE = 2.5
VERTEX_BIN_EDGES = torch.linspace(-VERTEX_BIN_RANGE, VERTEX_BIN_RANGE, VERTEX_BIN_COUNT + 1)
VERTEX_BIN_CENTERS = (VERTEX_BIN_EDGES[:-1] + VERTEX_BIN_EDGES[1:]) / 2

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
    vertex_features: torch.Tensor  # (B, vertex_count, vertex_feature_dim) -- ビン選択をソフトデコードした連続値(標準化空間)
    vertex_x_logits: torch.Tensor  # (B, vertex_count, VERTEX_BIN_COUNT)
    vertex_y_logits: torch.Tensor  # (B, vertex_count, VERTEX_BIN_COUNT)
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

        self.vertex_x_head = nn.Linear(config.slot_dim, VERTEX_BIN_COUNT)
        self.vertex_y_head = nn.Linear(config.slot_dim, VERTEX_BIN_COUNT)
        self.register_buffer("vertex_bin_centers", VERTEX_BIN_CENTERS.clone())
        self.vertex_existence_head = nn.Linear(config.slot_dim, 1)
        self.stroke_offset_head = nn.Linear(config.slot_dim, shape.stroke_feature_dim)
        self.stroke_existence_head = nn.Linear(config.slot_dim, 1)

        # ストロークの始点・終点それぞれについて、頂点トークンへの注意(スケール済み内積)で参照先を選ぶ
        self.pointer_key_head = nn.Linear(config.slot_dim, config.slot_dim)
        self.pointer_start_query_head = nn.Linear(config.slot_dim, config.slot_dim)
        self.pointer_end_query_head = nn.Linear(config.slot_dim, config.slot_dim)

    def _selection_weights(self, logits: torch.Tensor, soft_temperature: float | None) -> torch.Tensor:
        # ポインタ選択・頂点座標のビン選択の両方で使う共通ロジック。選択の重み(one-hotまたは確率)を返す。
        # soft_temperatureがNoneなら、one-hot(ハード)。学習時はStraight-Through Gumbel-Softmax
        # (前向き計算はone-hot、逆伝播はハード化前のsoftmax分布を通じて勾配が流れる)、推論時は
        # Gumbelノイズを乗せず決定的なargmax(ONNX変換後のargmax+gatherと一致させるため)。
        # 指定があれば、one-hotの代わりにsoftmax(logits / soft_temperature)の確率をそのまま返す
        if soft_temperature is not None:
            return torch.softmax(logits / soft_temperature, dim=-1)
        if self.training:
            return F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=True, dim=-1)
        return F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(logits.dtype)

    def _vertex_position(self, logits: torch.Tensor, soft_temperature: float | None) -> torch.Tensor:
        # 頂点座標をビンの分類として扱う。selection自体はSTEで微分可能だが、この関数が返す値は
        # forward側で呼び出し元がdetachする前提(頂点座標を教師する損失はlogitsを直接見るcross
        # entropyのみで、この座標値自体を経由する下流の損失には学習させない設計のため)
        selection = self._selection_weights(logits, soft_temperature)
        return (selection * self.vertex_bin_centers).sum(dim=-1)

    def _pointer(
        self,
        query_head: nn.Linear,
        stroke_slots: torch.Tensor,
        key: torch.Tensor,
        vertex_features: torch.Tensor,
        soft_temperature: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # スケール済みドット積アテンション(Transformerの標準的な手法)で、ストロークトークンから
        # 頂点トークンへの注意スコアを求める。scaleが無いと次元数が大きいほど内積の分散が増え、
        # softmax後の分布が極端に鋭くなりすぎて勾配が消える
        query = query_head(stroke_slots)  # (B, stroke_count, SLOT_DIM)
        scale = self.pointer_key_head.out_features**0.5
        logits = torch.bmm(query, key.transpose(1, 2)) / scale
        selection = self._selection_weights(logits, soft_temperature)

        # selectionをdetachすることで、points(→start_points/end_points)を経由する下流の損失からlogits
        # への逆流を断つ。「どの頂点を参照するか」を学習させたい唯一の損失(pointer_loss)は、この経路を
        # 経由せずlogitsを直接見て学習する。なおvertex_features自体も呼び出し元で既にdetach済みのため、
        # pointsはどちらの経路からも下流損失の勾配を受け取らない
        points = torch.bmm(selection.detach(), vertex_features)
        return logits, points

    def forward(
        self, z: torch.Tensor, detach_slots: bool = False, soft_temperature: float | None = None
    ) -> DecoderOutput:
        # soft_temperature: 指定すると、ポインタ選択・頂点座標のビン選択が、argmaxではなく確率加重平均になる
        # (温度が小さいほどargmaxに近づく)。zを動かしたときの出力が連続になる。生成時にだけ指定し、
        # 学習・validationでは指定しない(validation lossの意味が変わるため)
        context = self.z_to_context(z).unsqueeze(1)  # (B, 1, SLOT_DIM)
        # slot_queries/type_embedding: (1, token_count, SLOT_DIM) + context: (B, 1, SLOT_DIM) はbroadcastで
        # (B, token_count, SLOT_DIM)になる(バッチ方向の明示的なexpandは不要)
        type_embeddings = self.type_embedding(self.token_types)
        slots = self.slot_queries.unsqueeze(0) + type_embeddings.unsqueeze(0) + context
        slots = self.transformer(slots)
        # detach_slotsは、vertex_features・pointer_key等の複数の出力ヘッドが共有するこの
        # self-attention出力を経由して、片方のヘッドだけを教師したい損失の勾配がもう片方の
        # ヘッドの入力(=Transformer本体)まで意図せず遡ってしまうのを防ぐためのオプション。
        # 各ヘッド自身の重みには引き続き勾配が届く(入力をdetachしても、そこから先の
        # 線形変換自体は通常通り学習される)。既定はFalse(全ヘッドを通常通り共同学習する)
        if detach_slots:
            slots = slots.detach()
        vertex_slots, stroke_slots = slots[:, : self.shape.vertex_count], slots[:, self.shape.vertex_count :]

        vertex_x_logits = self.vertex_x_head(vertex_slots)
        vertex_y_logits = self.vertex_y_head(vertex_slots)
        # vertex_featuresはdetachする: 頂点座標を教師するcross entropy損失(vertex_x/y_logitsを直接
        # 見る)はこのdetachの影響を受けないが、vertex_repulsion_loss・start_points/end_points経由の
        # angle_naturalness_loss等、この値を経由する全ての下流損失からの逆流を断つ。ポインタ機構の
        # selection.detach()と同じ設計思想(頂点位置の学習は自身の分類損失だけに委ねる)
        vertex_features = torch.stack(
            [
                self._vertex_position(vertex_x_logits, soft_temperature),
                self._vertex_position(vertex_y_logits, soft_temperature),
            ],
            dim=-1,
        ).detach()
        vertex_existence_logits = self.vertex_existence_head(vertex_slots).squeeze(-1)
        stroke_offsets = self.stroke_offset_head(stroke_slots)
        stroke_existence_logits = self.stroke_existence_head(stroke_slots).squeeze(-1)

        # 始点・終点で同じkey(頂点トークン)投影を共有する。頂点トークン自体はどちらのポインタから見ても同じであるため
        pointer_key = self.pointer_key_head(vertex_slots)
        start_logits, start_points = self._pointer(
            self.pointer_start_query_head, stroke_slots, pointer_key, vertex_features, soft_temperature
        )
        end_logits, end_points = self._pointer(
            self.pointer_end_query_head, stroke_slots, pointer_key, vertex_features, soft_temperature
        )

        return DecoderOutput(
            vertex_features,
            vertex_x_logits,
            vertex_y_logits,
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

    def decode(
        self, z: torch.Tensor, detach_slots: bool = False, soft_temperature: float | None = None
    ) -> DecoderOutput:
        return self.decoder(z, detach_slots=detach_slots, soft_temperature=soft_temperature)

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
