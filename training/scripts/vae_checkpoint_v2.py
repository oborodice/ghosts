#!/usr/bin/env python3
from pathlib import Path
from typing import NamedTuple

import torch

from vae_model_v2 import CHECKPOINT_PATH, VAE, ModelShape, SlotAttentionConfig


class Checkpoint(NamedTuple):
    model: VAE
    shape: ModelShape
    hidden_dims: tuple[int, int]
    slot_attention_config: SlotAttentionConfig
    vertex_mean: torch.Tensor
    vertex_std: torch.Tensor
    stroke_offset_mean: torch.Tensor
    stroke_offset_std: torch.Tensor
    latent_dim: int


def load_checkpoint(device: torch.device, checkpoint_path: Path = CHECKPOINT_PATH) -> Checkpoint:
    # checkpoint_pathはデフォルトでこのモジュールの標準チェックポイントを指すが、複数候補を比較する
    # 用途で、候補ごとのチェックポイントを個別に読み込めるよう、明示的に上書きできるようにしている
    checkpoint = torch.load(checkpoint_path, map_location=device)
    shape = ModelShape(*checkpoint["shape"])
    slot_attention_config = SlotAttentionConfig(*checkpoint["slot_attention_config"])
    model = VAE(shape, checkpoint["hidden_dims"], checkpoint["latent_dim"], slot_attention_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return Checkpoint(
        model,
        shape,
        checkpoint["hidden_dims"],
        slot_attention_config,
        checkpoint["vertex_mean"].to(device),
        checkpoint["vertex_std"].to(device),
        checkpoint["stroke_offset_mean"].to(device),
        checkpoint["stroke_offset_std"].to(device),
        checkpoint["latent_dim"],
    )


def save_checkpoint(
    model: VAE,
    shape: ModelShape,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
    checkpoint_path: Path,
) -> None:
    # 生成(推論)に必要な情報のみを保存する(学習再開用のoptimizer状態・epoch数などは含まない)。
    # hidden_dims・latent_dim・checkpoint_pathはこのモジュールのグローバル定数ではなく呼び出し元から
    # 明示的に受け取ることで、異なるハイパーパラメータで構築したモデルを異なる保存先へ保存する場合
    # (複数候補を比較する用途など)にも安全に使えるようにしている
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            # torch.loadのweights_only=True(デフォルト)はNamedTupleサブクラスを許可しないため、
            # プレーンなtupleとして保存する
            "slot_attention_config": tuple(slot_attention_config),
            "shape": tuple(shape),
            "vertex_mean": vertex_mean.detach().cpu(),
            "vertex_std": vertex_std.detach().cpu(),
            "stroke_offset_mean": stroke_offset_mean.detach().cpu(),
            "stroke_offset_std": stroke_offset_std.detach().cpu(),
        },
        checkpoint_path,
    )
