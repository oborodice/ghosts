#!/usr/bin/env python3
import torch
import torch.nn.functional as F


def compute_loss(
    vertices: torch.Tensor,
    existence: torch.Tensor,
    vertices_recon: torch.Tensor,
    existence_logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    mask = existence.unsqueeze(-1)  # (B, slot_count, 1) -> 座標方向へブロードキャスト
    # 特徴量・スロット方向は和、バッチ方向は平均を取る(sum→batch mean)。
    # 要素方向で平均を取るとKLダイバージェンスに対して再構成損失が相対的に小さくなり、posterior collapseを起こしやすくなるため避ける
    vertex_loss = (((vertices_recon - vertices) ** 2) * mask).sum(dim=(1, 2)).mean()
    existence_loss = (
        F.binary_cross_entropy_with_logits(existence_logits, existence, reduction="none").sum(dim=1).mean()
    )
    kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    return vertex_loss + existence_loss + beta * kl_divergence
