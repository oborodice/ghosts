#!/usr/bin/env python3
import torch
import torch.nn.functional as F

from vae_model_v2 import DecoderOutput


def pointer_loss(logits: torch.Tensor, target_index: torch.Tensor, stroke_existence: torch.Tensor) -> torch.Tensor:
    # logits: (B, stroke_count, vertex_count) -> cross_entropyのクラス次元(vertex_count)を
    # dim=1に持ってくるためtransposeする
    per_stroke_loss = F.cross_entropy(logits.transpose(1, 2), target_index, reduction="none")
    return (per_stroke_loss * stroke_existence).sum(dim=1).mean()


def compute_loss(
    vertices: torch.Tensor,
    vertex_existence: torch.Tensor,
    stroke_vertex_indices: torch.Tensor,
    stroke_offsets: torch.Tensor,
    stroke_existence: torch.Tensor,
    decoder_output: DecoderOutput,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    # 特徴量・スロット方向は和、バッチ方向は平均を取る(sum→batch mean)。
    # 要素方向で平均を取るとKLダイバージェンスに対して再構成損失が相対的に小さくなり、posterior collapseを起こしやすくなるため避ける
    vertex_mask = vertex_existence.unsqueeze(-1)
    vertex_loss = (((decoder_output.vertex_features - vertices) ** 2) * vertex_mask).sum(dim=(1, 2)).mean()
    vertex_existence_loss = (
        F.binary_cross_entropy_with_logits(decoder_output.vertex_existence_logits, vertex_existence, reduction="none")
        .sum(dim=1)
        .mean()
    )

    # ポインタの参照先はcross entropyで直接教師する(座標自体へのMSEは加えない)。座標の正しさは、
    # ポインタが正解頂点に集中しさえすれば頂点座標MSEを通じて間接的に保証される設計のため
    start_pointer_loss = pointer_loss(
        decoder_output.start_pointer_logits, stroke_vertex_indices[..., 0], stroke_existence
    )
    end_pointer_loss = pointer_loss(
        decoder_output.end_pointer_logits, stroke_vertex_indices[..., 1], stroke_existence
    )

    stroke_offset_mask = stroke_existence.unsqueeze(-1)
    stroke_offset_loss = (
        ((decoder_output.stroke_offsets - stroke_offsets) ** 2) * stroke_offset_mask
    ).sum(dim=(1, 2)).mean()
    stroke_existence_loss = (
        F.binary_cross_entropy_with_logits(
            decoder_output.stroke_existence_logits, stroke_existence, reduction="none"
        )
        .sum(dim=1)
        .mean()
    )

    kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    return (
        vertex_loss
        + vertex_existence_loss
        + start_pointer_loss
        + end_pointer_loss
        + stroke_offset_loss
        + stroke_existence_loss
        + beta * kl_divergence
    )
