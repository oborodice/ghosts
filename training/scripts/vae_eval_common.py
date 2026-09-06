#!/usr/bin/env python3
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_vae import load_stroke_features, split_train_val_indices, standardize
from vae_checkpoint import Checkpoint
from vae_model import flatten_input, unflatten_output

# existenceの確率(Sigmoid(existence_logits))をbool判定に変換する閾値。evaluate_vae.pyの正答率算出とも共有する
EXISTENCE_THRESHOLD = 0.5


class ValidationData(NamedTuple):
    strokes: np.ndarray  # 標準化前(可視化の元データ用)
    existence: np.ndarray
    strokes_standardized: torch.Tensor  # モデル入力用
    existence_tensor: torch.Tensor


def load_validation_data(checkpoint: Checkpoint, device: torch.device) -> ValidationData:
    strokes, existence = load_stroke_features()
    # train_vae.pyと同じSEEDでスプリットを再現し、学習に使っていないデータのみを対象にする
    _, val_indices = split_train_val_indices(len(strokes))
    val_strokes, val_existence = strokes[val_indices], existence[val_indices]

    mean, std = checkpoint.mean.cpu().numpy(), checkpoint.std.cpu().numpy()
    val_strokes_standardized = standardize(val_strokes, mean, std)
    return ValidationData(
        val_strokes,
        val_existence,
        torch.tensor(val_strokes_standardized, dtype=torch.float32, device=device),
        torch.tensor(val_existence, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def encode(
    checkpoint: Checkpoint, strokes: torch.Tensor, existence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return checkpoint.model.encode(flatten_input(strokes, existence))


def existence_mask_from_logits(existence_logits: torch.Tensor) -> np.ndarray:
    return (torch.sigmoid(existence_logits) > EXISTENCE_THRESHOLD).cpu().numpy()


def strokes_to_segments(
    strokes: np.ndarray, existence_mask: np.ndarray
) -> list[tuple[complex, complex]]:
    # (start_x, start_y, angle, curvature, length) -> 始点-終点の線分
    # curvatureは弧長と弦長の差のみを保持し曲がる向きは復元できないため、直線近似で描画する
    segments = []
    for (start_x, start_y, angle, curvature, length), exists in zip(strokes, existence_mask):
        if not exists:
            continue
        start = complex(start_x, start_y)
        end = start + (length - curvature) * complex(np.cos(angle), np.sin(angle))
        segments.append((start, end))
    return segments


def draw_segments(ax: plt.Axes, segments: list[tuple[complex, complex]]) -> None:
    for start, end in segments:
        # SVGはy軸が下向きのため、view_kanji.pyと同様上向きに合わせて反転する
        ax.plot([start.real, end.real], [-start.imag, -end.imag], color="black")
    ax.set_aspect("equal")
    ax.axis("off")


def _destandardize(strokes_standardized: torch.Tensor, checkpoint: Checkpoint) -> torch.Tensor:
    return strokes_standardized * checkpoint.std + checkpoint.mean


@torch.no_grad()
def decode_to_segments(
    checkpoint: Checkpoint, z: torch.Tensor
) -> list[list[tuple[complex, complex]]]:
    # zはバッチ(複数サンプル)を想定し、サンプルごとの線分リストを返す
    recon = checkpoint.model.decode(z)
    strokes_recon, existence_logits = unflatten_output(recon, checkpoint.shape)
    strokes_recon = _destandardize(strokes_recon, checkpoint).cpu().numpy()
    existence_mask = existence_mask_from_logits(existence_logits)
    return [
        strokes_to_segments(strokes, mask)
        for strokes, mask in zip(strokes_recon, existence_mask)
    ]
