#!/usr/bin/env python3
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from vae_checkpoint import Checkpoint
from vae_data import load_stroke_features, split_train_val_indices, standardize
from vae_model import flatten_input, unflatten_output

# existenceの確率(Sigmoid(existence_logits))をbool判定に変換する閾値。evaluate_vae.pyの正答率算出とも共有する
EXISTENCE_THRESHOLD = 0.5

# 生成時にzを実データへ引き寄せるカーネル幅(実データ同士の最近傍距離の中央値を目安に選んだ値)。
# export_onnx.pyのエクスポート済みグラフにもこの値がそのまま焼き込まれる
KERNEL_BANDWIDTH = 0.6


class SplitData(NamedTuple):
    strokes: np.ndarray  # 標準化前(可視化・誤差計算の元データ用)
    existence: np.ndarray
    connections: np.ndarray
    strokes_standardized: torch.Tensor  # モデル入力用
    existence_tensor: torch.Tensor


def _build_split_data(
    indices: np.ndarray,
    strokes: np.ndarray,
    existence: np.ndarray,
    connections: np.ndarray,
    checkpoint: Checkpoint,
    device: torch.device,
) -> SplitData:
    split_strokes, split_existence = strokes[indices], existence[indices]
    mean, std = checkpoint.mean.cpu().numpy(), checkpoint.std.cpu().numpy()
    split_strokes_standardized = standardize(split_strokes, mean, std)
    return SplitData(
        split_strokes,
        split_existence,
        connections[indices],
        torch.tensor(split_strokes_standardized, dtype=torch.float32, device=device),
        torch.tensor(split_existence, dtype=torch.float32, device=device),
    )


def load_validation_data(checkpoint: Checkpoint, device: torch.device) -> SplitData:
    strokes, existence, connections = load_stroke_features()
    # vae_data.pyと同じSEEDでスプリットを再現し、学習に使っていないデータのみを対象にする
    _, val_indices = split_train_val_indices(len(strokes))
    return _build_split_data(val_indices, strokes, existence, connections, checkpoint, device)


def load_train_data(checkpoint: Checkpoint, device: torch.device) -> SplitData:
    strokes, existence, connections = load_stroke_features()
    # vae_data.pyと同じSEEDでスプリットを再現し、学習に使ったデータのみを対象にする
    # (丸暗記化の確認、生成時のカーネル重み付けに使う実データ全体のencode結果の取得などに使う)
    train_indices, _ = split_train_val_indices(len(strokes))
    return _build_split_data(train_indices, strokes, existence, connections, checkpoint, device)


@torch.no_grad()
def encode(
    checkpoint: Checkpoint, strokes: torch.Tensor, existence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return checkpoint.model.encode(flatten_input(strokes, existence))


@torch.no_grad()
def attract_to_latent_prior(z_raw: torch.Tensor, mu_real: torch.Tensor) -> torch.Tensor:
    # Nadaraya-Watson推定量。export_onnx.pyの_GenerationModelが生成グラフに焼き込む処理でもこの関数を使う
    dist_sq = torch.cdist(z_raw, mu_real) ** 2
    weights = torch.softmax(-dist_sq / (2 * KERNEL_BANDWIDTH * KERNEL_BANDWIDTH), dim=1)
    return weights @ mu_real


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


def _stroke_endpoints_array(strokes: np.ndarray) -> np.ndarray:
    # 戻り値のshapeは(slot_count, 2, 2) -- [スロット, 始点(0)/終点(1), xy]。
    # connections行列の点index(偶数=始点, 奇数=終点)と対応させるため、existenceに関わらず全スロット分計算する
    start = strokes[:, 0:2]
    angle = strokes[:, 2]
    curvature = strokes[:, 3]
    length = strokes[:, 4]
    radius = length - curvature
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    end = start + radius[:, None] * direction
    return np.stack([start, end], axis=1)


def connection_centers(strokes: np.ndarray, connections: np.ndarray) -> list[complex]:
    # connectionsは上三角のみが立っている(extract_stroke_features.py参照)ので、立っている
    # 各ペアについて2点の中点をそのままズームイン表示の中心として返せばよい(重複は発生しない)
    points = _stroke_endpoints_array(strokes).reshape(-1, 2)
    pair_indices = np.argwhere(connections)
    return [complex(*((points[i] + points[j]) / 2)) for i, j in pair_indices]


def draw_segments_zoomed(
    ax: plt.Axes, segments: list[tuple[complex, complex]], center: complex, margin: float
) -> None:
    # 接続点・交差点は文字全体のサムネイルでは小さすぎて崩れが見えないことがあるため、
    # 特定の点の周辺だけを拡大表示する
    draw_segments(ax, segments)
    ax.set_xlim(center.real - margin, center.real + margin)
    ax.set_ylim(-center.imag - margin, -center.imag + margin)


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
