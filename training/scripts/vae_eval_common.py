#!/usr/bin/env python3
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

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


def strokes_to_curves(
    strokes: np.ndarray, existence_mask: np.ndarray
) -> list[tuple[complex, complex, complex]]:
    # (start_x, start_y, angle, length, offset_x, offset_y) -> (始点, 制御点, 終点)の2次ベジェ
    # lengthは弦(始点-終点間)の長さ、制御点は弦の中点をoffset_x/offset_yだけずらした点
    curves = []
    for (start_x, start_y, angle, length, offset_x, offset_y), exists in zip(strokes, existence_mask):
        if not exists:
            continue
        start = complex(start_x, start_y)
        end = start + length * complex(np.cos(angle), np.sin(angle))
        control = (start + end) / 2 + complex(offset_x, offset_y)
        curves.append((start, control, end))
    return curves


def draw_curves(ax: plt.Axes, curves: list[tuple[complex, complex, complex]]) -> None:
    for start, control, end in curves:
        # SVGはy軸が下向きのため、view_kanji.pyと同様上向きに合わせて反転する
        path = MplPath(
            [(start.real, -start.imag), (control.real, -control.imag), (end.real, -end.imag)],
            [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
        )
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor="black"))
    # add_patchはax.plotと違ってビューを自動追従しないため、明示的にdataLimへ合わせる
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def _stroke_endpoints_array(strokes: np.ndarray) -> np.ndarray:
    # 戻り値のshapeは(slot_count, 2, 2) -- [スロット, 始点(0)/終点(1), xy]。
    # connections行列の点index(偶数=始点, 奇数=終点)と対応させるため、existenceに関わらず全スロット分計算する
    start = strokes[:, 0:2]
    angle = strokes[:, 2]
    length = strokes[:, 3]
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    end = start + length[:, None] * direction
    return np.stack([start, end], axis=1)


def connection_centers(strokes: np.ndarray, connections: np.ndarray) -> list[complex]:
    # connectionsは上三角のみが立っている(extract_stroke_features.py参照)ので、立っている
    # 各ペアについて2点の中点をそのままズームイン表示の中心として返せばよい(重複は発生しない)
    points = _stroke_endpoints_array(strokes).reshape(-1, 2)
    pair_indices = np.argwhere(connections)
    return [complex(*((points[i] + points[j]) / 2)) for i, j in pair_indices]


def draw_curves_zoomed(
    ax: plt.Axes, curves: list[tuple[complex, complex, complex]], center: complex, margin: float
) -> None:
    # 接続点・交差点は文字全体のサムネイルでは小さすぎて崩れが見えないことがあるため、
    # 特定の点の周辺だけを拡大表示する
    draw_curves(ax, curves)
    ax.set_xlim(center.real - margin, center.real + margin)
    ax.set_ylim(-center.imag - margin, -center.imag + margin)


def _destandardize(strokes_standardized: torch.Tensor, checkpoint: Checkpoint) -> torch.Tensor:
    return strokes_standardized * checkpoint.std + checkpoint.mean


@torch.no_grad()
def decode_to_curves(
    checkpoint: Checkpoint, z: torch.Tensor
) -> list[list[tuple[complex, complex, complex]]]:
    # zはバッチ(複数サンプル)を想定し、サンプルごとの曲線リストを返す
    recon = checkpoint.model.decode(z)
    strokes_recon, existence_logits = unflatten_output(recon, checkpoint.shape)
    strokes_recon = _destandardize(strokes_recon, checkpoint).cpu().numpy()
    existence_mask = existence_mask_from_logits(existence_logits)
    return [
        strokes_to_curves(strokes, mask)
        for strokes, mask in zip(strokes_recon, existence_mask)
    ]
