#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from vae_checkpoint import load_checkpoint
from vae_eval_common import (
    connection_centers,
    decode_to_segments,
    draw_segments,
    draw_segments_zoomed,
    encode,
    load_validation_data,
    strokes_to_segments,
)
from vae_model import select_device

# 平均的なサンプルに加え、再構成誤差が最悪だったindex 73も含める
SAMPLE_INDICES = [0, 1, 2, 3, 73]
ZOOM_MARGIN = 15.0  # ズームイン表示で中心から上下左右に取る範囲
MAX_ZOOMS_PER_SAMPLE = 4  # 1サンプルあたり表示する接続点の最大数(画数が多い字で図が煩雑になるのを防ぐ)


def _draw_connection_zooms(
    index: int,
    original_segments: list[tuple[complex, complex]],
    reconstructed_segments: list[tuple[complex, complex]],
    strokes: np.ndarray,
    connections: np.ndarray,
) -> None:
    # 接続点・交差点は文字全体のサムネイルでは小さすぎて崩れが見えないことがあるため、
    # 個々の接続点周辺を拡大して元データと再構成を見比べられるようにする
    centers = connection_centers(strokes, connections)[:MAX_ZOOMS_PER_SAMPLE]
    if not centers:
        return
    _, axes = plt.subplots(nrows=2, ncols=len(centers), squeeze=False)
    for col, center in enumerate(centers):
        draw_segments_zoomed(axes[0, col], original_segments, center, ZOOM_MARGIN)
        axes[0, col].set_title(f"idx {index} conn {col}")
        draw_segments_zoomed(axes[1, col], reconstructed_segments, center, ZOOM_MARGIN)


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    data = load_validation_data(checkpoint, device)

    strokes = data.strokes_standardized[SAMPLE_INDICES]
    existence = data.existence_tensor[SAMPLE_INDICES]
    # 再パラメータ化のサンプリングεによるランダム性を排除するため、muをそのままdecodeする
    mu, _ = encode(checkpoint, strokes, existence)
    reconstructed_segments = decode_to_segments(checkpoint, mu)

    _, axes = plt.subplots(nrows=2, ncols=len(SAMPLE_INDICES))
    for col, index in enumerate(SAMPLE_INDICES):
        original_mask = data.existence[index].astype(bool)
        original_segments = strokes_to_segments(data.strokes[index], original_mask)
        draw_segments(axes[0, col], original_segments)
        axes[0, col].set_title(f"idx {index}")
        draw_segments(axes[1, col], reconstructed_segments[col])
        _draw_connection_zooms(
            index, original_segments, reconstructed_segments[col], data.strokes[index], data.connections[index]
        )

    plt.show()


if __name__ == "__main__":
    main()
