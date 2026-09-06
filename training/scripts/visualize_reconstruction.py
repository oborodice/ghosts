#!/usr/bin/env python3
import matplotlib.pyplot as plt

from vae_eval_common import (
    decode_to_segments,
    draw_segments,
    encode,
    load_checkpoint,
    load_validation_data,
    strokes_to_segments,
)
from vae_model import select_device

# 平均的なサンプルに加え、再構成誤差が最悪だったindex 73も含める
SAMPLE_INDICES = [0, 1, 2, 3, 73]


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
    plt.show()


if __name__ == "__main__":
    main()
