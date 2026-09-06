#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from svgpathtools import svg2paths

# 画数や字形(直線/曲線/点/構成要素の複雑さ)に偏りが出ないよう選定
KANJI = ["一", "人", "水", "心", "愛", "鬱"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kanjivg"
SAMPLES_PER_STROKE = 100


def _draw_kanji(ax: plt.Axes, char: str) -> None:
    svg_path = DATA_DIR / f"{ord(char):05x}.svg"
    strokes, _ = svg2paths(str(svg_path))
    for stroke in strokes:
        points = np.array(
            [stroke.point(t / SAMPLES_PER_STROKE) for t in range(SAMPLES_PER_STROKE + 1)]
        )
        # SVGはy軸が下向きのため、上向きに合わせて反転する
        ax.plot(points.real, -points.imag, color="black")
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    _, axes = plt.subplots(nrows=1, ncols=len(KANJI))
    for ax, char in zip(axes, KANJI):
        _draw_kanji(ax, char)
    plt.show()


if __name__ == "__main__":
    main()
