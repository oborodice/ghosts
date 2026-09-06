#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kanjivg"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
PERCENTILES = (90, 95, 97, 99)


def _count_strokes(svg_path: Path) -> int:
    root = ET.parse(svg_path).getroot()
    return len(root.findall(f".//{SVG_NAMESPACE}path"))


def main() -> None:
    stroke_counts = np.array([_count_strokes(path) for path in DATA_DIR.glob("*.svg")])

    print(f"Kanji count: {len(stroke_counts)}")
    print(f"Min strokes: {stroke_counts.min()}")
    print(f"Max strokes: {stroke_counts.max()}")
    print(f"Mean strokes: {stroke_counts.mean():.2f}")
    for percentile in PERCENTILES:
        print(f"{percentile}th percentile: {np.percentile(stroke_counts, percentile):.1f}")


if __name__ == "__main__":
    main()
