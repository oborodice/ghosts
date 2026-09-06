#!/usr/bin/env python3
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
from svgpathtools import parse_path


class StrokeFeatures(NamedTuple):
    start_x: float
    start_y: float
    angle: float
    curvature: float
    length: float


class KanjiTensors(NamedTuple):
    strokes: np.ndarray
    existence: np.ndarray


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kanjivg"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stroke_features.npz"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
SLOT_COUNT = 22  # 画数の97パーセンタイル(stroke_count_stats.py参照)
FEATURE_DIM = len(StrokeFeatures._fields)
STROKE_NUMBER_PATTERN = re.compile(r"-s(\d+)$")


def parse_stroke_number(path_element: ET.Element) -> int:
    return int(STROKE_NUMBER_PATTERN.search(path_element.get("id")).group(1))


def compute_stroke_features(path_data: str) -> StrokeFeatures:
    path = parse_path(path_data)
    start, end = path.start, path.end
    length = path.length()
    # 弦長で割る比ではなく差を使う(ごく稀に始点と終点が一致するループ状ストロークがあり、比だと発散するため)
    curvature = length - abs(end - start)
    angle = np.angle(end - start)
    return StrokeFeatures(start.real, start.imag, angle, curvature, length)


def extract_kanji_tensors(svg_path: Path) -> KanjiTensors | None:
    root = ET.parse(svg_path).getroot()
    path_elements = sorted(root.findall(f".//{SVG_NAMESPACE}path"), key=parse_stroke_number)
    if len(path_elements) > SLOT_COUNT:
        return None

    strokes = np.zeros((SLOT_COUNT, FEATURE_DIM))
    existence = np.zeros(SLOT_COUNT)
    for slot, path_element in enumerate(path_elements):
        strokes[slot] = compute_stroke_features(path_element.get("d"))
        existence[slot] = 1.0
    return KanjiTensors(strokes, existence)


def main() -> None:
    all_strokes = []
    all_existence = []
    excluded_count = 0

    for svg_path in DATA_DIR.glob("*.svg"):
        kanji_tensors = extract_kanji_tensors(svg_path)
        if kanji_tensors is None:
            excluded_count += 1
            continue
        strokes, existence = kanji_tensors
        all_strokes.append(strokes)
        all_existence.append(existence)

    print(f"Included kanji: {len(all_strokes)}")
    print(f"Excluded kanji (stroke count > {SLOT_COUNT}): {excluded_count}")

    np.savez_compressed(
        OUTPUT_PATH,
        strokes=np.array(all_strokes),
        existence=np.array(all_existence),
    )
    print(f"Saved stroke features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
