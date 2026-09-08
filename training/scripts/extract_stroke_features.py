#!/usr/bin/env python3
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
from svgpathtools import Path as SvgPath, parse_path


class StrokeFeatures(NamedTuple):
    start_x: float
    start_y: float
    angle: float
    curvature: float
    length: float


class KanjiTensors(NamedTuple):
    strokes: np.ndarray
    existence: np.ndarray
    connections: np.ndarray


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kanjivg"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stroke_features.npz"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
SLOT_COUNT = 22  # 画数の97パーセンタイル(stroke_count_stats.py参照)
FEATURE_DIM = len(StrokeFeatures._fields)
STROKE_NUMBER_PATTERN = re.compile(r"-s(\d+)$")
# 接続点一致損失用。実データの端点間距離の分布を見ると、閾値未満の距離に「実際に接続している点」の
# 集団が偏っており、4付近を谷にして「接続していない点同士のランダムな距離」の分布に切り替わる
CONNECTION_THRESHOLD = 4.0


def _parse_stroke_number(path_element: ET.Element) -> int:
    return int(STROKE_NUMBER_PATTERN.search(path_element.get("id")).group(1))


def _compute_stroke_features(path: SvgPath) -> StrokeFeatures:
    start, end = path.start, path.end
    length = path.length()
    # 弦長で割る比ではなく差を使う(ごく稀に始点と終点が一致するループ状ストロークがあり、比だと発散するため)
    curvature = length - abs(end - start)
    angle = np.angle(end - start)
    return StrokeFeatures(start.real, start.imag, angle, curvature, length)


def _compute_connections(endpoints: np.ndarray) -> np.ndarray:
    # endpoints: (stroke_count, 2, 2) -- [スロット, 始点(0)/終点(1), xy]。
    # 戻り値のshapeは(stroke_count*2, stroke_count*2)で、偶数index=始点・奇数index=終点に対応する
    stroke_count = endpoints.shape[0]
    points = endpoints.reshape(stroke_count * 2, 2)
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    connected = distance < CONNECTION_THRESHOLD
    for slot in range(stroke_count):
        # 同じストロークの始点・終点同士を接続対象にすると、そのストローク自体の長さを0に潰す力が働いてしまう
        connected[2 * slot, 2 * slot + 1] = False
        connected[2 * slot + 1, 2 * slot] = False
    np.fill_diagonal(connected, False)
    return np.triu(connected)  # 対称行列なので上三角のみ残し、同じペアを2回カウントしないようにする


def _extract_kanji_tensors(svg_path: Path) -> KanjiTensors | None:
    root = ET.parse(svg_path).getroot()
    path_elements = sorted(root.findall(f".//{SVG_NAMESPACE}path"), key=_parse_stroke_number)
    if len(path_elements) > SLOT_COUNT:
        return None

    strokes = np.zeros((SLOT_COUNT, FEATURE_DIM))
    existence = np.zeros(SLOT_COUNT)
    endpoints = np.zeros((len(path_elements), 2, 2))
    for slot, path_element in enumerate(path_elements):
        path = parse_path(path_element.get("d"))
        strokes[slot] = _compute_stroke_features(path)
        existence[slot] = 1.0
        endpoints[slot] = [(path.start.real, path.start.imag), (path.end.real, path.end.imag)]

    stroke_count = len(path_elements)
    connections = np.zeros((2 * SLOT_COUNT, 2 * SLOT_COUNT), dtype=bool)
    connections[: 2 * stroke_count, : 2 * stroke_count] = _compute_connections(endpoints)
    return KanjiTensors(strokes, existence, connections)


def main() -> None:
    all_strokes = []
    all_existence = []
    all_connections = []
    excluded_count = 0

    for svg_path in DATA_DIR.glob("*.svg"):
        kanji_tensors = _extract_kanji_tensors(svg_path)
        if kanji_tensors is None:
            excluded_count += 1
            continue
        strokes, existence, connections = kanji_tensors
        all_strokes.append(strokes)
        all_existence.append(existence)
        all_connections.append(connections)

    print(f"Included kanji: {len(all_strokes)}")
    print(f"Excluded kanji (stroke count > {SLOT_COUNT}): {excluded_count}")
    connections_array = np.array(all_connections)
    avg_connections = connections_array.sum(axis=(1, 2)).mean()
    print(f"Average detected connections per kanji: {avg_connections:.2f}")

    np.savez_compressed(
        OUTPUT_PATH,
        strokes=np.array(all_strokes),
        existence=np.array(all_existence),
        connections=connections_array,
    )
    print(f"Saved stroke features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
