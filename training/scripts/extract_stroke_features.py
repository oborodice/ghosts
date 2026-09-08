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
SLOT_COUNT = 26  # 折れ・ハネ検出による分割後のセグメント数の97パーセンタイル(分割前の画数の97パーセンタイルは22、stroke_count_stats.py参照)
FEATURE_DIM = len(StrokeFeatures._fields)
STROKE_NUMBER_PATTERN = re.compile(r"-s(\d+)$")
# 接続点一致損失用。実データの端点間距離の分布を見ると、閾値未満の距離に「実際に接続している点」の
# 集団が偏っており、4付近を谷にして「接続していない点同士のランダムな距離」の分布に切り替わる
CONNECTION_THRESHOLD = 4.0
# 折れ・ハネの検出用(_detect_corner参照)
CORNER_SAMPLE_COUNT = 60
CORNER_WINDOW = 6
CORNER_ANGLE_THRESHOLD = 90.0  # 実データで「口」等の既知の折れ・ハネを検出できることを確認した値


def _parse_stroke_number(path_element: ET.Element) -> int:
    return int(STROKE_NUMBER_PATTERN.search(path_element.get("id")).group(1))


def _detect_corner(path: SvgPath) -> tuple[float, float]:
    # 等間隔なtでCORNER_SAMPLE_COUNT点の接線方向をサンプリングし、隣接点間の角度差を
    # CORNER_WINDOW点(パス全体の約1/10)の幅で合計した最大値を「最も集中した方向転換」とする。
    # セグメント境界(ベジェ曲線の繋ぎ目)での角度差をそのまま見る方法も試したが、ほとんどのstrokeで
    # ほぼ0度であり機能しなかった。1本の滑らかな曲線が複数のベジェで近似されているだけのケースが
    # ほとんどで、繋ぎ目自体が実際の折れ・ハネの位置とは限らないため、この方式を採用している
    # 戻り値: (最も集中した方向転換の合計角度[度], パス全体に対するその位置の比率[0-1])
    ts = np.linspace(0, 1, CORNER_SAMPLE_COUNT)
    tangents = np.array([path.derivative(t) for t in ts])
    angles = np.angle(tangents)
    diffs = np.abs(np.diff(angles))
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    diffs_deg = np.degrees(diffs)

    window_sums = np.convolve(diffs_deg, np.ones(CORNER_WINDOW), mode="valid")
    max_index = window_sums.argmax()
    # 角度変化が最大のウィンドウの中心位置を、折れ・ハネの位置とする
    position_ratio = (max_index + CORNER_WINDOW / 2) / len(diffs_deg)
    return window_sums[max_index], position_ratio


def _split_at_corner(path: SvgPath) -> list[SvgPath]:
    max_angle, position_ratio = _detect_corner(path)
    if max_angle < CORNER_ANGLE_THRESHOLD:
        return [path]
    return [path.cropped(0, position_ratio), path.cropped(position_ratio, 1)]


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

    # 画数ではなく、折れ・ハネで分割した後のセグメント数がSLOT_COUNTの基準になる
    segments = []
    for path_element in path_elements:
        segments.extend(_split_at_corner(parse_path(path_element.get("d"))))
    if len(segments) > SLOT_COUNT:
        return None

    strokes = np.zeros((SLOT_COUNT, FEATURE_DIM))
    existence = np.zeros(SLOT_COUNT)
    endpoints = np.zeros((len(segments), 2, 2))
    for slot, segment in enumerate(segments):
        strokes[slot] = _compute_stroke_features(segment)
        existence[slot] = 1.0
        endpoints[slot] = [(segment.start.real, segment.start.imag), (segment.end.real, segment.end.imag)]

    # 分割で生まれた同一ストローク内のセグメント間の接続点は、分割点の座標がそのまま一致するため、
    # 異なるストローク同士の接続と同じ距離判定でそのまま検出される(特別扱いは不要)
    connections = np.zeros((2 * SLOT_COUNT, 2 * SLOT_COUNT), dtype=bool)
    connections[: 2 * len(segments), : 2 * len(segments)] = _compute_connections(endpoints)
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
    print(f"Excluded kanji (segment count after corner splitting > {SLOT_COUNT}): {excluded_count}")
    avg_segments = np.array(all_existence).sum(axis=1).mean()
    print(f"Average segments per kanji (after corner splitting): {avg_segments:.2f}")
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
