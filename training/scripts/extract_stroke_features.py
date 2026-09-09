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
    length: float
    offset_x: float
    offset_y: float


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
# 折れ・ハネの検出用(_find_corners参照)
CORNER_SAMPLE_COUNT = 60
CORNER_WINDOW = 6
CORNER_ANGLE_THRESHOLD = 90.0  # 実データで「口」等の既知の折れ・ハネを検出できることを確認した値
CORNER_MIN_SEPARATION = 0.15  # これ未満しか離れていない2つの角は同一の角とみなし、大きい方だけを残す
MAX_CORNERS_PER_STROKE = 2  # 1ストロークに3箇所以上角があるケースは0.18%のみなので割り切る
# 制御点の最小二乗フィット用サンプル点数。この点数でのフィットにより、2次ベジェ近似の残差が
# ストローク長に対して中央値3.1%程度に収まることを確認済み
CONTROL_POINT_SAMPLE_COUNT = 21


def _parse_stroke_number(path_element: ET.Element) -> int:
    return int(STROKE_NUMBER_PATTERN.search(path_element.get("id")).group(1))


def _corner_angle_profile(path: SvgPath) -> tuple[np.ndarray, np.ndarray]:
    # 等間隔なtでCORNER_SAMPLE_COUNT点の接線方向をサンプリングし、隣接点間の角度差を
    # CORNER_WINDOW点(パス全体の約1/10)の幅で合計する。セグメント境界(ベジェ曲線の繋ぎ目)での
    # 角度差をそのまま見る方法も試したが、ほとんどのstrokeでほぼ0度であり機能しなかった。1本の
    # 滑らかな曲線が複数のベジェで近似されているだけのケースがほとんどで、繋ぎ目自体が実際の
    # 折れ・ハネの位置とは限らないため、この方式を採用している
    # 戻り値: (ウィンドウごとの合計角度変化[度], パス全体に対する各ウィンドウ中心の位置比率[0-1])
    ts = np.linspace(0, 1, CORNER_SAMPLE_COUNT)
    tangents = np.array([path.derivative(t) for t in ts])
    angles = np.angle(tangents)
    diffs = np.abs(np.diff(angles))
    diffs = np.minimum(diffs, 2 * np.pi - diffs)
    diffs_deg = np.degrees(diffs)
    window_sums = np.convolve(diffs_deg, np.ones(CORNER_WINDOW), mode="valid")
    positions = (np.arange(len(window_sums)) + CORNER_WINDOW / 2) / len(diffs_deg)
    return window_sums, positions


def _corner_peaks(window_sums: np.ndarray, positions: np.ndarray) -> list[tuple[float, float]]:
    # 閾値を超えた区間の集まり(山)ごとに最大値の位置を折れ・ハネの候補とする。1本のストロークに
    # 横→縦の折れとハネの両方を持つケース(肉月の二画目など)があるため、山は1つとは限らない
    above_threshold = window_sums >= CORNER_ANGLE_THRESHOLD
    peaks: list[tuple[float, float]] = []
    index = 0
    while index < len(above_threshold):
        if not above_threshold[index]:
            index += 1
            continue
        run_end = index
        while run_end < len(above_threshold) and above_threshold[run_end]:
            run_end += 1
        peak_index = index + window_sums[index:run_end].argmax()
        peaks.append((window_sums[peak_index], positions[peak_index]))
        index = run_end
    return peaks


def _merge_nearby_corners(peaks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    # 隣接する山同士がCORNER_MIN_SEPARATION未満しか離れていない場合は同一の角とみなし、大きい方だけ残す
    corners: list[tuple[float, float]] = []
    for angle, position in peaks:
        if corners and position - corners[-1][1] < CORNER_MIN_SEPARATION:
            if angle > corners[-1][0]:
                corners[-1] = (angle, position)
        else:
            corners.append((angle, position))
    return corners


def _find_corners(path: SvgPath) -> list[tuple[float, float]]:
    # 戻り値: [(方向転換の合計角度[度], パス全体に対するその位置の比率[0-1]), ...]
    window_sums, positions = _corner_angle_profile(path)
    peaks = _corner_peaks(window_sums, positions)
    corners = _merge_nearby_corners(peaks)
    corners.sort(key=lambda corner: corner[0], reverse=True)
    return corners[:MAX_CORNERS_PER_STROKE]


def _split_at_corner(path: SvgPath) -> list[SvgPath]:
    corners = _find_corners(path)
    if not corners:
        return [path]
    # _find_cornersの戻り値は角度の大きい順なので、分割点として使うにはパス上の位置順に並べ直す
    positions = sorted(position for _, position in corners)
    boundaries = [0.0, *positions, 1.0]
    return [path.cropped(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _fit_control_point_offset(path: SvgPath, start: complex, end: complex) -> complex:
    # 2次ベジェ B(t) = (1-t)^2*start + 2(1-t)t*p1 + t^2*end はp1について線形なので、
    # パス上のサンプル点群から最小二乗でp1を直接解ける
    ts = np.linspace(0, 1, CONTROL_POINT_SAMPLE_COUNT)
    points = np.array([path.point(t) for t in ts])
    basis = 2 * (1 - ts) * ts
    target = points - (1 - ts) ** 2 * start - ts**2 * end
    control_point = (basis * target).sum() / (basis * basis).sum()
    return complex(control_point - (start + end) / 2)


def _compute_stroke_features(path: SvgPath) -> StrokeFeatures:
    start, end = path.start, path.end
    chord = end - start
    angle = np.angle(chord)
    length = abs(chord)
    offset = _fit_control_point_offset(path, start, end)
    return StrokeFeatures(start.real, start.imag, angle, length, offset.real, offset.imag)


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
