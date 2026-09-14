#!/usr/bin/env python3
# 各ストロークを独立した(始点, 終点, 角度, 長さ, オフセット)として表現する代わりに、接続されている
# 端点同士をクラスタ化した「頂点」テーブルと、頂点ペアを参照する「ストローク」テーブルに変換する。
# 同じ頂点を参照するストローク同士は必ず接続している、という保証をデータ構造として持たせるための前処理
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import networkx as nx
import numpy as np
from svgpathtools import Path as SvgPath, parse_path


class KanjiTensors(NamedTuple):
    vertices: np.ndarray  # (VERTEX_COUNT, FEATURE_DIM)
    vertex_existence: np.ndarray  # (VERTEX_COUNT,)
    stroke_vertex_indices: np.ndarray  # (SLOT_COUNT, POINTS_PER_SEGMENT) -- 始点/終点それぞれが指す頂点スロット番号
    stroke_offsets: np.ndarray  # (SLOT_COUNT, FEATURE_DIM)
    stroke_existence: np.ndarray  # (SLOT_COUNT,)


class _ClusteringDiagnostics(NamedTuple):
    max_vertex_error: float  # 元の端点座標と、割り当てられた頂点(重心)座標との距離の最大値
    max_clique_diameter: float  # 同じクリークに属す端点同士の距離の最大値(構造上CONNECTION_THRESHOLD未満のはず)


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kanjivg"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "stroke_features_v2.npz"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
SLOT_COUNT = 26  # 折れ・ハネ検出による分割後のセグメント数の97パーセンタイル(分割前の画数の97パーセンタイルは22)
VERTEX_COUNT = 40  # 実データでクリークベースのクラスタリングを行った場合の頂点数は97パーセンタイルで39
# (SLOT_COUNTと同じ「97パーセンタイル基準」の考え方)。超過する字はSLOT_COUNT同様に切り捨てる
FEATURE_DIM = 2  # 座標(x, y)の次元数
POINTS_PER_SEGMENT = 2
FILENAME_CODEPOINT_PATTERN = re.compile(r"^([0-9a-fA-F]+)")
STROKE_NUMBER_PATTERN = re.compile(r"-s(\d+)$")
# 実データの端点間距離の分布を見ると、この値未満の距離に「実際に接続している点」の集団が偏っており、
# 4付近を谷にして「接続していない点同士のランダムな距離」の分布に切り替わる。頂点として1つにまとめる
# 端点同士の閾値として使う
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


def _filename_codepoint(svg_path: Path) -> int:
    # KanjiVGのファイル名は先頭がコードポイントの16進表記で、字形バリアント違いのファイルには
    # 「0789a-Kaisho.svg」のように末尾へ追加のサフィックスが付く。先頭の16進部分だけを見れば十分なため
    # fullmatchではなくmatchを使う
    return int(FILENAME_CODEPOINT_PATTERN.match(svg_path.stem).group(1), 16)


def _is_kanji_codepoint(codepoint: int) -> bool:
    # KanjiVGには漢字以外の字形(ASCII記号・数字・英字、ひらがな、カタカナ、康熙部首等)も含まれているため、
    # 漢字のUnicodeブロックに含まれるコードポイントだけを対象とし、それ以外は除外する
    return (
        0x3400 <= codepoint <= 0x4DBF  # CJK統合漢字拡張A
        or 0x4E00 <= codepoint <= 0x9FFF  # CJK統合漢字
        or 0xF900 <= codepoint <= 0xFAFF  # CJK互換漢字
        or codepoint >= 0x20000  # CJK拡張B以降(補助漢字面はCJK関連ブロックのみのため上限を設けない)
    )


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


def _split_svg_into_segments(svg_path: Path) -> list[SvgPath] | None:
    root = ET.parse(svg_path).getroot()
    path_elements = sorted(root.findall(f".//{SVG_NAMESPACE}path"), key=_parse_stroke_number)

    # 画数ではなく、折れ・ハネで分割した後のセグメント数がSLOT_COUNTの基準になる
    segments = []
    for path_element in path_elements:
        segments.extend(_split_at_corner(parse_path(path_element.get("d"))))
    if len(segments) > SLOT_COUNT:
        return None
    return segments


def _start_point(segment_index: int) -> int:
    return POINTS_PER_SEGMENT * segment_index


def _end_point(segment_index: int) -> int:
    return _start_point(segment_index) + 1


def _fit_control_point_offset(path: SvgPath, start: complex, end: complex) -> complex:
    # 2次ベジェ B(t) = (1-t)^2*start + 2(1-t)t*p1 + t^2*end はp1について線形なので、
    # パス上のサンプル点群から最小二乗でp1を直接解ける
    ts = np.linspace(0, 1, CONTROL_POINT_SAMPLE_COUNT)
    points = np.array([path.point(t) for t in ts])
    basis = 2 * (1 - ts) * ts
    target = points - (1 - ts) ** 2 * start - ts**2 * end
    control_point = (basis * target).sum() / (basis * basis).sum()
    return complex(control_point - (start + end) / 2)


def _segment_geometry(segments: list[SvgPath]) -> tuple[np.ndarray, np.ndarray]:
    # points: (POINTS_PER_SEGMENT*len(segments), FEATURE_DIM) -- 偶数index=始点・奇数index=終点。
    # offsets: (len(segments), FEATURE_DIM)
    points = np.zeros((POINTS_PER_SEGMENT * len(segments), FEATURE_DIM))
    offsets = np.zeros((len(segments), FEATURE_DIM))
    for segment_index, segment in enumerate(segments):
        start, end = segment.start, segment.end
        points[_start_point(segment_index)] = (start.real, start.imag)
        points[_end_point(segment_index)] = (end.real, end.imag)
        offset = _fit_control_point_offset(segment, start, end)
        offsets[segment_index] = (offset.real, offset.imag)
    return points, offsets


def _compute_adjacency(points: np.ndarray, segment_count: int) -> np.ndarray:
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    adjacency = distance < CONNECTION_THRESHOLD
    for segment_index in range(segment_count):
        # 同じストロークの始点・終点同士を接続対象にすると、そのストローク自体の長さを0に潰す力が働いてしまう
        adjacency[_start_point(segment_index), _end_point(segment_index)] = False
        adjacency[_end_point(segment_index), _start_point(segment_index)] = False
    np.fill_diagonal(adjacency, False)
    return adjacency


def _clique_cover(component: set[int], graph: nx.Graph) -> list[set[int]]:
    remaining = set(component)
    groups: list[set[int]] = []
    while remaining:
        subgraph = graph.subgraph(remaining)
        # 同サイズの最大クリークが複数ある場合、最小のインデックス(=ストローク処理順で最初に
        # 登場する点)を含む方を優先する。この優先順位が最終的な頂点座標(重心)に直接影響するため
        # 任意ではなく明示的なルールとして固定する
        clique = max(nx.find_cliques(subgraph), key=lambda c: (len(c), -min(c)))
        groups.append(set(clique))
        remaining -= set(clique)
    return groups


def _cluster_endpoints_into_vertices(point_indices: list[int], adjacency: np.ndarray) -> list[set[int]]:
    graph = nx.Graph()
    graph.add_nodes_from(point_indices)
    graph.add_edges_from((i, j) for i in point_indices for j in point_indices if i < j and adjacency[i, j])

    groups: list[set[int]] = []
    for component in nx.connected_components(graph):
        groups.extend(_clique_cover(component, graph))
    return groups


def _assign_vertex_slots(point_groups: list[set[int]], segment_count: int) -> dict[int, int]:
    point_to_group = {point: group_id for group_id, group in enumerate(point_groups) for point in group}

    # 頂点のスロット番号は、ストローク処理順で最初に登場したクラスタから順に割り当てる。
    # クラスタ探索自体が返す順序(連結成分やクリークの列挙順)に依存させないための決定的な割り当て
    group_to_slot: dict[int, int] = {}
    for segment_index in range(segment_count):
        for point in (_start_point(segment_index), _end_point(segment_index)):
            group_to_slot.setdefault(point_to_group[point], len(group_to_slot))

    return {point: group_to_slot[group_id] for point, group_id in point_to_group.items()}


def _compute_vertex_coordinates(points: np.ndarray, point_to_slot: dict[int, int]) -> np.ndarray:
    vertex_count = max(point_to_slot.values()) + 1
    sums = np.zeros((vertex_count, FEATURE_DIM))
    counts = np.zeros(vertex_count)
    for point, slot in point_to_slot.items():
        sums[slot] += points[point]
        counts[slot] += 1
    return sums / counts[:, None]


def _max_vertex_error(points: np.ndarray, point_to_slot: dict[int, int], vertex_coords: np.ndarray) -> float:
    return float(max(np.linalg.norm(points[point] - vertex_coords[slot]) for point, slot in point_to_slot.items()))


def _max_clique_diameter(point_groups: list[set[int]], points: np.ndarray) -> float:
    max_distance = 0.0
    for group in point_groups:
        members = points[list(group)]
        if len(members) < 2:
            continue
        pairwise = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=-1)
        max_distance = max(max_distance, pairwise.max())
    return float(max_distance)


def _pack_vertices(vertex_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # _assign_vertex_slotsが頂点スロット番号を0始まりの連番で割り当てるため、先頭len(vertex_coords)件を
    # 埋めるだけで実在する頂点をすべてカバーできる
    vertices = np.zeros((VERTEX_COUNT, FEATURE_DIM))
    vertex_existence = np.zeros(VERTEX_COUNT)
    vertices[: len(vertex_coords)] = vertex_coords
    vertex_existence[: len(vertex_coords)] = 1.0
    return vertices, vertex_existence


def _pack_strokes(
    segment_count: int, point_to_slot: dict[int, int], offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stroke_vertex_indices = np.zeros((SLOT_COUNT, POINTS_PER_SEGMENT), dtype=np.int64)
    stroke_offsets = np.zeros((SLOT_COUNT, FEATURE_DIM))
    stroke_existence = np.zeros(SLOT_COUNT)
    for segment_index in range(segment_count):
        stroke_vertex_indices[segment_index] = (
            point_to_slot[_start_point(segment_index)],
            point_to_slot[_end_point(segment_index)],
        )
        stroke_offsets[segment_index] = offsets[segment_index]
        stroke_existence[segment_index] = 1.0
    return stroke_vertex_indices, stroke_offsets, stroke_existence


def _pack_kanji_tensors(
    segment_count: int, point_to_slot: dict[int, int], vertex_coords: np.ndarray, offsets: np.ndarray
) -> KanjiTensors:
    vertices, vertex_existence = _pack_vertices(vertex_coords)
    stroke_vertex_indices, stroke_offsets, stroke_existence = _pack_strokes(segment_count, point_to_slot, offsets)
    return KanjiTensors(vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence)


def _cluster_into_kanji_tensors(segments: list[SvgPath]) -> tuple[KanjiTensors, _ClusteringDiagnostics] | None:
    segment_count = len(segments)
    points, offsets = _segment_geometry(segments)
    adjacency = _compute_adjacency(points, segment_count)
    point_groups = _cluster_endpoints_into_vertices(list(range(len(points))), adjacency)
    if len(point_groups) > VERTEX_COUNT:
        return None

    point_to_slot = _assign_vertex_slots(point_groups, segment_count)
    vertex_coords = _compute_vertex_coordinates(points, point_to_slot)
    diagnostics = _ClusteringDiagnostics(
        _max_vertex_error(points, point_to_slot, vertex_coords),
        _max_clique_diameter(point_groups, points),
    )

    tensors = _pack_kanji_tensors(segment_count, point_to_slot, vertex_coords, offsets)
    return tensors, diagnostics


def main() -> None:
    all_tensors: list[KanjiTensors] = []
    all_diagnostics: list[_ClusteringDiagnostics] = []
    non_kanji_count = 0
    excluded_segment_count = 0
    excluded_vertex_count = 0

    for svg_path in DATA_DIR.glob("*.svg"):
        if not _is_kanji_codepoint(_filename_codepoint(svg_path)):
            non_kanji_count += 1
            continue

        segments = _split_svg_into_segments(svg_path)
        if segments is None:
            excluded_segment_count += 1
            continue

        result = _cluster_into_kanji_tensors(segments)
        if result is None:
            excluded_vertex_count += 1
            continue
        tensors, diagnostics = result
        all_tensors.append(tensors)
        all_diagnostics.append(diagnostics)

    print(f"Included kanji: {len(all_tensors)}")
    print(f"Excluded (non-kanji codepoint): {non_kanji_count}")
    print(f"Excluded (segment count after corner splitting > {SLOT_COUNT}): {excluded_segment_count}")
    print(f"Excluded (vertex count after clique clustering > {VERTEX_COUNT}): {excluded_vertex_count}")

    # KanjiTensorsは5フィールドのNamedTupleなので、zip(*all_tensors)でフィールドごとのタプルへ転置してから
    # それぞれをnp.array化する
    vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence = (
        np.array(field) for field in zip(*all_tensors)
    )
    max_vertex_errors = [d.max_vertex_error for d in all_diagnostics]
    max_clique_diameters = [d.max_clique_diameter for d in all_diagnostics]

    avg_vertices = vertex_existence.sum(axis=1).mean()
    print(f"Average vertices per kanji: {avg_vertices:.2f}")
    print(
        f"Vertex reconstruction error (endpoint vs. assigned vertex centroid): "
        f"mean={np.mean(max_vertex_errors):.4f} max={np.max(max_vertex_errors):.4f}"
    )
    max_diameter = np.max(max_clique_diameters)
    print(f"Max clique diameter across all kanji: {max_diameter:.17g} (below {CONNECTION_THRESHOLD}: {max_diameter < CONNECTION_THRESHOLD})")

    np.savez_compressed(
        OUTPUT_PATH,
        vertices=vertices,
        vertex_existence=vertex_existence,
        stroke_vertex_indices=stroke_vertex_indices,
        stroke_offsets=stroke_offsets,
        stroke_existence=stroke_existence,
    )
    print(f"Saved vertex+stroke features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
