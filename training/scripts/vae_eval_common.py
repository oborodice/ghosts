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

# 生成時にzを実データへ引き寄せるカーネル幅。元は実データ同士の最近傍距離の中央値を目安に0.6としていたが、
# それだと生成結果が特定の実在字とほぼ一致しやすかった。decoderをcompute_synthetic_grammar_loss
# (vae_synthetic_losses.py)で広いbandwidthでも崩れないよう学習し直した上でsweepし、ノベルティ(最近傍実データとの距離)・交差数・
# 斜め関与交差のバランスが最も良かった1.5を採用した。export_onnx.pyのエクスポート済みグラフにもこの値が
# そのまま焼き込まれる
KERNEL_BANDWIDTH = 1.5


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


SEGMENTS_PER_CURVE = 12  # ベジェ曲線をポリライン近似する際の線分数
INTERIOR_RANGE = (0.08, 0.92)  # ストローク端点付近(接続点)を交差から除外する範囲
AXIS_TOLERANCE_DEG = 15.0  # 0/90/180/270度からこの範囲内なら「軸方向(水平・垂直)」とみなす
JUNCTION_CONNECTION_THRESHOLD = 4.0  # 3本以上合流の検出用。extract_stroke_features.CONNECTION_THRESHOLDと同じ
JUNCTION_CLUSTER_RADIUS = 6.0  # 接続点・交差点同士を「同じ場所」とみなす半径


def bezier_polyline(start: complex, control: complex, end: complex, n: int) -> np.ndarray:
    ts = np.linspace(0, 1, n + 1)
    pts = (1 - ts) ** 2 * start + 2 * (1 - ts) * ts * control + ts**2 * end
    return np.stack([pts.real, pts.imag], axis=1)


def _segment_intersection(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> tuple[float, float] | None:
    # p1->p2とp3->p4の交点をパラメータt(p1->p2上)・u(p3->p4上)で返す。平行ならNone
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t, u
    return None


def _find_curve_intersections(
    curves: list[tuple[complex, complex, complex]]
) -> list[tuple[int, int, np.ndarray]]:
    # ストローク中間での交差(端点付近=接続点は除外)を(i, j, 交点座標)の組で列挙する。
    # find_crossing_pairs/find_interaction_pointsの共通処理
    polylines = [bezier_polyline(s, c, e, SEGMENTS_PER_CURVE) for s, c, e in curves]
    n = len(polylines)
    intersections: list[tuple[int, int, np.ndarray]] = []
    for i in range(n):
        poly_i = polylines[i]
        for j in range(i + 1, n):
            poly_j = polylines[j]
            found = False
            for a in range(SEGMENTS_PER_CURVE):
                if found:
                    break
                for b in range(SEGMENTS_PER_CURVE):
                    res = _segment_intersection(poly_i[a], poly_i[a + 1], poly_j[b], poly_j[b + 1])
                    if res is None:
                        continue
                    t_local, u_local = res
                    pos_i = (a + t_local) / SEGMENTS_PER_CURVE
                    pos_j = (b + u_local) / SEGMENTS_PER_CURVE
                    if INTERIOR_RANGE[0] < pos_i < INTERIOR_RANGE[1] and INTERIOR_RANGE[0] < pos_j < INTERIOR_RANGE[1]:
                        point = poly_i[a] + t_local * (poly_i[a + 1] - poly_i[a])
                        intersections.append((i, j, point))
                        found = True
                        break
    return intersections


def find_crossing_pairs(curves: list[tuple[complex, complex, complex]]) -> list[tuple[int, int]]:
    return [(i, j) for i, j, _ in _find_curve_intersections(curves)]


def count_crossings(curves: list[tuple[complex, complex, complex]]) -> int:
    return len(find_crossing_pairs(curves))


def is_axis_aligned(angle_rad: float) -> bool:
    deg = np.degrees(angle_rad) % 90
    return deg < AXIS_TOLERANCE_DEG or deg > (90 - AXIS_TOLERANCE_DEG)


def classify_crossings(curves: list[tuple[complex, complex, complex]], angles: np.ndarray) -> dict[str, int]:
    # 交差ペアを「両方軸方向」か「斜めが関与」かに分類する。anglesはcurvesと同じ順番・同じ本数であること
    pairs = find_crossing_pairs(curves)
    both_axis = sum(1 for i, j in pairs if is_axis_aligned(angles[i]) and is_axis_aligned(angles[j]))
    return {"total": len(pairs), "both_axis": both_axis, "diagonal_involved": len(pairs) - both_axis}


def find_interaction_points(curves: list[tuple[complex, complex, complex]]) -> list[tuple[np.ndarray, frozenset[int]]]:
    # 3本以上の合流検出用。端点同士の近接(接続)とストローク中間の交差の両方を検出する
    n = len(curves)
    points: list[tuple[np.ndarray, frozenset[int]]] = []

    endpoints = [(np.array([s.real, s.imag]), np.array([e.real, e.imag])) for s, c, e in curves]
    for i in range(n):
        for j in range(i + 1, n):
            for pi in endpoints[i]:
                for pj in endpoints[j]:
                    if np.linalg.norm(pi - pj) < JUNCTION_CONNECTION_THRESHOLD:
                        points.append(((pi + pj) / 2, frozenset((i, j))))

    for i, j, point in _find_curve_intersections(curves):
        points.append((point, frozenset((i, j))))
    return points


def count_triple_junctions(curves: list[tuple[complex, complex, complex]]) -> int:
    interactions = find_interaction_points(curves)
    n = len(interactions)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(interactions[i][0] - interactions[j][0]) < JUNCTION_CLUSTER_RADIUS:
                union(i, j)

    clusters: dict[int, set[int]] = {}
    for idx, (_, strokes) in enumerate(interactions):
        clusters.setdefault(find(idx), set()).update(strokes)

    return sum(1 for strokes in clusters.values() if len(strokes) >= 3)


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
