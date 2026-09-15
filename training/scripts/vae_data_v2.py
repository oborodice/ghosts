#!/usr/bin/env python3
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset

from vae_data import AngleGMMParams
from vae_model_v2 import ModelShape

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stroke_features_v2.npz"

VAL_SPLIT = 0.1
SEED = 0
STD_EPSILON = 1e-8  # 分散が0の特徴量があった場合のゼロ割回避
ANGLE_GMM_COMPONENTS = 12  # 実データの角度分布をGMMで近似する際のコンポーネント数。BICは単調に改善し続け
# 明確な肘がないため、ヒストグラムとの目視比較で主要な山を捉えつつ過度に細かくならない値として選んだ
ANGLE_GMM_REG_COVAR = 0.01  # GMMの共分散に足す正則化項。実データが極端に密集しているため既定値(1e-6)では
# 密度関数が急峻すぎ、わずかな再構成誤差で損失が跳ね上がってしまうのを緩和する


def _load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH)
    return (
        data["vertices"],
        data["vertex_existence"],
        data["stroke_vertex_indices"],
        data["stroke_offsets"],
        data["stroke_existence"],
    )


def _split_train_val_indices(kanji_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(kanji_count)
    val_size = int(kanji_count * VAL_SPLIT)
    return shuffled_indices[val_size:], shuffled_indices[:val_size]


def _compute_standardization_stats(
    features: np.ndarray, existence: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 標準化の統計量はtrainスプリットの実在スロット(existence=1)のみから算出する
    train_valid_features = features[train_indices][existence[train_indices].astype(bool)]
    mean = train_valid_features.mean(axis=0)
    std = train_valid_features.std(axis=0)
    return mean, np.where(std < STD_EPSILON, STD_EPSILON, std)


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def _fit_angle_gmm(
    vertices: np.ndarray,
    stroke_vertex_indices: np.ndarray,
    stroke_existence: np.ndarray,
    train_indices: np.ndarray,
) -> AngleGMMParams:
    # 頂点ペア(始点・終点)から実スケールでatan2により角度を求め直してフィットする。角度は周期的
    # (0度と360度が同じ)なので、単純な数値としてではなく(cosθ, sinθ)の2次元ベクトルに変換してから
    # フィットすることで、0度と360度が近いことを正しく扱う
    train_vertices = vertices[train_indices]
    train_stroke_vertex_indices = stroke_vertex_indices[train_indices]
    start_index = train_stroke_vertex_indices[..., 0:1]
    end_index = train_stroke_vertex_indices[..., 1:2]
    start = np.take_along_axis(train_vertices, start_index, axis=1)
    end = np.take_along_axis(train_vertices, end_index, axis=1)
    delta = (end - start)[stroke_existence[train_indices].astype(bool)]
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    gmm = GaussianMixture(
        n_components=ANGLE_GMM_COMPONENTS, random_state=SEED, n_init=3, reg_covar=ANGLE_GMM_REG_COVAR
    )
    gmm.fit(points)
    return AngleGMMParams(gmm.means_, gmm.covariances_, gmm.weights_)


def _build_dataset(
    indices: np.ndarray,
    vertices: np.ndarray,
    vertex_existence: np.ndarray,
    stroke_vertex_indices: np.ndarray,
    stroke_offsets: np.ndarray,
    stroke_existence: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.tensor(vertices[indices], dtype=torch.float32),
        torch.tensor(vertex_existence[indices], dtype=torch.float32),
        torch.tensor(stroke_vertex_indices[indices], dtype=torch.int64),
        torch.tensor(stroke_offsets[indices], dtype=torch.float32),
        torch.tensor(stroke_existence[indices], dtype=torch.float32),
    )


class Datasets(NamedTuple):
    train: TensorDataset
    val: TensorDataset
    shape: ModelShape
    vertex_mean: np.ndarray
    vertex_std: np.ndarray
    stroke_offset_mean: np.ndarray
    stroke_offset_std: np.ndarray
    angle_gmm_params: AngleGMMParams


def prepare_datasets() -> Datasets:
    vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence = _load_features()
    kanji_count, vertex_count, vertex_feature_dim = vertices.shape
    stroke_count, stroke_feature_dim = stroke_offsets.shape[1:]
    shape = ModelShape(vertex_count, vertex_feature_dim, stroke_count, stroke_feature_dim)

    train_indices, val_indices = _split_train_val_indices(kanji_count)
    vertex_mean, vertex_std = _compute_standardization_stats(vertices, vertex_existence, train_indices)
    stroke_offset_mean, stroke_offset_std = _compute_standardization_stats(
        stroke_offsets, stroke_existence, train_indices
    )
    vertices_standardized = _standardize(vertices, vertex_mean, vertex_std)
    stroke_offsets_standardized = _standardize(stroke_offsets, stroke_offset_mean, stroke_offset_std)
    angle_gmm_params = _fit_angle_gmm(vertices, stroke_vertex_indices, stroke_existence, train_indices)

    train_dataset = _build_dataset(
        train_indices,
        vertices_standardized,
        vertex_existence,
        stroke_vertex_indices,
        stroke_offsets_standardized,
        stroke_existence,
    )
    val_dataset = _build_dataset(
        val_indices,
        vertices_standardized,
        vertex_existence,
        stroke_vertex_indices,
        stroke_offsets_standardized,
        stroke_existence,
    )
    return Datasets(
        train_dataset, val_dataset, shape, vertex_mean, vertex_std, stroke_offset_mean, stroke_offset_std,
        angle_gmm_params,
    )
