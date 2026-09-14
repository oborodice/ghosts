#!/usr/bin/env python3
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from vae_model_v2 import ModelShape

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stroke_features_v2.npz"

VAL_SPLIT = 0.1
SEED = 0
STD_EPSILON = 1e-8  # 分散が0の特徴量があった場合のゼロ割回避


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
    return Datasets(train_dataset, val_dataset, shape, vertex_mean, vertex_std, stroke_offset_mean, stroke_offset_std)
