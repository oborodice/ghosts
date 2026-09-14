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


def _load_vertex_features() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH)
    return data["vertices"], data["vertex_existence"]


def _split_train_val_indices(kanji_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(kanji_count)
    val_size = int(kanji_count * VAL_SPLIT)
    return shuffled_indices[val_size:], shuffled_indices[:val_size]


def _compute_standardization_stats(
    vertices: np.ndarray, existence: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 標準化の統計量はtrainスプリットの実在頂点(existence=1)のみから算出する
    train_valid_vertices = vertices[train_indices][existence[train_indices].astype(bool)]
    mean = train_valid_vertices.mean(axis=0)
    std = train_valid_vertices.std(axis=0)
    return mean, np.where(std < STD_EPSILON, STD_EPSILON, std)


def _standardize(vertices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (vertices - mean) / std


def _build_dataset(indices: np.ndarray, vertices: np.ndarray, existence: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(vertices[indices], dtype=torch.float32),
        torch.tensor(existence[indices], dtype=torch.float32),
    )


class Datasets(NamedTuple):
    train: TensorDataset
    val: TensorDataset
    shape: ModelShape
    mean: np.ndarray
    std: np.ndarray


def prepare_datasets() -> Datasets:
    vertices, existence = _load_vertex_features()
    kanji_count, vertex_count, feature_dim = vertices.shape
    shape = ModelShape(vertex_count, feature_dim)

    train_indices, val_indices = _split_train_val_indices(kanji_count)
    mean, std = _compute_standardization_stats(vertices, existence, train_indices)
    vertices_standardized = _standardize(vertices, mean, std)

    train_dataset = _build_dataset(train_indices, vertices_standardized, existence)
    val_dataset = _build_dataset(val_indices, vertices_standardized, existence)
    return Datasets(train_dataset, val_dataset, shape, mean, std)
