#!/usr/bin/env python3
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset

from vae_model import ModelShape

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stroke_features.npz"

VAL_SPLIT = 0.1
SEED = 0
STD_EPSILON = 1e-8  # 分散が0の特徴量があった場合のゼロ割回避
ANGLE_GMM_COMPONENTS = 12  # 実データの角度分布をGMMで近似する際のコンポーネント数。BICは単調に改善し続け明確な肘がないため、ヒストグラムとの目視比較で主要な山を捉えつつ過度に細かくならない値として選んだ
ANGLE_GMM_REG_COVAR = 0.01  # GMMの共分散に足す正則化項。実データが極端に密集しているため既定値(1e-6)では密度関数が急峻すぎ(0度から5度ずれただけで密度が1/160000に)、わずかな再構成誤差で損失が跳ね上がっていた。この値は急峻さを緩和しつつ、非典型的な角度(180度など)との識別力も保てる範囲として選んだ


def load_stroke_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH)
    return data["strokes"], data["existence"], data["connections"]


def split_train_val_indices(kanji_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(kanji_count)
    val_size = int(kanji_count * VAL_SPLIT)
    return shuffled_indices[val_size:], shuffled_indices[:val_size]


def _compute_standardization_stats(
    strokes: np.ndarray, existence: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 標準化の統計量はtrainスプリットの実在ストローク(existence=1)のみから算出する
    train_valid_features = strokes[train_indices][existence[train_indices].astype(bool)]
    mean = train_valid_features.mean(axis=0)
    std = train_valid_features.std(axis=0)
    return mean, np.where(std < STD_EPSILON, STD_EPSILON, std)


def standardize(strokes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (strokes - mean) / std


class AngleGMMParams(NamedTuple):
    # _fit_angle_gmmの生の(numpyの)フィット結果。torch化・精度行列/対数行列式の計算まで済ませた
    # 学習用の表現はvae_losses.AngleGMM(vae_losses.build_angle_gmm参照)
    means: np.ndarray  # (K, 2)
    covariances: np.ndarray  # (K, 2, 2)
    weights: np.ndarray  # (K,)


def _fit_angle_gmm(strokes: np.ndarray, existence: np.ndarray, train_indices: np.ndarray) -> AngleGMMParams:
    # 角度は周期的(0度と360度が同じ)なので、単純な数値としてではなく(cosθ, sinθ)の2次元ベクトルに
    # 変換してからフィットすることで、0度と360度が近いことを正しく扱う
    train_valid = strokes[train_indices][existence[train_indices].astype(bool)]
    angles = train_valid[:, 2]
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    gmm = GaussianMixture(
        n_components=ANGLE_GMM_COMPONENTS, random_state=SEED, n_init=3, reg_covar=ANGLE_GMM_REG_COVAR
    )
    gmm.fit(points)
    return AngleGMMParams(gmm.means_, gmm.covariances_, gmm.weights_)


def _build_dataset(
    indices: np.ndarray, strokes: np.ndarray, existence: np.ndarray, connections: np.ndarray
) -> TensorDataset:
    return TensorDataset(
        torch.tensor(strokes[indices], dtype=torch.float32),
        torch.tensor(existence[indices], dtype=torch.float32),
        torch.tensor(connections[indices], dtype=torch.float32),
    )


class Datasets(NamedTuple):
    train: TensorDataset
    val: TensorDataset
    shape: ModelShape
    mean: np.ndarray
    std: np.ndarray
    angle_gmm_params: AngleGMMParams


def prepare_datasets() -> Datasets:
    strokes, existence, connections = load_stroke_features()
    kanji_count, slot_count, feature_dim = strokes.shape
    shape = ModelShape(slot_count, feature_dim)

    train_indices, val_indices = split_train_val_indices(kanji_count)
    mean, std = _compute_standardization_stats(strokes, existence, train_indices)
    strokes_standardized = standardize(strokes, mean, std)
    angle_gmm_params = _fit_angle_gmm(strokes, existence, train_indices)

    train_dataset = _build_dataset(train_indices, strokes_standardized, existence, connections)
    val_dataset = _build_dataset(val_indices, strokes_standardized, existence, connections)
    return Datasets(train_dataset, val_dataset, shape, mean, std, angle_gmm_params)
