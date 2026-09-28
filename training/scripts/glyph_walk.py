# 字の変わり方を決める、潜在空間の軌跡(simplex noiseの値)。NumPyとopensimplexだけで動き(表示にtorchを要らなくするため)、
# 表示(ONNX Runtime)・評価・ONNXへの書き出しで共通に使う。潜在の次元ごとにsimplex noiseの場の別の行をたどる
import numpy as np
from opensimplex import OpenSimplex

DISPLAY_FPS = 30.0
DIMENSION_SPACING = 10.0  # 潜在の次元ごとに、ノイズ場のy方向にこれだけ離した行を使い、次元どうしを無関係にする
SCATTER_RANGE = 100_000.0  # 互いに離れた位置を選ぶ、ノイズ場のx方向の範囲(ノイズの模様の大きさ(約1)より十分に広い)
# ノイズ場の中を進む既定の速さ(1秒あたり)。この速さで、字の移り変わりがなめらか(別の字への急な切り替わりがない)ことを確かめた
DEFAULT_NOISE_SPEED = 0.0633


def _values_at(noise: OpenSimplex, positions: np.ndarray, latent_dim: int) -> np.ndarray:
    # ノイズ場のx方向の位置ごとの値。返り値は(位置の数, 潜在の次元の数)の、noise2の生の値
    return noise.noise2array(positions, np.arange(latent_dim) * DIMENSION_SPACING).T.astype(np.float32)


def simplex_walk(latent_dim: int, frames: int, fps: float, noise_speed: float, seed: int) -> np.ndarray:
    # 0フレーム目から frames 個の軌跡。返り値は(フレームの数, 潜在の次元の数)
    return _values_at(OpenSimplex(seed), np.arange(frames) / fps * noise_speed, latent_dim)


def simplex_frame(noise: OpenSimplex, frame: int, latent_dim: int, fps: float, noise_speed: float) -> np.ndarray:
    # simplex_walk の frame 番目のフレームだけの値(終わりのない表示で、1フレームずつ作るため)。返り値は(1, 潜在の次元の数)
    return _values_at(noise, np.array([frame / fps * noise_speed]), latent_dim)


def simplex_scattered(latent_dim: int, count: int, seed: int) -> np.ndarray:
    # 同じノイズ場の、互いに離れたcount個の位置の値。返り値は(count, 潜在の次元の数)。
    # 軌跡の上のフレームは互いに似ているので、表示で出うる字の分布を見るときはこちらを使う
    positions = np.random.default_rng(seed).uniform(0, SCATTER_RANGE, count)
    return _values_at(OpenSimplex(seed), positions, latent_dim)
