# 学習したGANで字を作る推論の部分(PyTorch。ONNXへの書き出しと評価で使う)。
# simplex noiseの値(glyph/walk.py)を正規分布に写してから、写像ネットワーク・生成器に通す
import functools
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from glyph import gan
from glyph.walk import simplex_scattered

NOISE_IMAGE_SEED = 0
# simplex noiseの値を正規分布に写す対応表: simplex noiseの値の範囲を等間隔に区切った点ごとに、その値以下になる割合(大量の値から数えた
# 累積分布)を正規分布の分位点に写した値を持ち、点の間は直線でつなぐ。simplex noiseの値は1次元ごとに正規分布より裾が軽く偏るので、
# 学習時の入力(正規分布)に合わせる
GAUSSIANIZE_POINTS = 2001
GAUSSIANIZE_RANGE = 1.0  # opensimplexのnoise2の値はこの範囲に収まる
GAUSSIANIZE_SAMPLES = 2_000_000
GAUSSIANIZE_SEED = 12345


class GlyphGenerator(nn.Module):
    # simplex noiseの値 (バッチ, 潜在の次元の数) → インクの画像 (バッチ, 1, 解像度, 解像度)、0=紙〜1=インク。
    # ノイズの画像(ノイズの注入)は1枚に固定する(フレームごとに変えると、字がちらつくため)
    def __init__(self, mapping: nn.Module, generator: gan.Generator, latent_dim: int, table: torch.Tensor, noise_image: torch.Tensor):
        super().__init__()
        self.latent_dim = latent_dim
        self.mapping = mapping
        self.generator = generator
        self.register_buffer("table", table)
        self.register_buffer("noise_image", noise_image)

    def _gaussianize(self, values: torch.Tensor) -> torch.Tensor:
        # 対応表の引き方を、添字の計算と読み出しだけで書く(ONNXの標準の演算で表せるように)
        position = (values.clamp(-GAUSSIANIZE_RANGE, GAUSSIANIZE_RANGE) + GAUSSIANIZE_RANGE) / (2 * GAUSSIANIZE_RANGE) * (GAUSSIANIZE_POINTS - 1)
        lower = position.floor().clamp(max=GAUSSIANIZE_POINTS - 2)
        fraction = position - lower
        lower_index = lower.long()
        return self.table[lower_index] * (1 - fraction) + self.table[lower_index + 1] * fraction

    def forward(self, simplex_values: torch.Tensor) -> torch.Tensor:
        style = self.mapping(self._gaussianize(simplex_values))
        styles = style[:, None, :].expand(-1, self.generator.num_layers, -1)
        return self.generator(styles, self.noise_image).clamp(0, 1)

    def generate_in_chunks(self, simplex_values: torch.Tensor) -> torch.Tensor:
        # ONNXに書き出すforwardは1字ずつ呼ぶ。評価などで多くの字を作るときはこちら
        chunk = gan.GENERATION_CHUNK
        return torch.cat([self(simplex_values[start:start + chunk]) for start in range(0, len(simplex_values), chunk)])


def _load_mapping_and_generator(checkpoint_path: Path, device: torch.device | str) -> tuple[gan.MappingNetwork, gan.Generator, int]:
    # train_gan.py のチェックポイントから、推論に使う移動平均の版の写像ネットワークと生成器を作る。3つめの返り値は潜在の次元の数
    state = torch.load(checkpoint_path, map_location=device)
    config = state["config"]
    generator_state = state["generator_ema"]
    num_layers = len([key for key in generator_state if key.endswith(".conv1.weight")])
    image_size = gan.INITIAL_SIZE * 2 ** (num_layers - 1)
    mapping = gan.MappingNetwork(config["latent_dim"], config["mapping_depth"], config["mapping_learning_rate_multiplier"])
    generator = gan.Generator(image_size, config["latent_dim"], config["capacity"], image_channels=1)
    mapping.load_state_dict(state["mapping_ema"])
    generator.load_state_dict(generator_state)
    return mapping, generator, config["latent_dim"]


@functools.cache  # 作るのに約10秒かかり、中身は毎回同じなので、1回だけ作る(学習中に保存ごとに読むときなど)
def _gaussianize_table() -> torch.Tensor:
    # 表示でノイズを作るのと同じ関数(opensimplexのnoise2)を、ばらばらの位置で大量に評価する
    side = int(math.sqrt(GAUSSIANIZE_SAMPLES))
    samples = np.sort(simplex_scattered(side, side, GAUSSIANIZE_SEED).ravel())
    grid = np.linspace(-GAUSSIANIZE_RANGE, GAUSSIANIZE_RANGE, GAUSSIANIZE_POINTS)
    # 累積の割合を(0, 1)の内側に保ち、両端で無限大にならないようにする
    cumulative = (np.searchsorted(samples, grid, side="right") + 0.5) / (len(samples) + 1)
    return torch.special.ndtri(torch.from_numpy(cumulative)).float()


def load_glyph_generator(checkpoint_path: Path, device: torch.device | str = "cpu") -> GlyphGenerator:
    mapping, generator, latent_dim = _load_mapping_and_generator(checkpoint_path, device)
    noise_image = torch.rand(1, generator.image_size, generator.image_size, 1, generator=torch.Generator().manual_seed(NOISE_IMAGE_SEED))
    return GlyphGenerator(mapping, generator, latent_dim, _gaussianize_table(), noise_image).to(device).eval()
