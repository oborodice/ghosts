# フォントで描いた漢字の画像を、どの漢字かに分類する小さなモデル(ResNetの形)。
# 分類そのものより、中間の特徴(features)を、生成した字の評価(実在字との近さ、種類の数、分布の重なり)の物差しに使う
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

DROPOUT = 0.2
# 評価に使うしきい値を、実在字の特徴の距離の分布から決めるための値。同じ字を別の書風で描いた組の距離の SAME_GLYPH_QUANTILE 点と、
# 別の字どうしの組の距離の DIFFERENT_GLYPH_QUANTILE 点を測り、しきい値をその間(または倍率)の決まった位置に置く。
# 位置は、同じ種類とみなす距離8・ジャンプとみなす距離13.39を目視と合わせて決めた文字認識のモデル(同じ字の90%点5.77、
# 別の字の1%点14.21)から逆算した(特徴の空間の大きさが変わっても、分布に対して同じ位置になるように)
SAME_GLYPH_QUANTILE = 0.9
DIFFERENT_GLYPH_QUANTILE = 0.01
SAME_GLYPH_POSITION = 0.264  # 同じ種類とみなす距離 = 同じ字の分位点 + この割合 x (別の字の分位点 - 同じ字の分位点)
JUMP_RATIO = 0.942  # ジャンプとみなす距離 = 別の字の分位点 x この倍率


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        needs_projection = stride != 1 or in_channels != out_channels
        self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)) if needs_projection else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.skip(x) + self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))))


class GlyphClassifier(nn.Module):
    FEATURE_DIM = 256

    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.body = nn.Sequential(
            _ResidualBlock(32, 64, 2), _ResidualBlock(64, 64, 1), _ResidualBlock(64, 128, 2), _ResidualBlock(128, 128, 1), _ResidualBlock(128, self.FEATURE_DIM, 2), _ResidualBlock(self.FEATURE_DIM, self.FEATURE_DIM, 1)
        )
        self.head = nn.Linear(self.FEATURE_DIM, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

    def features(self, ink: torch.Tensor) -> torch.Tensor:
        # ink: (字の数, 1, 高さ, 幅)、0=紙〜1=インク
        return self.body(self.stem(ink)).mean((2, 3))

    def forward(self, ink: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(ink)))


class Calibration(NamedTuple):
    # 学習データの実在字で測った、特徴の距離の分布と、そこから決めたしきい値
    same_glyph_p90: float  # 同じ字を別の書風で描いた組の距離の90%点
    different_glyph_p1: float  # 別の字どうしの組の距離の1%点
    same_glyph_distance: float  # 生成物どうしを同じ種類の字とみなす距離
    jump_distance: float  # モーフィングの1フレームの変化が、別の字への急な切り替わりとみなせる距離


def calibrate(same_distances: torch.Tensor, different_distances: torch.Tensor) -> Calibration:
    same_glyph_p90 = torch.quantile(same_distances, SAME_GLYPH_QUANTILE).item()
    different_glyph_p1 = torch.quantile(different_distances, DIFFERENT_GLYPH_QUANTILE).item()
    return Calibration(
        same_glyph_p90=same_glyph_p90,
        different_glyph_p1=different_glyph_p1,
        same_glyph_distance=same_glyph_p90 + SAME_GLYPH_POSITION * (different_glyph_p1 - same_glyph_p90),
        jump_distance=JUMP_RATIO * different_glyph_p1,
    )


def load_classifier(path: Path, device: torch.device | str) -> tuple[GlyphClassifier, Calibration]:
    state = torch.load(path, map_location=device)
    model = GlyphClassifier(state["num_classes"]).to(device).eval()
    model.load_state_dict(state["model"])
    return model, Calibration(**state["calibration"])
