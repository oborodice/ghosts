# フォントで描いた漢字の画像を学習するGAN(StyleGAN2の設計)のモデル: 写像ネットワーク、生成器、判別器。
#
# 生成器・判別器・写像ネットワークの構造と計算は、lucidrains/stylegan2-pytorch
# (https://github.com/lucidrains/stylegan2-pytorch、MIT License、Copyright (c) 2020 Phil Wang)から、使う部分だけを移したもの。
# ライセンスの全文は training/licenses/stylegan2-pytorch.txt
# 層の並びと計算は元のコードと同じ(属性の名前は変えたので、元のコードの重みは名前を付け替えれば読める)。元のコードからの変更:
# - 判別器に、ミニバッチの標準偏差の層(StyleGAN2と同じ形)を入れた(同じ画像ばかり出す生成器を見抜けるように)
# - 写像ネットワークの重みの初期化を、学習率の倍率によらず出力の大きさが同じになるようにした
# - 画像のチャンネル数を引数にした(学習データは1チャンネル)
# - ぼかしを、kornia を使わずに同じ計算で書いた(3x3の[1, 2, 1]のフィルタ、端は反射で埋める)
# - 使っていない機能(注意機構、量子化、透過チャンネル、定数を使わない初期ブロック)を除いた
import math

import torch
import torch.nn.functional as F
from torch import nn

LEAKY_RELU_SLOPE = 0.2
# 写像ネットワークの全結合の重みの、実際に使われる値の標準偏差(学習率の倍率が0.1のとき、標準正規分布の初期値に
# 倍率を掛けた大きさ)。倍率を変えても、この大きさで始まるように初期値を割り戻す
MAPPING_WEIGHT_SCALE = 0.1
MINIBATCH_STD_GROUP = 4
EPSILON = 1e-8
INITIAL_SIZE = 4  # 生成器は、この大きさ(px)の学習する定数から始めて、2倍ずつ拡大する
FINAL_SIZE = 2  # 判別器は、画像をこの大きさ(px)まで2分の1ずつ縮めてから、1つの値にする
MAX_FEATURES = 512  # 各層の特徴の数の上限(判別器の奥の層は、容量からの計算ではこれを超える)
# スタイルで重みを変える畳み込みは、バッチの数 x 特徴の数のチャンネルでグループ化した畳み込みになる。
# Apple Silicon(MPS)は、このチャンネルが 65,535 を超えると、エラーを出さずに誤った値を返す(65,280 は正しく、65,536 から誤る)
MPS_GROUPED_CONV_MAX_CHANNELS = 65_535
GENERATION_CHUNK = MPS_GROUPED_CONV_MAX_CHANNELS // MAX_FEATURES  # 多くの字をまとめて作るときに、生成器に一度に渡す数の上限


def _leaky_relu() -> nn.Module:
    return nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True)


class _Blur(nn.Module):
    # 拡大・縮小のときの折り返し雑音を抑える
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel_1d", torch.tensor([1.0, 2.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel_1d[None, :] * self.kernel_1d[:, None]
        kernel = (kernel / kernel.sum()).expand(x.shape[1], 1, 3, 3)
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel, groups=x.shape[1])


class _EqualLinear(nn.Module):
    # 実際に使う重み = 保存する重み x 学習率の倍率。倍率を小さくすると、同じ最適化の歩幅で重みの動きが小さくなる
    def __init__(self, in_features: int, out_features: int, learning_rate_multiplier: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (MAPPING_WEIGHT_SCALE / learning_rate_multiplier))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.learning_rate_multiplier = learning_rate_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.learning_rate_multiplier, bias=self.bias * self.learning_rate_multiplier)


class MappingNetwork(nn.Module):
    # 入力のノイズ(潜在ベクトル)を、生成器の各層に渡すスタイルに写す
    def __init__(self, latent_dim: int, depth: int, learning_rate_multiplier: float):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([_EqualLinear(latent_dim, latent_dim, learning_rate_multiplier), _leaky_relu()])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(F.normalize(z, dim=1))


class _Conv2DMod(nn.Module):
    # スタイルで重みを変える畳み込み(変調)。demodulate=True では、出力の大きさがスタイルによらずそろうよう重みを正規化する
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, demodulate: bool = True):
        super().__init__()
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size)))
        nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        weights = self.weight[None] * (style[:, None, :, None, None] + 1)
        if self.demodulate:
            weights = weights * torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPSILON)
        # 画像ごとに違う重みを、グループ化した畳み込み1回で計算する
        x = x.reshape(1, -1, height, width)
        weights = weights.reshape(batch * self.out_channels, *weights.shape[2:])
        x = F.conv2d(x, weights, padding=(self.kernel_size - 1) // 2, groups=batch)
        return x.reshape(-1, self.out_channels, height, width)


class _ToImage(nn.Module):
    # 各解像度の特徴から画像への寄与を作り、前の解像度からの寄与に足して、次の解像度へ拡大する
    def __init__(self, latent_dim: int, in_channels: int, image_channels: int, upsample: bool):
        super().__init__()
        self.to_style = nn.Linear(latent_dim, in_channels)
        self.conv = _Conv2DMod(in_channels, image_channels, 1, demodulate=False)
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), _Blur()) if upsample else None

    def forward(self, x: torch.Tensor, previous: torch.Tensor | None, style: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, self.to_style(style))
        if previous is not None:
            x = x + previous
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class _GeneratorBlock(nn.Module):
    def __init__(self, latent_dim: int, in_channels: int, filters: int, image_channels: int, upsample: bool, upsample_image: bool):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if upsample else None
        self.to_style1 = nn.Linear(latent_dim, in_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = _Conv2DMod(in_channels, filters, 3)
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = _Conv2DMod(filters, filters, 3)
        self.activation = _leaky_relu()
        self.to_image = _ToImage(latent_dim, filters, image_channels, upsample_image)

    def forward(self, x: torch.Tensor, previous_image: torch.Tensor | None, style: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.upsample is not None:
            x = self.upsample(x)
        # ノイズの画像(画像ごとに1枚、最終の解像度)の左上を、この解像度の大きさだけ使う
        noise = noise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(noise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(noise).permute((0, 3, 2, 1))
        x = self.activation(self.conv1(x, self.to_style1(style)) + noise1)
        x = self.activation(self.conv2(x, self.to_style2(style)) + noise2)
        return x, self.to_image(x, previous_image, style)


class Generator(nn.Module):
    def __init__(self, image_size: int, latent_dim: int, capacity: int, image_channels: int):
        super().__init__()
        self.image_size = image_size
        self.num_layers = int(math.log2(image_size // INITIAL_SIZE)) + 1
        filters = [min(capacity * (2 ** (i + 1)), MAX_FEATURES) for i in range(self.num_layers)][::-1]
        filters = [filters[0], *filters]
        self.initial_block = nn.Parameter(torch.randn((1, filters[0], INITIAL_SIZE, INITIAL_SIZE)))
        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList(
            _GeneratorBlock(latent_dim, in_channels, out_channels, image_channels, upsample=index != 0, upsample_image=index != self.num_layers - 1)
            for index, (in_channels, out_channels) in enumerate(zip(filters[:-1], filters[1:]))
        )

    def forward(self, styles: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # styles: (バッチ, 段の数, 潜在の次元)、noise: (バッチ, image_size, image_size, 1)。出力はおおむね0〜1の画像
        x = self.initial_conv(self.initial_block.expand(styles.shape[0], -1, -1, -1))
        image = None
        for style, block in zip(styles.transpose(0, 1), self.blocks):
            x, image = block(x, image, style, noise)
        return image


def generate_in_chunks(generator: Generator, styles: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # noise は字ごと(バッチと同じ数)か、全員で共有する1枚
    return torch.cat([
        generator(styles[start:start + GENERATION_CHUNK], noise if len(noise) == 1 else noise[start:start + GENERATION_CHUNK])
        for start in range(0, len(styles), GENERATION_CHUNK)
    ])


class _DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int, downsample: bool):
        super().__init__()
        self.residual_conv = nn.Conv2d(in_channels, filters, 1, stride=2 if downsample else 1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1), _leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1), _leaky_relu(),
        )
        self.downsample = nn.Sequential(_Blur(), nn.Conv2d(filters, filters, 3, padding=1, stride=2)) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return (x + residual) * (1 / math.sqrt(2))  # 2つを足した分散が、足す前と同じ大きさになるようにそろえる


class Discriminator(nn.Module):
    # 出力は1画像につき1つの値。このモデルの損失では、負が本物らしく、正が偽物らしい向き
    def __init__(self, image_size: int, capacity: int, image_channels: int):
        super().__init__()
        num_downsamples = int(math.log2(image_size // FINAL_SIZE))
        filters = [image_channels] + [min((capacity * 4) * (2 ** i), MAX_FEATURES) for i in range(num_downsamples + 1)]
        channel_pairs = list(zip(filters[:-1], filters[1:]))
        self.blocks = nn.ModuleList(_DiscriminatorBlock(in_channels, out_channels, downsample=index != len(channel_pairs) - 1) for index, (in_channels, out_channels) in enumerate(channel_pairs))
        last_channels = filters[-1]
        self.final_conv = nn.Conv2d(last_channels + 1, last_channels, 3, padding=1)  # +1 はミニバッチの標準偏差の特徴
        self.to_logit = nn.Linear(FINAL_SIZE * FINAL_SIZE * last_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        # ミニバッチの標準偏差: MINIBATCH_STD_GROUP 枚ずつの組で、画像どうしの特徴の違い(標準偏差の平均)を1チャンネル加える。
        # 組の中の画像は、バッチの中で batch/group ずつ離れた位置にある
        group = math.gcd(MINIBATCH_STD_GROUP, x.shape[0])
        stddev = x.view(group, -1, *x.shape[1:])
        stddev = (stddev - stddev.mean(0)).pow(2).mean(0).add(EPSILON).sqrt().mean(dim=(1, 2, 3), keepdim=True)
        stddev = stddev.repeat(group, 1, x.shape[2], x.shape[3])
        x = self.final_conv(torch.cat([x, stddev], 1))
        return self.to_logit(x.reshape(x.shape[0], -1)).squeeze(1)


def init_weights(generator: Generator, discriminator: Discriminator) -> None:
    # 全結合・畳み込みはHeの初期化。ノイズの寄与は0から始める(写像ネットワークは _EqualLinear の初期化を使う)
    for module in [*generator.modules(), *discriminator.modules()]:
        if type(module) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
    for block in generator.blocks:
        for layer in (block.to_noise1, block.to_noise2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
