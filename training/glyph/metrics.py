# 生成した字を評価する指標のうち、学習中の見張り(train_gan.py)と評価(evaluate_gan.py)で共通に使うもの
from typing import NamedTuple

import torch

BORDER = 4  # 崩れの割合で見る、画像の外周の幅(px)
FRAMED_BORDER_INK = 0.2  # 外周のインクの平均がこれを超えたら、外周に枠や塊がある
INVERTED_BORDER_INK = 0.5  # 外周のインクの平均がこれを超えたら、白黒が反転している
FILLED_INK = 0.478  # 全体のインクの平均がこれ(実在字の99.9%点)を超えたら、塗りつぶしている


class ArtifactRates(NamedTuple):
    framed: float  # 外周に枠や塊がある字の割合
    inverted: float  # 白黒が反転した字の割合
    filled: float  # 外周は崩れていないが、塗りつぶした字の割合


def artifact_rates(ink: torch.Tensor) -> ArtifactRates:
    # ink: (字の数, 1, 高さ, 幅)、0=紙〜1=インク。実在字の外周にはインクがない(99.9%点で0)ので、外周は決め打ちのしきい値で見る
    ink = ink[:, 0]
    border = torch.cat([ink[:, :BORDER].flatten(1), ink[:, -BORDER:].flatten(1), ink[:, :, :BORDER].flatten(1), ink[:, :, -BORDER:].flatten(1)], 1).mean(1)
    total = ink.flatten(1).mean(1)
    return ArtifactRates(
        framed=(border > FRAMED_BORDER_INK).float().mean().item(),
        inverted=(border > INVERTED_BORDER_INK).float().mean().item(),
        filled=((total > FILLED_INK) & (border <= FRAMED_BORDER_INK)).float().mean().item(),
    )


def pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    # 全ての組の距離(重複なし)。torch.pdist はApple Silicon(MPS)で使えないので、cdistの上半分を取る
    rows, cols = torch.triu_indices(len(points), len(points), 1, device=points.device)
    return torch.cdist(points, points)[rows, cols]


def pair_types(features: torch.Tensor, same_glyph_distance: float) -> float:
    # 2字の種類の数: ランダムに選んだ2字が同じ種類(文字認識のモデルの特徴の距離がしきい値未満)である確率の逆数。
    # 同じ字ばかり出る崩壊を見る。かたまりをつなげないので、測る字の数に左右されにくい
    same_rate = (pairwise_distances(features) < same_glyph_distance).float().mean().item()
    return 1 / same_rate if same_rate > 0 else float("inf")
