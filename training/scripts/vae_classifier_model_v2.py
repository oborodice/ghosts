#!/usr/bin/env python3
# 画像ベースの本物/偽物判定分類器のアーキテクチャ定義と、学習済みモデルの保存・読み込み
from pathlib import Path

import torch
import torch.nn as nn

from vae_classifier_dataset_v2 import CANVAS_SIZE

HIDDEN_CHANNELS = (16, 32)  # 「小さいCNN」で十分という判断(既存の数値特徴量分類器と同程度のパラメータ規模感)


class ImageClassifier(nn.Module):
    def __init__(self, image_size: int, hidden_channels: tuple[int, int]) -> None:
        super().__init__()
        c1, c2 = hidden_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # プーリング段数からの手計算(image_size // 4)は、self.convの構成を変えたときに追従し忘れて
        # 形状不一致を起こしやすいため、実際に1回通して畳み込み後のサイズを直接求める
        with torch.no_grad():
            conv_out_features = self.conv(torch.zeros(1, 1, image_size, image_size)).numel()
        self.head = nn.Linear(conv_out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1)).squeeze(-1)  # logits


def load_classifier(path: Path, device: torch.device) -> ImageClassifier:
    saved = torch.load(path, map_location=device)
    model = ImageClassifier(saved["image_size"], saved["hidden_channels"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model


def save_classifier(model: ImageClassifier, path: Path) -> None:
    # テストサンプルごとの分類確率を使った事後分析(既知指標・単純な交絡との相関など)を、学習のたびの
    # 初期化・ミニバッチ順序のランダム性に左右されず繰り返し行えるようにするため、学習済みモデルを
    # 保存する。CANVAS_SIZE・HIDDEN_CHANNELSは現状モジュール定数で固定だが、将来変更されても
    # 保存済みモデルの構造を復元できるよう、値ごと保存しておく
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "image_size": CANVAS_SIZE, "hidden_channels": HIDDEN_CHANNELS}, path
    )
