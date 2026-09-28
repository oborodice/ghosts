#!/usr/bin/env python3
# 学習したGAN(train_glyph_gan.py のチェックポイント)を、文字認識のモデル(train_glyph_classifier.py)の特徴の空間で評価する。
# 推論と同じ流れ(simplex noiseの値 → 正規分布への変換 → 写像ネットワーク → 生成器、移動平均の版の重み)で字を作り、次を測る:
# - 新しさ: 知らない字になっているかを見るための、実在字として確信を持って読まれない(最も高い確率が0.5未満の)字の割合
# - 2字の種類の数: 同じ字ばかり出る崩壊を見るための、ランダムに選んだ2字が同じ種類である確率の逆数(実在字どうしの値を並べる)
# - 崩れの割合: 外周の枠や塊・白黒の反転・塗りつぶし
# - 精度・再現率(Kynkäänniemi et al. 2019)と、密度・網羅率(Naeem et al. 2020): 生成物と実在字の分布の重なり(実在字どうしの値を上限の目安に並べる)
#   - 精度・密度は形の質(生成物が実在字の分布の中に入るか)、再現率・網羅率は広がり(実在字の分布を覆うか)を見る
# - なめらかさ: simplex noiseの軌跡(表示側と同じ作り方)の上で、1フレームの特徴の変化が、別の字への急な切り替わりとみなせる距離を超える割合
# あわせて、生成した字を並べた画像を、チェックポイントの隣に保存する
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from glyph_classifier import Calibration, GlyphClassifier, load_classifier
from glyph_inference import GlyphGenerator, load_glyph_generator
from glyph_metrics import artifact_rates, pair_types
from glyph_walk import DEFAULT_NOISE_SPEED, DISPLAY_FPS, simplex_scattered, simplex_walk

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NOVELTY_PROBABILITY = 0.5
PRECISION_RECALL_NEIGHBORS = 5
PAIR_TYPE_SAMPLES = 2000
FEATURE_CHUNK = 500
WALK_SECONDS = 10  # なめらかさに使う軌跡1本の長さ(秒)
SAMPLE_GRID = 8  # 保存する画像に並べる字の数(縦横それぞれ)
SAMPLE_IMAGE_SIZE = 1024  # 保存する画像の大きさ(px)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, default=DATA_DIR / "glyph_classifier.pt")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "glyphs_64.npz")
    parser.add_argument("--samples", type=int, default=10000)  # 精度・再現率に使う字の数(生成物・実在字それぞれ)
    parser.add_argument("--walks", type=int, default=64)  # なめらかさに使う軌跡の数
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def _scattered_glyphs(model: GlyphGenerator, samples: int, seed: int, device: torch.device) -> torch.Tensor:
    # 軌跡とは別に、ばらばらの位置のsimplex noiseから字を作る(表示側で出うる字の分布)
    return model.generate_in_chunks(torch.from_numpy(simplex_scattered(model.latent_dim, samples, seed)).to(device))


def _features_and_probabilities(classifier: GlyphClassifier, ink: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.cat([classifier.features(ink[start:start + FEATURE_CHUNK]) for start in range(0, len(ink), FEATURE_CHUNK)])
    return features, F.softmax(classifier.head(features), 1)


def _kth_radius(features: torch.Tensor, k: int) -> torch.Tensor:
    # 各点から、自分を除いて近いk個目の点までの距離
    return torch.cdist(features, features).topk(k + 1, largest=False).values[:, -1]


def _precision_recall_density_coverage(real: torch.Tensor, fake: torch.Tensor, k: int) -> tuple[float, float, float, float]:
    real_radius, fake_radius = _kth_radius(real, k), _kth_radius(fake, k)
    fake_to_real = torch.cdist(fake, real)
    inside_real = fake_to_real < real_radius[None, :]
    precision = inside_real.any(1).float().mean().item()
    density = inside_real.float().sum(1).mean().item() / k
    recall = (fake_to_real.T < fake_radius[None, :]).any(1).float().mean().item()
    coverage = (fake_to_real.min(0).values < real_radius).float().mean().item()
    return precision, recall, density, coverage


@torch.no_grad()
def _sample_metrics(ink: torch.Tensor, classifier: GlyphClassifier, calibration: Calibration, real: torch.Tensor, reference: torch.Tensor) -> None:
    fake_features, probabilities = _features_and_probabilities(classifier, ink)
    real_features, _ = _features_and_probabilities(classifier, real)
    reference_features, _ = _features_and_probabilities(classifier, reference)

    top_probability = probabilities.max(1).values
    novel_share = (top_probability < NOVELTY_PROBABILITY).float().mean().item()
    print(f"novelty: share with max prob < {NOVELTY_PROBABILITY} = {100 * novel_share:.1f}% (median max prob {top_probability.median().item():.3f})")

    pair_samples = min(PAIR_TYPE_SAMPLES, len(ink))
    generated_types = pair_types(fake_features[:pair_samples], calibration.same_glyph_distance)
    real_types = pair_types(real_features[:pair_samples], calibration.same_glyph_distance)
    print(f"pair types ({pair_samples} glyphs): generated {generated_types:.1f} | real {real_types:.1f}")

    rates = artifact_rates(ink)
    print(f"artifacts: framed {100 * rates.framed:.1f}% | inverted {100 * rates.inverted:.1f}% | filled {100 * rates.filled:.1f}%")

    for name, fake in (("real vs real (reference)", reference_features), ("generated", fake_features)):
        precision, recall, density, coverage = _precision_recall_density_coverage(real_features, fake, PRECISION_RECALL_NEIGHBORS)
        print(f"{name}: precision {precision:.3f} recall {recall:.3f} density {density:.3f} coverage {coverage:.3f} "
              f"(n={len(fake)}, k={PRECISION_RECALL_NEIGHBORS})")


@torch.no_grad()
def _smoothness(model: GlyphGenerator, classifier: GlyphClassifier, calibration: Calibration, walks: int, seed: int, device: torch.device) -> None:
    frames = int(DISPLAY_FPS * WALK_SECONDS)
    feature_changes = []
    for walk in range(walks):
        simplex_values = torch.from_numpy(simplex_walk(model.latent_dim, frames, DISPLAY_FPS, DEFAULT_NOISE_SPEED, seed + 1 + walk)).to(device)
        features, _ = _features_and_probabilities(classifier, model.generate_in_chunks(simplex_values))
        feature_changes.append((features[1:] - features[:-1]).norm(dim=1).cpu())
    feature_changes = torch.cat(feature_changes)
    jump_share = (feature_changes > calibration.jump_distance).float().mean().item()
    print(f"smoothness: jump rate {100 * jump_share:.2f}% of frame transitions (jump distance {calibration.jump_distance:.2f}) | "
          f"per-frame feature change median {feature_changes.median().item():.2f} p99 {feature_changes.quantile(0.99).item():.2f}")


def _save_sample_grid(ink: torch.Tensor, output: Path) -> None:
    # 見やすいよう、白地に黒の字にして保存する
    grid = ink[:SAMPLE_GRID ** 2, 0].cpu().numpy()
    tiles = np.concatenate([np.concatenate(list(grid[row * SAMPLE_GRID:(row + 1) * SAMPLE_GRID]), 1) for row in range(SAMPLE_GRID)], 0)
    Image.fromarray(((1 - tiles) * 255).astype(np.uint8)).resize((SAMPLE_IMAGE_SIZE, SAMPLE_IMAGE_SIZE), Image.BILINEAR).save(output)
    print(f"Saved {output}")


def main() -> None:
    args = _parse_args()
    device = torch.accelerator.current_accelerator() or torch.device("cpu")

    model = load_glyph_generator(args.checkpoint, device)
    classifier, calibration = load_classifier(args.classifier, device)
    # 実在字: 学習データは書風ごとに並んでいるので、ランダムな順のまま前半(比べる相手)・後半(実在字どうしの基準)に分ける
    images = np.load(args.data)["images"]
    real_indices = np.random.default_rng(args.seed).permutation(len(images))[:2 * args.samples]
    real_images = torch.from_numpy(images[real_indices]).float().div(255).unsqueeze(1).to(device)
    real, reference = real_images[:args.samples], real_images[args.samples:]

    print(f"=== {args.checkpoint}, classifier {args.classifier}: {calibration}")
    ink = _scattered_glyphs(model, args.samples, args.seed, device)
    _sample_metrics(ink, classifier, calibration, real, reference)
    _smoothness(model, classifier, calibration, args.walks, args.seed, device)
    _save_sample_grid(ink, args.checkpoint.with_name(f"{args.checkpoint.stem}_samples.png"))


if __name__ == "__main__":
    main()
