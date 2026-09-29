# 学習したGANを、文字認識のモデル(train_classifier.py)の特徴の空間で評価する(評価のスクリプトと、学習中の保存ごとの評価で共通)。
# 推論と同じ流れ(simplex noiseの値 → 正規分布への変換 → 写像ネットワーク → 生成器、移動平均の版の重み)で字を作り、次を測る:
# - 新しさ: 知らない字になっているかを見るための、実在字として確信を持って読まれない(最も高い確率が0.5未満の)字の割合
# - 2字の種類の数: 同じ字ばかり出る崩壊を見るための、ランダムに選んだ2字が同じ種類である確率の逆数(実在字どうしの値を並べる)
# - 崩れの割合: 外周の枠や塊・白黒の反転・塗りつぶし
# - 精度・再現率(Kynkäänniemi et al. 2019)と、密度・網羅率(Naeem et al. 2020): 生成物と実在字の分布の重なり(実在字どうしの値を上限の目安に並べる)
#   - 精度・密度は形の質(生成物が実在字の分布の中に入るか)、再現率・網羅率は広がり(実在字の分布を覆うか)を見る
# - なめらかさ: simplex noiseの軌跡(表示と同じ作り方)の上で、1フレームの特徴の変化が、別の字への急な切り替わりとみなせる距離を超える割合
# - 写し: 学習データの丸写しを見るための、学習データの中で最も近い画像との距離(ピクセルの差の二乗平均の平方根)が、実在字どうしの距離
#   (実在字と、学習データの中で最も近い別の画像。同じ字の別の書風が近くにある)の1%点以下の字の割合。新しさは実在字として読まれる字を
#   数えるが、実在字に似ているだけの字と写した字を区別できないので、ピクセルの近さで別に見る
# 学習の途中のどの時点を使うかは、関門(崩れ・崩壊・新しさ・写し・なめらかさ)をすべて満たすもののうち、網羅率が最も高いものを選ぶ
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from glyph.classifier import Calibration, GlyphClassifier
from glyph.inference import GlyphGenerator
from glyph.metrics import artifact_rates, pair_types
from glyph.walk import DEFAULT_NOISE_SPEED, DISPLAY_FPS, simplex_scattered, simplex_walk

NOVELTY_PROBABILITY = 0.5
PRECISION_RECALL_NEIGHBORS = 5
PAIR_TYPE_SAMPLES = 2000
FEATURE_CHUNK = 500
WALK_SECONDS = 10  # なめらかさに使う軌跡1本の長さ(秒)
COPY_REFERENCE_SAMPLES = 2000  # 実在字どうしの距離を測る実在字の数
COPY_REFERENCE_QUANTILE = 0.01  # 学習データとの距離が、実在字どうしの距離のこの分位点以下の字を、写しとみなす
NEAREST_CHUNK = 256  # 最も近い画像を探すときに、一度に比べる字の数(距離の表が 字の数 x 学習データの数 になるため)
# 関門(使う時点の候補に残すための条件)。外周の枠・白黒の反転は、これまでの学習の合格の条件と同じ。ほかは基準の学習の結果を見て見直す仮の値
# (新しさは、これまでで最も良い重みで約98%だったので、それを落とさない程度にしている)
MAX_FRAMED = 0.02
MAX_INVERTED = 0.005
MAX_FILLED = 0.02
MIN_PAIR_TYPES_RATIO = 0.5  # 2字の種類の数が、実在字の値のこの割合以上(崩壊していない)
MIN_NOVELTY = 0.95
MAX_COPY_SHARE = 0.01
MAX_JUMP_RATE = 0.001


class Evaluation(NamedTuple):
    novelty: float  # 実在字として確信を持って読まれない字の割合
    median_max_probability: float
    copy_share: float  # 学習データとの距離が、実在字どうしの距離の1%点以下の字の割合
    nearest_median: float  # 学習データの中で最も近い画像との距離の中央値
    pair_types: float
    real_pair_types: float  # 実在字どうしの2字の種類の数(比べる目安)
    framed: float
    inverted: float
    filled: float
    precision: float
    recall: float
    density: float
    coverage: float
    jump_rate: float  # 軌跡の1フレームの変化のうち、別の字への急な切り替わりとみなせる割合
    change_median: float  # 軌跡の1フレームの特徴の変化の中央値
    change_p99: float

    def failed_gates(self) -> list[str]:
        # 満たさなかった関門の名前(空なら候補に残る)
        checks = {"framed": self.framed <= MAX_FRAMED, "inverted": self.inverted <= MAX_INVERTED, "filled": self.filled <= MAX_FILLED,
                  "pair_types": self.pair_types >= MIN_PAIR_TYPES_RATIO * self.real_pair_types, "novelty": self.novelty >= MIN_NOVELTY,
                  "copy_share": self.copy_share <= MAX_COPY_SHARE, "jump_rate": self.jump_rate <= MAX_JUMP_RATE}
        return [name for name, passed in checks.items() if not passed]

    def report(self) -> str:
        return "\n".join([
            f"novelty: share with max prob < {NOVELTY_PROBABILITY} = {100 * self.novelty:.1f}% (median max prob {self.median_max_probability:.3f})",
            f"copies: share as close to a training image as real kanji at their {COPY_REFERENCE_QUANTILE:.0%} point = {100 * self.copy_share:.2f}% "
            f"(median nearest distance {self.nearest_median:.3f})",
            f"pair types: generated {self.pair_types:.1f} | real {self.real_pair_types:.1f}",
            f"artifacts: framed {100 * self.framed:.1f}% | inverted {100 * self.inverted:.1f}% | filled {100 * self.filled:.1f}%",
            f"generated: precision {self.precision:.3f} recall {self.recall:.3f} density {self.density:.3f} coverage {self.coverage:.3f}",
            f"smoothness: jump rate {100 * self.jump_rate:.2f}% of frame transitions | "
            f"per-frame feature change median {self.change_median:.2f} p99 {self.change_p99:.2f}",
            f"gates: {'passed' if not self.failed_gates() else 'failed ' + ', '.join(self.failed_gates())}",
        ])

    def csv_row(self) -> list:
        # CSV_HEADER の順の値
        return [*self, " ".join(self.failed_gates())]


CSV_HEADER = [*Evaluation._fields, "failed_gates"]  # 評価の値を1行ずつ書き出すときの列(先頭に、どの時点かの列を足して使う)


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
    density = inside_real.sum(1).float().mean().item() / k
    recall = (fake_to_real.T < fake_radius[None, :]).any(1).float().mean().item()
    coverage = (fake_to_real.min(0).values < real_radius).float().mean().item()
    return precision, recall, density, coverage


class GlyphEvaluator:
    # 実在字の特徴は初めに1回だけ計算し、学習の途中の多くの時点を同じ実在字・同じsimplex noiseの位置で比べる
    @torch.no_grad()
    def __init__(self, classifier: GlyphClassifier, calibration: Calibration, images: torch.Tensor, samples: int, walks: int, seed: int,
                 device: torch.device):
        # images: 学習データ(字の数, 高さ, 幅)、uint8(0=紙〜255=インク)
        self.classifier, self.calibration = classifier, calibration
        self.samples, self.walks, self.seed, self.device = samples, walks, seed, device
        # 学習データは書風ごとに並んでいるので、ランダムな順のまま前半(比べる相手)・後半(実在字どうしの基準)に分ける
        real_indices = torch.from_numpy(np.random.default_rng(seed).permutation(len(images))[:2 * samples])
        real_images = images[real_indices.to(images.device)].to(device).float().div(255).unsqueeze(1)
        self.real_features, _ = _features_and_probabilities(classifier, real_images[:samples])
        reference_features, _ = _features_and_probabilities(classifier, real_images[samples:])
        self.real_pair_types = pair_types(self.real_features[:PAIR_TYPE_SAMPLES], calibration.same_glyph_distance)
        self.reference = _precision_recall_density_coverage(self.real_features, reference_features, PRECISION_RECALL_NEIGHBORS)
        # 写しの基準: 実在字それぞれと、学習データの中で最も近い別の画像との距離の分位点
        self.training_pixels = images.to(device).flatten(1).float().div_(255)  # 学習データ全体(約4GB)なので、その場で割り、メモリの上に2つ持たない
        self.training_squared_norms = self.training_pixels.pow(2).sum(1)
        copy_reference = torch.from_numpy(np.random.default_rng(seed).choice(len(images), COPY_REFERENCE_SAMPLES, replace=False)).to(device)
        reference_distances = self._nearest_training_distances(self.training_pixels[copy_reference], exclude=copy_reference)
        self.copy_distance = reference_distances.quantile(COPY_REFERENCE_QUANTILE).item()

    def _nearest_training_distances(self, pixels: torch.Tensor, exclude: torch.Tensor | None = None) -> torch.Tensor:
        # 各行について、学習データの中で最も近い画像との距離(ピクセルの差の二乗平均の平方根、0〜1)。excludeは、その行自身の添字(除く)
        distances = []
        for start in range(0, len(pixels), NEAREST_CHUNK):
            chunk = pixels[start:start + NEAREST_CHUNK]
            squared = chunk.pow(2).sum(1, keepdim=True) + self.training_squared_norms[None] - 2 * chunk @ self.training_pixels.T
            if exclude is not None:
                squared[torch.arange(len(chunk), device=chunk.device), exclude[start:start + NEAREST_CHUNK]] = float("inf")
            distances.append((squared.min(1).values.clamp(min=0) / pixels.shape[1]).sqrt())
        return torch.cat(distances)

    def reference_report(self) -> str:
        precision, recall, density, coverage = self.reference
        return (f"real vs real (reference): precision {precision:.3f} recall {recall:.3f} density {density:.3f} coverage {coverage:.3f} "
                f"(n={self.samples}, k={PRECISION_RECALL_NEIGHBORS}) | pair types from {min(PAIR_TYPE_SAMPLES, self.samples)} glyphs | "
                f"jump distance {self.calibration.jump_distance:.2f} | copy distance {self.copy_distance:.3f}")

    def _feature_changes(self, model: GlyphGenerator) -> torch.Tensor:
        # 表示と同じ速さで進む軌跡の、隣り合うフレームの特徴の距離(すべての軌跡をつなげたもの)
        frames = int(DISPLAY_FPS * WALK_SECONDS)
        changes = []
        for walk in range(self.walks):
            simplex_values = torch.from_numpy(simplex_walk(model.latent_dim, frames, DISPLAY_FPS, DEFAULT_NOISE_SPEED, self.seed + 1 + walk))
            features, _ = _features_and_probabilities(self.classifier, model.generate_in_chunks(simplex_values.to(self.device)))
            changes.append((features[1:] - features[:-1]).norm(dim=1).cpu())
        return torch.cat(changes)

    @torch.no_grad()
    def evaluate(self, model: GlyphGenerator) -> tuple[Evaluation, torch.Tensor]:
        # 返り値の2つめは、ばらばらの位置のsimplex noiseから作った字(表示で出うる字の分布)
        ink = model.generate_in_chunks(torch.from_numpy(simplex_scattered(model.latent_dim, self.samples, self.seed)).to(self.device))
        fake_features, probabilities = _features_and_probabilities(self.classifier, ink)
        top_probability = probabilities.max(1).values
        rates = artifact_rates(ink)
        precision, recall, density, coverage = _precision_recall_density_coverage(self.real_features, fake_features, PRECISION_RECALL_NEIGHBORS)
        changes = self._feature_changes(model)
        nearest = self._nearest_training_distances(ink.flatten(1))
        evaluation = Evaluation(
            novelty=(top_probability < NOVELTY_PROBABILITY).float().mean().item(),
            median_max_probability=top_probability.median().item(),
            copy_share=(nearest <= self.copy_distance).float().mean().item(),  # 同じ画像が2枚ある学習データでは基準が0になりうるので、0でも写しを数える
            nearest_median=nearest.median().item(),
            pair_types=pair_types(fake_features[:PAIR_TYPE_SAMPLES], self.calibration.same_glyph_distance),
            real_pair_types=self.real_pair_types,
            framed=rates.framed, inverted=rates.inverted, filled=rates.filled,
            precision=precision, recall=recall, density=density, coverage=coverage,
            jump_rate=(changes > self.calibration.jump_distance).float().mean().item(),
            change_median=changes.median().item(), change_p99=changes.quantile(0.99).item(),
        )
        return evaluation, ink


def best_by_coverage(evaluations: dict[str, Evaluation]) -> str | None:
    candidates = {name: evaluation for name, evaluation in evaluations.items() if not evaluation.failed_gates()}
    return max(candidates, key=lambda name: candidates[name].coverage) if candidates else None
