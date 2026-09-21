#!/usr/bin/env python3
# 学習済みの画像ベース分類器(evaluate_generation_realism_visual_v2.pyで学習・保存したもの)が、
# 実データと生成データをどのような根拠で見分けているかを分析する。テストサンプルごとの分類確率と、
# (a) triple_junctions・ストローク長・offsetのサンプル内ばらつきといった既知指標、(b) ストローク数・
# 総ストローク長といった単純な交絡、との相関(ピアソン相関係数)を計算する。ペアワイズ相関は指標を
# 1つずつ見た関係しか測れないため、既知指標・交絡を組み合わせたロジスティック回帰のAUCも計算し、
# 「指標の組み合わせでどこまで画像分類器のAUCの高さを説明できるか」を直接比較できるようにする
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from report_generation_stats_v2 import crossings_and_triple_junctions
from vae_checkpoint_v2 import load_checkpoint
from vae_classifier_dataset_v2 import render_batch, sample_fake_strokes
from vae_classifier_model_v2 import load_classifier
from vae_data_v2 import prepare_datasets
from vae_eval_common_v2 import encode_batch, load_batch, true_strokes_real
from vae_model_v2 import CHECKPOINT_PATH, select_device
from vae_synthetic_losses import masked_mean_std


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to the VAE checkpoint the classifier was trained on",
    )
    parser.add_argument("--classifier", type=Path, required=True, help="Path to the saved classifier model")
    return parser.parse_args()


def _known_indicators(
    start_points: np.ndarray, end_points: np.ndarray, offsets: np.ndarray, existence: np.ndarray
) -> dict[str, np.ndarray]:
    # 各指標をサンプル(1字)単位で計算する。triple_junctionsはもともとサンプルごとのカウントだが、
    # offset・ストローク長は通常、複数サンプルにまたがる集団の標準偏差・平均として測る指標であり、
    # ここではサンプルごとの分類確率と相関を取るため、いずれも「そのサンプル自身のストローク間での
    # 平均・ばらつき」に読み替える
    length = np.linalg.norm(end_points - start_points, axis=-1)
    offset_magnitude = np.linalg.norm(offsets, axis=-1)
    existence_t = torch.from_numpy(existence.astype("float32"))

    length_mean, _, _ = masked_mean_std(torch.from_numpy(length).float(), existence_t)
    offset_mean, offset_std, _ = masked_mean_std(torch.from_numpy(offset_magnitude).float(), existence_t)
    crossings, _, triple = crossings_and_triple_junctions(start_points, end_points, offsets, existence)

    return {
        "crossings": crossings,
        "triple_junctions": triple,
        "stroke_length_mean": length_mean.numpy(),
        "offset_mean_within_sample": offset_mean.numpy(),
        "offset_std_within_sample": offset_std.numpy(),
    }


def _confounds(start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray) -> dict[str, np.ndarray]:
    length = np.linalg.norm(end_points - start_points, axis=-1)
    return {
        "stroke_count": existence.sum(axis=1).astype(float),
        "total_stroke_length": (length * existence).sum(axis=1),
    }


def _logistic_regression_auc(labels: np.ndarray, *indicator_dicts: dict[str, np.ndarray]) -> float:
    # ペアワイズ相関と異なり、指標を組み合わせて1つのスコアにしたときの分離性能(AUC)を測る。
    # 特徴量5個・サンプル数千件規模の単純な凸最適化のため過学習のリスクは小さく、画像分類器の
    # AUCと同じval集合上でfit・評価してよいと判断し、train/testを分けていない。指標間でスケールが
    # 大きく異なる(件数系は数個、総ストローク長は数百)ため、標準化しないとlbfgsが収束しないことがある
    features = np.column_stack([values for indicators in indicator_dicts for values in indicators.values()])
    features = StandardScaler().fit_transform(features)
    model = LogisticRegression().fit(features, labels)
    predicted = model.predict_proba(features)[:, 1]
    return roc_auc_score(labels, predicted)


def _discordant_pairs(auc: float, n_real: int, n_fake: int) -> float:
    # AUCは「ランダムな正例・負例ペアのうち正しく順位付けできた割合」という意味を持つため、
    # (1-AUC)×正例数×負例数で不一致ペア数(タイは0.5単位で数えられる)を直接求められる。
    # 天井付近のAUC同士を比較するとき、この実数のほうがAUCの差が意味を持つかを判断しやすい
    return (1.0 - auc) * n_real * n_fake


def _probability_quantiles(probs: np.ndarray) -> str:
    quantile_points = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    values = np.percentile(probs, quantile_points)
    return " ".join(f"p{q}={v:.4f}" for q, v in zip(quantile_points, values))


def main() -> None:
    args = _parse_args()
    device = select_device()
    checkpoint = load_checkpoint(device, checkpoint_path=args.checkpoint)
    classifier = load_classifier(args.classifier, device)
    datasets = prepare_datasets()
    train_batch = load_batch(datasets, "train", device)
    val_batch = load_batch(datasets, "val", device)

    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, train_batch)

    true_strokes = true_strokes_real(checkpoint, val_batch)
    true_existence = val_batch.stroke_existence.cpu().numpy().astype(bool)
    fake_strokes, fake_existence = sample_fake_strokes(checkpoint, mu_real, true_strokes, true_existence)

    real_x = render_batch(true_strokes.start, true_strokes.end, true_strokes.offsets, true_existence)
    fake_x = render_batch(fake_strokes.start, fake_strokes.end, fake_strokes.offsets, fake_existence)
    x = torch.cat([real_x, fake_x]).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(classifier(x)).cpu().numpy()

    def _combine(real: dict[str, np.ndarray], fake: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: np.concatenate([real[name], fake[name]]) for name in real}

    known_indicators = _combine(
        _known_indicators(true_strokes.start, true_strokes.end, true_strokes.offsets, true_existence),
        _known_indicators(fake_strokes.start, fake_strokes.end, fake_strokes.offsets, fake_existence),
    )
    confounds = _combine(
        _confounds(true_strokes.start, true_strokes.end, true_existence),
        _confounds(fake_strokes.start, fake_strokes.end, fake_existence),
    )

    print(f"n_real={len(true_strokes.start)} n_fake={len(fake_strokes.start)}")
    print()
    print("=== correlation(classifier probability, known indicator) ===")
    for name, values in known_indicators.items():
        r, p = pearsonr(probs, values)
        print(f"{name}: r={r:.4f} (p={p:.2e})")
    print()
    print("=== correlation(classifier probability, confound) ===")
    for name, values in confounds.items():
        r, p = pearsonr(probs, values)
        print(f"{name}: r={r:.4f} (p={p:.2e})")
    print()

    # _combineがreal→fakeの順で連結しているため、ラベルも同じ順で揃える
    n_real, n_fake = len(true_strokes.start), len(fake_strokes.start)
    labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
    aucs = {
        "image classifier": roc_auc_score(labels, probs),
        "logistic regression (known indicators only)": _logistic_regression_auc(labels, known_indicators),
        "logistic regression (known indicators + confounds)": _logistic_regression_auc(
            labels, known_indicators, confounds
        ),
    }
    print(
        "=== AUC comparison "
        "(with discordant pair counts, since AUCs alone can look identical near the ceiling) ==="
    )
    for name, auc in aucs.items():
        discordant = _discordant_pairs(auc, n_real, n_fake)
        print(f"{name}: auc={auc:.5f} discordant_pairs≈{discordant:.0f} (of {n_real * n_fake})")
    print()

    print(
        "=== classifier probability distribution "
        "(real vs. fake should separate at the extremes under a ceiling effect) ==="
    )
    print(f"real: {_probability_quantiles(probs[:n_real])}")
    print(f"fake: {_probability_quantiles(probs[n_real:])}")


if __name__ == "__main__":
    main()
