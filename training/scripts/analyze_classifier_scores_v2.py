#!/usr/bin/env python3
# 学習済みの画像ベース分類器(evaluate_generation_realism_visual_v2.pyで学習・保存したもの)が、
# 実データと生成データをどのような根拠で見分けているかを分析する。テストサンプルごとの分類確率と、
# (a) triple_junctions・ストローク長・offsetのサンプル内ばらつきといった既知指標、(b) ストローク数・
# 総ストローク長といった単純な交絡、との相関(ピアソン相関係数)を計算する
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

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
    # offsetのばらつき・ストローク長は通常、複数サンプルにまたがる集団の標準偏差・平均として測る
    # 指標であり、ここではサンプルごとの分類確率と相関を取るため、offset_stdは「そのサンプル自身の
    # ストローク間でのばらつき」、stroke_lengthは「そのサンプル自身のストロークの平均長」に読み替える
    length = np.linalg.norm(end_points - start_points, axis=-1)
    offset_magnitude = np.linalg.norm(offsets, axis=-1)
    existence_t = torch.from_numpy(existence.astype("float32"))

    length_mean, _, _ = masked_mean_std(torch.from_numpy(length).float(), existence_t)
    _, offset_std, _ = masked_mean_std(torch.from_numpy(offset_magnitude).float(), existence_t)
    _, _, triple = crossings_and_triple_junctions(start_points, end_points, offsets, existence)

    return {
        "triple_junctions": triple,
        "stroke_length_mean": length_mean.numpy(),
        "offset_std_within_sample": offset_std.numpy(),
    }


def _confounds(start_points: np.ndarray, end_points: np.ndarray, existence: np.ndarray) -> dict[str, np.ndarray]:
    length = np.linalg.norm(end_points - start_points, axis=-1)
    return {
        "stroke_count": existence.sum(axis=1).astype(float),
        "total_stroke_length": (length * existence).sum(axis=1),
    }


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


if __name__ == "__main__":
    main()
