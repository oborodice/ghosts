#!/usr/bin/env python3
# 学習したGAN(train_glyph_gan.py のチェックポイント・スナップショット)を評価する(測る指標と関門は glyph_evaluation.py)。
# 複数を渡すと、同じ実在字・同じsimplex noiseの位置で比べ、関門をすべて満たすもののうち網羅率が最も高いものを示す。
# あわせて、生成した字を並べた画像を、それぞれの隣に保存する
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from glyph_classifier import load_classifier
from glyph_evaluation import CSV_HEADER, Evaluation, GlyphEvaluator, best_by_coverage
from glyph_inference import load_glyph_generator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_GRID = 8  # 保存する画像に並べる字の数(縦横それぞれ)
SAMPLE_IMAGE_SIZE = 1024  # 保存する画像の大きさ(px)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--classifier", type=Path, default=DATA_DIR / "glyph_classifier.pt")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "glyphs_64.npz")
    parser.add_argument("--samples", type=int, default=10000)  # 精度・再現率に使う字の数(生成物・実在字それぞれ)
    parser.add_argument("--walks", type=int, default=64)  # なめらかさに使う軌跡の数
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)  # 評価の値を1行ずつ書き出す先
    return parser.parse_args()


def _save_sample_grid(ink: torch.Tensor, output: Path) -> None:
    # 見やすいよう、白地に黒の字にして保存する
    grid = ink[:SAMPLE_GRID ** 2, 0].cpu().numpy()
    tiles = np.concatenate([np.concatenate(list(grid[row * SAMPLE_GRID:(row + 1) * SAMPLE_GRID]), 1) for row in range(SAMPLE_GRID)], 0)
    Image.fromarray(((1 - tiles) * 255).astype(np.uint8)).resize((SAMPLE_IMAGE_SIZE, SAMPLE_IMAGE_SIZE), Image.BILINEAR).save(output)
    print(f"Saved {output}")


def _write_csv(evaluations: dict[str, Evaluation], output: Path) -> None:
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["checkpoint", *CSV_HEADER])
        for name, evaluation in evaluations.items():
            writer.writerow([name, *evaluation.csv_row()])
    print(f"Saved {output}")


def main() -> None:
    args = _parse_args()
    device = torch.accelerator.current_accelerator() or torch.device("cpu")

    classifier, calibration = load_classifier(args.classifier, device)
    evaluator = GlyphEvaluator(classifier, calibration, torch.from_numpy(np.load(args.data)["images"]), args.samples, args.walks, args.seed, device)
    print(f"classifier {args.classifier}: {calibration}")
    print(evaluator.reference_report())

    evaluations = {}
    for checkpoint in args.checkpoint:
        evaluation, ink = evaluator.evaluate(load_glyph_generator(checkpoint, device))
        evaluations[str(checkpoint)] = evaluation
        print(f"=== {checkpoint}\n{evaluation.report()}", flush=True)
        _save_sample_grid(ink, checkpoint.with_name(f"{checkpoint.stem}_samples.png"))

    if args.csv is not None:
        _write_csv(evaluations, args.csv)
    if len(evaluations) > 1:
        best = best_by_coverage(evaluations)
        print(f"best by coverage among those passing the gates: {best or 'none passed'}")


if __name__ == "__main__":
    main()
