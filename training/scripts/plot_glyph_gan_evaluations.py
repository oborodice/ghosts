#!/usr/bin/env python3
# 学習中の保存ごとの評価の値(train_glyph_gan.py が学習の名前のディレクトリに書く evaluation.csv)を、複数の学習ぶん並べて、
# 歩数ごとの推移を1枚の画像に描く(学習を延ばすか・どの時点を使うか・スイープの条件どうしの比べを、曲線で見るため)。
# 評価の値の意味と関門は glyph_evaluation.py。関門を満たさなかった時点は、白抜きの点で描く
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METRICS = ("coverage", "density", "precision", "recall", "novelty", "copy_share")  # 描く評価の値(evaluation.csv の列の名前)
COLUMNS = 3
FIGURE_SIZE_PER_PANEL = (4.5, 3.2)  # グラフ1つの大きさ(インチ)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 学習の名前(data/checkpoints/glyph_gan/<名前>/evaluation.csv を読む)。学習中に書いた evaluation.csv のパスでもよい
    # (評価のスクリプトの --csv の出力は、歩数の列がないので描けない)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _csv_path(run: str) -> Path:
    path = Path(run)
    return path if path.suffix == ".csv" else DATA_DIR / "checkpoints" / "glyph_gan" / run / "evaluation.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if "step" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} has no 'step' column: pass an evaluation.csv written during training")
        return list(reader)


def _draw_panel(axis: plt.Axes, metric: str, runs: dict[str, list[dict[str, str]]]) -> None:
    # 1つの評価の値の、学習ごとの曲線
    for run, rows in runs.items():
        # 評価の値を足す前に書いた evaluation.csv には、その列がない
        points = [(int(row["step"]), float(row[metric]), not row["failed_gates"]) for row in rows if row.get(metric)]
        if not points:
            continue
        steps, values, passed = zip(*points)
        line, = axis.plot(steps, values, label=run)
        axis.scatter(steps, values, s=14, zorder=3, edgecolors=line.get_color(),
                     facecolors=[line.get_color() if ok else "white" for ok in passed])
    axis.set_title(metric.replace("_", " "))
    axis.set_xlabel("step")
    axis.grid(alpha=0.3)
    if axis.lines:
        axis.legend(fontsize=8)


def _draw_figure(runs: dict[str, list[dict[str, str]]]) -> plt.Figure:
    rows_count = -(-len(METRICS) // COLUMNS)
    figure, axes = plt.subplots(rows_count, COLUMNS, figsize=(FIGURE_SIZE_PER_PANEL[0] * COLUMNS, FIGURE_SIZE_PER_PANEL[1] * rows_count))
    for axis, metric in zip(axes.flat, METRICS):
        _draw_panel(axis, metric, runs)
    for axis in axes.flat[len(METRICS):]:
        axis.set_visible(False)
    figure.tight_layout()
    return figure


def main() -> None:
    args = _parse_args()
    runs = {run: _read_rows(_csv_path(run)) for run in args.runs}
    figure = _draw_figure(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=110)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
