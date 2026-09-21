#!/usr/bin/env python3
# 画像ベースの本物/偽物判定分類器が使う、実データ+生成データのデータセットを構築する共通ロジック。
# 分類器の学習・学習済み分類器のスコア分析の両方から使われる
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from vae_checkpoint_v2 import Checkpoint
from vae_eval_common import SEGMENTS_PER_CURVE, bezier_polyline, existence_mask_from_logits
from vae_eval_common_v2 import RealScaleStrokes, attract_to_latent_prior, reconstructed_strokes_real, stroke_curves

NEAREST_REAL_FILTER_THRESHOLD = 1.0  # これより実在字に近い生成サンプルは、ラベルの矛盾(ほぼ同じ入力なのに本物・偽物の両方に現れる)を避けるため除外する
VIEWBOX_SIZE = 109.0  # KanjiVGのSVGのviewBoxサイズ(データの座標系そのもの。training/data/kanjivg/*.svg参照)
# 診断用レンダリング解像度。フロントエンドの実装(描画バッファのサイズなど)がどうなっているかは切り離し、
# 「小さいアイコン程度の表示サイズで見てもなお判別できてしまうか」を確認するための値を、それ単体で
# 妥当かどうかで直接決める(目視で崩れず読み取れることを確認済み)
CANVAS_SIZE = 128
LINE_WIDTH = 4  # 字全体に対して自然な太さになるよう目視で選んだ値
BLUR_RADIUS = 0.5  # ラスタライズ特有のジャギー(輪郭のギザつき)を均し、サブピクセル単位の情報を分類器に渡さないための軽いぼかし


def _render_curves_to_array(curves: list[tuple[complex, complex, complex]]) -> np.ndarray:
    scale = CANVAS_SIZE / VIEWBOX_SIZE
    img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw = ImageDraw.Draw(img)
    r = LINE_WIDTH / 2
    for start, control, end in curves:
        poly = bezier_polyline(start, control, end, SEGMENTS_PER_CURVE) * scale
        points = [tuple(p) for p in poly]
        draw.line(points, fill=255, width=LINE_WIDTH, joint="curve")
        # PILのdraw.lineは端点が四角く切れるため、筆で書いたような自然な丸みを出すために円を足す
        for p in (points[0], points[-1]):
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    if BLUR_RADIUS > 0:
        img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    return np.asarray(img, dtype=np.float32) / 255.0


def render_batch(start_points: np.ndarray, end_points: np.ndarray, offsets: np.ndarray, existence: np.ndarray) -> torch.Tensor:
    images = np.stack(
        [
            _render_curves_to_array(
                stroke_curves(start_points[i], end_points[i], offsets[i], existence[i].astype(bool))
            )
            for i in range(len(start_points))
        ]
    )
    return torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # (N, 1, H, W)


def _sample_candidates(
    checkpoint: Checkpoint, mu_real: torch.Tensor, count: int
) -> tuple[RealScaleStrokes, np.ndarray, torch.Tensor]:
    # 後段のフィルタで減る分を見込み、countより多めにサンプリングする。実測ではフィルタで除外される
    # サンプルはごく僅か(数千件に1件程度)なため、係数・加算値自体は厳密なチューニング値ではなく
    # 余裕を持たせた値
    oversample = int(count * 1.2) + 50
    z_raw = torch.randn(oversample, checkpoint.latent_dim, device=mu_real.device)
    with torch.no_grad():
        z = attract_to_latent_prior(z_raw, mu_real)
        nearest_dist = torch.cdist(z, mu_real).min(dim=1).values
        decoder_output = checkpoint.model.decode(z)
    strokes = reconstructed_strokes_real(checkpoint, decoder_output)
    existence_pred = existence_mask_from_logits(decoder_output.stroke_existence_logits)
    return strokes, existence_pred, nearest_dist


def _filter_to_count(
    strokes: RealScaleStrokes, existence_pred: np.ndarray, nearest_dist: torch.Tensor, count: int
) -> tuple[RealScaleStrokes, np.ndarray]:
    # 実在字に極端に近いサンプルは、ラベルの矛盾(ほぼ同じ入力なのに本物・偽物の両方に現れる)を避けるため除外する
    keep = (nearest_dist >= NEAREST_REAL_FILTER_THRESHOLD).cpu().numpy()
    filtered_count = int(keep.sum())
    print(f"Generated {len(nearest_dist)} samples, filtered out {len(nearest_dist) - filtered_count} near-duplicate(s)")
    if filtered_count < count:
        raise RuntimeError(f"Not enough fake examples after filtering: {filtered_count} < {count}")

    idx = np.where(keep)[0][:count]
    return RealScaleStrokes(strokes.start[idx], strokes.end[idx], strokes.offsets[idx]), existence_pred[idx]


def sample_fake_strokes(
    checkpoint: Checkpoint, mu_real: torch.Tensor, true_strokes: RealScaleStrokes, true_existence: np.ndarray
) -> tuple[RealScaleStrokes, np.ndarray]:
    # 実データ(true_strokes/true_existence)と同数の生成データ(フェイク)を、実データに極端に
    # 近いサンプルを除外した上でサンプリングする。呼び出し側(学習・分析)がそれぞれ画像化・
    # 指標計算に使えるよう、ストロークデータのまま返す(画像へのレンダリングはしない)
    strokes, existence_pred, nearest_dist = _sample_candidates(checkpoint, mu_real, len(true_strokes.start))
    return _filter_to_count(strokes, existence_pred, nearest_dist, len(true_strokes.start))
