#!/usr/bin/env python3
# 潜在変数をsimplex noiseで動かしたときの生成結果を、アニメーションGIFとして書き出す目視確認用の
# ツール。軌跡・デコードは、なめらかさを数値化する診断スクリプトと同じ実装を使う。
# VAEの学習・推論(生成)には一切組み込まれない、独立した事後診断用のスクリプト
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from report_morph_smoothness_v2 import FPS, SEED, DecodedFrames, decode_frames, generate_walks, prepare_latents
from vae_checkpoint_v2 import load_checkpoint
from vae_classifier_dataset_v2 import VIEWBOX_SIZE
from vae_data_v2 import prepare_datasets
from vae_eval_common import SEGMENTS_PER_CURVE, bezier_polyline
from vae_eval_common_v2 import GENERATION_SOFT_TEMPERATURE, encode_batch, load_batch, stroke_curves
from vae_model_v2 import CHECKPOINT_PATH

PERIOD_SECONDS = 30.0  # フロントエンドのSPEED(=1/周期)に対応する周期の秒数。大きいほどゆっくり動く
SECONDS = 20.0
IMAGE_SIZE = 240  # 1フレームの一辺(ピクセル)
SUPERSAMPLING = 2  # 線を滑らかに描くため、この倍率で描いてから縮小する
LINE_WIDTH = 3  # 縮小後の線の太さ(ピクセル)
GIF_DURATION_UNIT_MS = 10  # GIFの1フレームの表示時間は、この単位でしか指定できない


def _render_frame(frames: DecodedFrames, frame_index: int) -> Image.Image:
    image = Image.new("L", (IMAGE_SIZE * SUPERSAMPLING, IMAGE_SIZE * SUPERSAMPLING), 255)
    draw = ImageDraw.Draw(image)
    scale = IMAGE_SIZE * SUPERSAMPLING / VIEWBOX_SIZE
    curves = stroke_curves(
        frames.start[frame_index], frames.end[frame_index], frames.offsets[frame_index], frames.existence[frame_index]
    )
    for start, control, end in curves:
        points = bezier_polyline(start, control, end, SEGMENTS_PER_CURVE)
        draw.line(
            [(float(x) * scale, float(y) * scale) for x, y in points], fill=0, width=LINE_WIDTH * SUPERSAMPLING, joint="curve"
        )
    return image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def _frame_durations_ms(frame_count: int) -> list[int]:
    # 1フレームの表示時間をGIFの単位に丸めると、FPSによっては再生が実際より速く(遅く)なる(30fpsなら33msが
    # 30msになり、約11%速い)。累積の時刻を単位に丸めて、フレームごとの差を表示時間にすることで、
    # 単位に収まらない端数が積み重なって、全体の再生時間がずれるのを防ぐ
    def end_time_ms(index: int) -> int:
        return GIF_DURATION_UNIT_MS * round(index * 1000 / FPS / GIF_DURATION_UNIT_MS)

    return [end_time_ms(index + 1) - end_time_ms(index) for index in range(frame_count)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an animated GIF of generated characters while z follows simplex noise.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="Path to the VAE checkpoint to visualize")
    parser.add_argument("--output", type=Path, required=True, help="Path of the GIF to write")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cpu")  # モデルが小さく、CPUで十分な速さのため
    checkpoint = load_checkpoint(device, checkpoint_path=args.checkpoint)
    with torch.no_grad():
        mu_real, _ = encode_batch(checkpoint, load_batch(prepare_datasets(), "train", device))

    frame_count = int(SECONDS * FPS)
    walks = generate_walks(1, frame_count, checkpoint.latent_dim, PERIOD_SECONDS, SEED)
    frames = decode_frames(checkpoint, prepare_latents(walks, mu_real), GENERATION_SOFT_TEMPERATURE)

    images = [_render_frame(frames, frame_index) for frame_index in range(frame_count)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.output, save_all=True, append_images=images[1:], duration=_frame_durations_ms(frame_count), loop=0)
    print(f"Saved {args.output} ({frame_count} frames, {SECONDS:g}s at {FPS:g} fps)")


if __name__ == "__main__":
    main()
