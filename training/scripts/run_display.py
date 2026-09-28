#!/usr/bin/env python3
# 生成器のONNX(export_onnx.py)をONNX Runtime(CPU)で1フレームずつ動かし、形を変え続ける字を、
# GPUでフィルタをかけて(glyph/renderer.py)ウィンドウに描き続ける。
# 1秒ごとに、FPSと、生成器・描画の1フレームの時間をターミナルに出す。--record を指定すると、ウィンドウを出さずに mp4 か GIF に書き出す
import argparse
import itertools
import time
from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import onnxruntime
import pygame
from OpenGL import GL
from opensimplex import OpenSimplex
from PIL import Image

from glyph.renderer import GlyphRenderer, create_window
from glyph.walk import DEFAULT_NOISE_SPEED, DISPLAY_FPS, simplex_frame

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Raspberry Pi Touch Display 2(5インチ・7インチ、720x1280で縦長が本来の向き)を横向きにした大きさ(Piでは画面を回転させて使う)
WINDOW_SIZE = (1280, 720)
REPORT_EVERY_SECONDS = 1.0
GIF_SIZE = 350  # GIFはファイルを小さくするため、この大きさで描く
GIF_TIME_UNIT_MS = 10
MP4_CRF = 18


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DATA_DIR / "onnx" / "glyph_generator.onnx")  # export_onnx.py で書き出した生成器
    parser.add_argument("--seed", type=int, default=0)  # simplex noiseの軌跡を選ぶ(変えると、別の字の変わり方になる)
    parser.add_argument("--fullscreen", action="store_true")
    # ウィンドウの大きさの倍率。既定は、開発機のMacで実物に近い見かけにする値(Macのウィンドウは約110〜130ppiで、7インチの画面の
    # 約210ppiより大きく見える)。Piでウィンドウとして実寸で開くときは 1 にする(全画面・書き出しでは使わない)
    parser.add_argument("--scale", type=float, default=0.6)
    # 指定すると、ウィンドウを出さずに --seconds 秒ぶんを書き出して終わる。拡張子で形式を選ぶ(.mp4 は画面全体を実寸で、.gif は字の正方形を小さく)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--seconds", type=float, default=10)
    return parser.parse_args()


class _GlyphSource:
    # フレームの番号から、字のインクの画像を作る(simplex noiseの軌跡を進め、生成器のONNXに通す)
    def __init__(self, model_path: Path, seed: int):
        self.generator = onnxruntime.InferenceSession(str(model_path))
        self.latent_dim = self.generator.get_inputs()[0].shape[1]
        self.image_size = self.generator.get_outputs()[0].shape[-1]
        self.walk_noise = OpenSimplex(seed)  # 字の変わり方を決める、潜在空間の軌跡のノイズ

    def ink(self, frame: int) -> np.ndarray:
        # 返り値は (解像度, 解像度)、0=紙〜1=インク
        simplex_values = simplex_frame(self.walk_noise, frame, self.latent_dim, DISPLAY_FPS, DEFAULT_NOISE_SPEED)
        return self.generator.run(None, {"simplex_values": simplex_values})[0][0, 0]


def _window_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.record is None:
        return round(WINDOW_SIZE[0] * args.scale), round(WINDOW_SIZE[1] * args.scale)
    return (GIF_SIZE, GIF_SIZE) if args.record.suffix == ".gif" else WINDOW_SIZE  # 動画は実物の画面と同じ画素で書き出す


def _timed_frame(source: _GlyphSource, renderer: GlyphRenderer, frame: int) -> tuple[float, float]:
    # 1フレームを生成して描く。返り値は(生成器の時間, 描画の時間)(ミリ秒)
    started = time.perf_counter()
    ink = source.ink(frame)
    generated = time.perf_counter()
    renderer.draw(ink, frame / DISPLAY_FPS)
    GL.glFinish()  # GPUの描画が終わるまで待ち、描画の時間を正しく測る
    return (generated - started) * 1000, (time.perf_counter() - generated) * 1000


def _quit_requested() -> bool:
    # Esc・q・ウィンドウを閉じる操作で終わる
    return any(event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q))
               for event in pygame.event.get())


def _run_window(source: _GlyphSource, renderer: GlyphRenderer) -> None:
    clock = pygame.time.Clock()
    frame, report_started, timings = 0, time.perf_counter(), []
    while not _quit_requested():
        timings.append(_timed_frame(source, renderer, frame))
        pygame.display.flip()
        clock.tick(DISPLAY_FPS)
        frame += 1
        elapsed = time.perf_counter() - report_started
        if elapsed >= REPORT_EVERY_SECONDS:
            generator_ms, render_ms = np.mean(np.array(timings), axis=0)
            print(f"{len(timings) / elapsed:.1f} fps | generator {generator_ms:.1f} ms, render {render_ms:.1f} ms per frame", flush=True)
            report_started, timings = time.perf_counter(), []


def _recorded_frames(source: _GlyphSource, renderer: GlyphRenderer, seconds: float, whole_window: bool) -> Iterator[np.ndarray]:
    # 1フレームずつ描いて読み戻す(長い動画でも全フレームをメモリに溜めないよう、1枚ずつ渡す)。終わったら1フレームの時間を出す
    timings = []
    for frame in range(int(seconds * DISPLAY_FPS)):
        timings.append(_timed_frame(source, renderer, frame))
        yield renderer.read_pixels(whole_window)
    generator_ms, render_ms = np.median(np.array(timings[1:]), axis=0)  # 最初のフレームは初期化の時間を含むので除く
    print(f"per frame: generator {generator_ms:.1f} ms, render {render_ms:.1f} ms (median over {len(timings) - 1} frames)")


def _frame_durations(count: int, fps: float) -> list[int]:
    # GIFの表示時間は10ミリ秒単位なので、1/fps 秒ずつに丸めると速さがずれる(30fpsの33ミリ秒は30ミリ秒になる)。
    # 始まりからの時刻を丸めて差をとり、30・40・30ミリ秒…のように配って、全体の長さを合わせる
    frame_starts_ms = [round(frame * 1000 / fps / GIF_TIME_UNIT_MS) * GIF_TIME_UNIT_MS for frame in range(count + 1)]
    return [end - start for start, end in zip(frame_starts_ms, frame_starts_ms[1:])]


def _write_gif(frames: Iterator[np.ndarray], output: Path) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(output, save_all=True, append_images=images[1:], duration=_frame_durations(len(images), DISPLAY_FPS), loop=0)


def _write_mp4(frames: Iterator[np.ndarray], output: Path) -> None:
    # H.264(yuv420p はほとんどの再生環境で開ける形式)。CRF は画質(小さいほど高画質で、18 は見た目では元とほぼ区別がつかない)
    first = next(frames)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=round(DISPLAY_FPS), options={"crf": str(MP4_CRF)})
        stream.pix_fmt = "yuv420p"
        stream.height, stream.width = first.shape[:2]  # 大きさは最初のフレームに合わせる(指定しないと 640x480 になる)
        for frame in itertools.chain([first], frames):
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())  # エンコーダーに溜まっている残りを書き出す


def main() -> None:
    args = _parse_args()
    source = _GlyphSource(args.model, args.seed)
    target_size = create_window(_window_size(args), fullscreen=args.fullscreen and args.record is None, hidden=args.record is not None)
    renderer = GlyphRenderer(source.image_size, target_size)
    # ウィンドウの後片付けは、pygame が終了時に自分で行う
    if args.record is None:
        _run_window(source, renderer)
        return
    args.record.parent.mkdir(parents=True, exist_ok=True)
    record_gif = args.record.suffix == ".gif"
    frames = _recorded_frames(source, renderer, args.seconds, whole_window=not record_gif)
    if record_gif:
        _write_gif(frames, args.record)
    else:
        _write_mp4(frames, args.record)
    print(f"Saved {args.record}")


if __name__ == "__main__":
    main()
