#!/usr/bin/env python3
# フォントで描いた漢字の画像(build_glyph_dataset.py で作る)で、GAN(glyph_gan.py)を学習する。
#
# 下の定数は学習の設定。スイープでは --override NAME=VALUE で1回の学習だけ変える(使った値はチェックポイントに残る)。
# そのうち、部分的な崩壊(同じ字ばかり出る)と長い学習での崩れを避けるために、比べて選んだもの:
# - 損失: 非飽和のロジスティック損失
# - 判別器: ミニバッチの標準偏差の層ありで、学習率は生成器と同じ
# - 写像ネットワーク: 学習率の倍率0.01(ゆっくり動かす)
# - 勾配の罰則: 判別器を滑らかに保つため、本物と偽物の両方に重み10で数歩ごとにかける(弱いR1では完全に崩壊した)
# - データ: インクの濃さ(黒地に白の字)のまま渡し、増強(位置ずれ・切り抜き)で空いた場所が背景と同じ0になるようにする
# - 増強: 位置ずれと切り抜き(漢字として成り立たない鏡文字を本物として見せないよう、左右反転はしない)
# ほかの値(学習率・バッチ・容量・潜在の次元・スタイルの切り替え確率・増強の確率・なめらかさの正則化・移動平均の半減期など)は、
# 比べていない仮の値
import argparse
import ast
import copy
import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import glyph_gan
from glyph_classifier import load_classifier
from glyph_evaluation import CSV_HEADER, GlyphEvaluator
from glyph_inference import load_glyph_generator
from glyph_metrics import artifact_rates, pairwise_distances

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVALUATION_CSV = "evaluation.csv"  # 保存ごとの評価の値の書き出し先(学習の名前のディレクトリの中)
# 以下の大文字の名前の、数・数の組の定数は、すべて学習の設定として扱う(--override で変えられ、チェックポイントに残る)
# モデル
CAPACITY = 12  # 生成器・判別器の容量(Pi 5 に載る大きさ)
LATENT_DIM = 256
MAPPING_DEPTH = 4
MAPPING_LEARNING_RATE_MULTIPLIER = 0.01
# 最適化
BATCH = 64
LEARNING_RATE = 2e-4  # 生成器(と写像ネットワーク)の学習率
DISCRIMINATOR_LEARNING_RATE_RATIO = 1.0  # 判別器の学習率 / 生成器の学習率
ADAM_BETAS = (0.5, 0.9)
# 正則化・増強
GRADIENT_PENALTY_WEIGHT = 10.0
GRADIENT_PENALTY_INTERVAL = 4
PATH_LENGTH_INTERVAL = 32
PATH_LENGTH_START = 5000
PATH_LENGTH_DECAY = 0.99
STYLE_MIX_PROBABILITY = 0.9  # 2つのスタイルを段の途中で切り替える確率
AUGMENT_PROBABILITY = 0.25  # バッチ単位
MAX_SHIFT_FRACTION = 1 / 8  # 増強の位置ずれの最大(縦横それぞれ、画像の大きさに対する割合)
CUTOUT_FRACTION = 1 / 2  # 増強の切り抜きの四角の一辺(画像の大きさに対する割合)
# 移動平均の版(推論に使う重み)
EMA_HALF_LIFE_KIMG = 10.0  # 半減期(画像の数、千枚単位)
EMA_RAMPUP_RATIO = 0.05  # 半減期を、学習の初めは「これまでに見た画像の数 x この割合」までに抑える
# 表示・見張り・保存
LOG_EVERY = 100
METRICS_EVERY = 1000
METRIC_SAMPLES = 512
SAVE_EVERY = 2500
KEEP_CHECKPOINTS = 3
MILESTONE_EVERY = 10000  # この歩数ごとのチェックポイントは、最新でなくても残す
# 保存ごとの評価(--classifier を指定したとき)。評価のスクリプトの既定と同じ量にし、あとで測り直した値と比べられるようにする
EVALUATION_SAMPLES = 10000
EVALUATION_WALKS = 64
EVALUATION_SEED = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)  # 出力先 data/checkpoints/glyph_gan/<name>/
    parser.add_argument("--data", type=Path, default=DATA_DIR / "glyphs_64.npz")
    # 学習の長さ(見せる本物の画像の数、千枚単位)。バッチの大きさによらず、同じ量の学習を同じ数で指定できる。
    # 小数も受け付ける(確かめのための短い学習用。例: バッチ64で 0.64 は10歩)
    parser.add_argument("--kimg", type=float, default=1920)
    parser.add_argument("--classifier", type=Path)  # 保存ごとにスナップショットを評価する文字認識のモデル(省略時は評価しない)
    parser.add_argument("--resume", action="store_true")  # 同じ名前の最新のチェックポイントから続ける
    parser.add_argument("--seed", type=int)
    # 定数を1回の学習だけ変える(例: --override LEARNING_RATE=1e-4)。何度でも指定できる
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE")
    return parser.parse_args()


def _fix_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _load_latest_checkpoint(output_dir: Path, device: torch.device) -> dict:
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoint to resume in {output_dir}")
    print(f"resuming from {checkpoints[-1].name}", flush=True)
    return torch.load(checkpoints[-1], map_location=device)


def _setting_constants() -> dict[str, int | float | tuple]:
    return {name: value for name, value in globals().items() if name.isupper() and isinstance(value, (int, float, tuple))}


def _set_setting(name: str, value: int | float | tuple) -> None:
    current = _setting_constants().get(name)
    if current is None:
        raise SystemExit(f"unknown setting {name} (choose from {', '.join(_setting_constants())})")
    if isinstance(current, float) and isinstance(value, int):
        value = float(value)
    if type(value) is not type(current):
        raise SystemExit(f"setting {name} must be {type(current).__name__}, got {value!r}")
    globals()[name] = value


def _restore_settings(config: dict) -> None:
    for name in _setting_constants():
        if name.lower() in config:
            _set_setting(name, config[name.lower()])


def _apply_overrides(overrides: list[str]) -> None:
    for override in overrides:
        name, _, text = override.partition("=")
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise SystemExit(f"--override {override}: the value must be a number or a tuple of numbers")
        _set_setting(name, value)
        print(f"override {name} = {value!r}", flush=True)


def _checkpoint_config(args: argparse.Namespace) -> dict:
    # チェックポイントに残す、学習に使った設定(推論側は、モデルの形(capacity・latent_dim など)をここから読む)
    return {"data": str(args.data), "kimg": args.kimg, "classifier": str(args.classifier) if args.classifier else None, "seed": args.seed} | {
        name.lower(): value for name, value in _setting_constants().items()}


@dataclass
class _Models:
    mapping: glyph_gan.MappingNetwork
    generator: glyph_gan.Generator
    discriminator: glyph_gan.Discriminator
    mapping_ema: glyph_gan.MappingNetwork  # 移動平均の版(推論に使う)
    generator_ema: glyph_gan.Generator
    generator_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer

    def checkpoint_components(self) -> tuple[tuple[str, nn.Module | torch.optim.Optimizer], ...]:
        return (("mapping", self.mapping), ("generator", self.generator), ("discriminator", self.discriminator),
                ("mapping_ema", self.mapping_ema), ("generator_ema", self.generator_ema),
                ("generator_optimizer", self.generator_optimizer), ("discriminator_optimizer", self.discriminator_optimizer))


def _build_models(image_size: int, device: torch.device) -> _Models:
    mapping = glyph_gan.MappingNetwork(LATENT_DIM, MAPPING_DEPTH, MAPPING_LEARNING_RATE_MULTIPLIER).to(device)
    generator = glyph_gan.Generator(image_size, LATENT_DIM, CAPACITY, image_channels=1).to(device)
    discriminator = glyph_gan.Discriminator(image_size, CAPACITY, image_channels=1).to(device)
    glyph_gan.init_weights(generator, discriminator)
    mapping_ema, generator_ema = copy.deepcopy(mapping).eval(), copy.deepcopy(generator).eval()
    for param in [*mapping_ema.parameters(), *generator_ema.parameters()]:
        param.requires_grad_(False)
    generator_optimizer = torch.optim.Adam([*generator.parameters(), *mapping.parameters()], lr=LEARNING_RATE, betas=ADAM_BETAS)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE * DISCRIMINATOR_LEARNING_RATE_RATIO, betas=ADAM_BETAS)
    return _Models(mapping, generator, discriminator, mapping_ema, generator_ema, generator_optimizer, discriminator_optimizer)


def _restore_models(models: _Models, state: dict) -> tuple[int, float | None]:
    for key, component in models.checkpoint_components():
        component.load_state_dict(state[key])
    # 最適化の状態には保存したときの学習率とβも入っているので、再開のときに上書きした設定を効かせるため、設定の値に戻す
    for optimizer, learning_rate in ((models.generator_optimizer, LEARNING_RATE), (models.discriminator_optimizer, LEARNING_RATE * DISCRIMINATOR_LEARNING_RATE_RATIO)):
        for group in optimizer.param_groups:
            group["lr"], group["betas"] = learning_rate, ADAM_BETAS
    return state["step"], state["path_length_mean"]


def _image_noise(batch: int, image_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch, image_size, image_size, 1, device=device)


@dataclass
class _Monitor:
    reference_noise: torch.Tensor  # 指標を測るときのノイズの画像(固定する)
    evaluator: GlyphEvaluator | None  # 保存ごとのスナップショットの評価(省略時は評価しない)


def _build_monitor(classifier_path: Path | None, images: torch.Tensor, device: torch.device) -> _Monitor:
    evaluator = None
    if classifier_path is not None:  # train_glyph_classifier.py で学習した文字認識のモデル(しきい値も一緒に入っている)
        # モデルを作るときの重みの初期化は、学習と同じ乱数を使う。評価の有無で学習が変わらないよう、別の乱数の流れで作る
        with torch.random.fork_rng(devices=[]):
            classifier, calibration = load_classifier(classifier_path, device)
        evaluator = GlyphEvaluator(classifier, calibration, images, EVALUATION_SAMPLES, EVALUATION_WALKS, EVALUATION_SEED, device)
    return _Monitor(_image_noise(1, images.shape[-1], device), evaluator)


def _prepare_evaluation_csv(output_dir: Path, step: int) -> None:
    # 再開するときに、これまでの評価の値を今の列に合わせて書き直す(評価の値の種類が増えたあとに再開すると、古い行の列がずれるため。
    # 古い行にない値は空にする)。再開した時点より後の行は消す(評価のあと、再開用のチェックポイントを書く前に止まると、
    # その時点を再開のあとにもう一度評価するため)
    csv_path = output_dir / EVALUATION_CSV
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if int(row["step"]) <= step]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, ["step", *CSV_HEADER], restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _random_styles(mapping: glyph_gan.MappingNetwork, num_layers: int, device: torch.device) -> torch.Tensor:
    # 返り値は (バッチ, 段の数, 潜在の次元)
    styles = mapping(torch.randn(BATCH, LATENT_DIM, device=device))[:, None, :].expand(-1, num_layers, -1)
    if random.random() < STYLE_MIX_PROBABILITY:
        switch_layer = int(random.random() * num_layers)
        second_styles = mapping(torch.randn(BATCH, LATENT_DIM, device=device))[:, None, :].expand(-1, num_layers - switch_layer, -1)
        styles = torch.cat([styles[:, :switch_layer], second_styles], 1)
    return styles


def _pixel_grid(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 行と列の番号(バッチの次元に合わせて並べ、字ごとの乱数と足せる形)
    _, _, height, width = images.shape
    return torch.arange(height, device=images.device)[None, :, None], torch.arange(width, device=images.device)[None, None, :]


def _random_translation(images: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = images.shape
    rows, cols = _pixel_grid(images)
    max_shift_y, max_shift_x = int(height * MAX_SHIFT_FRACTION + 0.5), int(width * MAX_SHIFT_FRACTION + 0.5)
    shift_y = torch.randint(-max_shift_y, max_shift_y + 1, (batch, 1, 1), device=images.device)
    shift_x = torch.randint(-max_shift_x, max_shift_x + 1, (batch, 1, 1), device=images.device)
    source_y, source_x = rows + shift_y, cols + shift_x  # 出力の各画素に、元の画像のどの画素を置くか
    inside_mask = (source_y >= 0) & (source_y < height) & (source_x >= 0) & (source_x < width)
    batch_index = torch.arange(batch, device=images.device)[:, None, None]
    shifted = images[batch_index, :, source_y.clamp(0, height - 1), source_x.clamp(0, width - 1)].permute(0, 3, 1, 2)
    return shifted * inside_mask[:, None]


def _random_cutout(images: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = images.shape
    rows, cols = _pixel_grid(images)
    cut_height, cut_width = int(height * CUTOUT_FRACTION + 0.5), int(width * CUTOUT_FRACTION + 0.5)
    # 四角の中心は、一辺が偶数のとき、四角が画像の外へ左右(上下)同じだけはみ出せるよう、範囲を1つ広げる
    top = torch.randint(0, height + (1 - cut_height % 2), (batch, 1, 1), device=images.device) - cut_height // 2
    left = torch.randint(0, width + (1 - cut_width % 2), (batch, 1, 1), device=images.device) - cut_width // 2
    cutout_mask = (rows >= top) & (rows < top + cut_height) & (cols >= left) & (cols < left + cut_width)
    return images * ~cutout_mask[:, None]


def _augment(images: torch.Tensor) -> torch.Tensor:
    # 大きさ・位置の範囲は、DiffAugment の公式の実装(https://github.com/mit-han-lab/data-efficient-gans)と同じ
    images = _random_translation(images)
    return _random_cutout(images)


def _maybe_augment(images: torch.Tensor) -> torch.Tensor:
    return _augment(images) if random.random() < AUGMENT_PROBABILITY else images


def _gradient_penalty(inputs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    gradients = torch.autograd.grad(outputs=logits.sum(), inputs=inputs, create_graph=True)[0]
    return (gradients.reshape(inputs.shape[0], -1).norm(2, dim=1) ** 2).mean()


def _discriminator_step(models: _Models, images: torch.Tensor, step: int) -> torch.Tensor:
    # 判別器の出力は、負が本物らしく、正が偽物らしい向き。
    # 損失はテンソルのまま返す(毎歩数値にすると、GPUの計算の終わりを毎歩待つことになり遅くなる)
    generator, device = models.generator, images.device
    real = images[torch.randint(0, len(images), (BATCH,), device=device)].float().div(255).unsqueeze(1)
    with torch.no_grad():
        styles = _random_styles(models.mapping, generator.num_layers, device)
        fake = generator(styles, _image_noise(BATCH, generator.image_size, device))
    real_inputs, fake_inputs = _maybe_augment(real), _maybe_augment(fake)
    apply_penalty = step % GRADIENT_PENALTY_INTERVAL == 0
    if apply_penalty:
        real_inputs.requires_grad_()
        fake_inputs.requires_grad_()
    real_logits, fake_logits = models.discriminator(real_inputs), models.discriminator(fake_inputs)
    loss = F.softplus(real_logits).mean() + F.softplus(-fake_logits).mean()
    total = loss
    if apply_penalty:
        total = total + GRADIENT_PENALTY_WEIGHT * (_gradient_penalty(real_inputs, real_logits) + _gradient_penalty(fake_inputs, fake_logits))
    models.discriminator_optimizer.zero_grad(set_to_none=True)
    total.backward()
    models.discriminator_optimizer.step()
    return loss.detach()


def _path_lengths(styles: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # スタイルを少し動かしたときの、画像の変化の大きさ(乱数の方向への投影の勾配)。一定に保つと、潜在空間の動きがなめらかになる
    noise = torch.randn_like(images) / math.sqrt(images.shape[2] * images.shape[3])
    gradients = torch.autograd.grad(outputs=(images * noise).sum(), inputs=styles, create_graph=True)[0]
    return (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()


def _generator_step(models: _Models, step: int, path_length_mean: float | None, device: torch.device) -> tuple[torch.Tensor, float | None]:
    # 判別器の重みの勾配は使わないので計算しない。返り値は(損失, 更新した経路長の移動平均)
    generator = models.generator
    models.discriminator.requires_grad_(False)
    styles = _random_styles(models.mapping, generator.num_layers, device)
    fake = generator(styles, _image_noise(BATCH, generator.image_size, device))
    loss = F.softplus(models.discriminator(_maybe_augment(fake))).mean()
    total = loss
    if step > PATH_LENGTH_START and step % PATH_LENGTH_INTERVAL == 0:
        path_lengths = _path_lengths(styles, fake)
        if path_length_mean is not None:
            total = total + ((path_lengths - path_length_mean) ** 2).mean()
        batch_path_length = path_lengths.detach().mean().item()
        path_length_mean = batch_path_length if path_length_mean is None else path_length_mean * PATH_LENGTH_DECAY + batch_path_length * (1 - PATH_LENGTH_DECAY)
    models.generator_optimizer.zero_grad(set_to_none=True)
    total.backward()
    models.generator_optimizer.step()
    models.discriminator.requires_grad_(True)
    return loss.detach(), path_length_mean


@torch.no_grad()
def _update_ema(models: _Models, images_seen: int) -> None:
    half_life = min(EMA_HALF_LIFE_KIMG * 1000, images_seen * EMA_RAMPUP_RATIO)
    ema_decay = 0.5 ** (BATCH / max(half_life, glyph_gan.EPSILON))
    for ema, model in ((models.mapping_ema, models.mapping), (models.generator_ema, models.generator)):
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.lerp_(param, 1 - ema_decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def _train_step(models: _Models, images: torch.Tensor, step: int, path_length_mean: float | None) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    # 返り値は(判別器の損失, 生成器の損失, 経路長の移動平均)
    discriminator_loss = _discriminator_step(models, images, step)
    generator_loss, path_length_mean = _generator_step(models, step, path_length_mean, images.device)
    _update_ema(models, (step + 1) * BATCH)
    return discriminator_loss, generator_loss, path_length_mean


@torch.no_grad()
def _metrics_text(models: _Models, monitor: _Monitor) -> str:
    # 保存ごとの評価の間を埋める、文字認識のモデルなしで測れる軽い見張り(学習の初めの崩壊・崩れを早く見つける):
    # スタイルの散らばり(別々のノイズから作ったスタイルどうしの平均距離 / スタイルの大きさ)と、崩れの割合
    mapping, generator = models.mapping, models.generator
    mapping.eval()
    generator.eval()
    style = mapping(torch.randn(METRIC_SAMPLES, LATENT_DIM, device=monitor.reference_noise.device))
    spread = (pairwise_distances(style).mean() / style.norm(dim=1).mean()).item()
    ink = glyph_gan.generate_in_chunks(generator, style[:, None, :].expand(-1, generator.num_layers, -1), monitor.reference_noise).clamp(0, 1)
    rates = artifact_rates(ink)
    mapping.train()
    generator.train()
    return f"style spread {spread:.3f} | framed {100 * rates.framed:.1f}% inverted {100 * rates.inverted:.1f}% filled {100 * rates.filled:.1f}%"


def _save_snapshot(models: _Models, step: int, config: dict, output_dir: Path) -> Path:
    # 推論に使う移動平均の版の重みだけを、保存ごとにすべて残す(再開用のチェックポイントより小さく、あとで学習の途中のどの時点も評価し直せる。
    # load_glyph_generator でそのまま読める)
    path = output_dir / f"snapshot_{step:07d}.pt"
    torch.save({"mapping_ema": models.mapping_ema.state_dict(), "generator_ema": models.generator_ema.state_dict(), "step": step, "config": config}, path)
    return path


def _evaluate_snapshot(evaluator: GlyphEvaluator, snapshot: Path, step: int, output_dir: Path) -> None:
    # 読み込むモデルの重みの初期化で学習の乱数が進まないよう、別の乱数の流れで読む(評価の有無で学習が変わらないように)
    with torch.random.fork_rng(devices=[]):
        model = load_glyph_generator(snapshot, evaluator.device)
    evaluation, _ = evaluator.evaluate(model)
    gates = "passed" if not evaluation.failed_gates() else "failed " + ", ".join(evaluation.failed_gates())
    print(f"evaluation at step {step}: coverage {evaluation.coverage:.3f} density {evaluation.density:.3f} precision {evaluation.precision:.3f} "
          f"recall {evaluation.recall:.3f} | pair types {evaluation.pair_types:.1f} | novelty {100 * evaluation.novelty:.1f}% | "
          f"copies {100 * evaluation.copy_share:.2f}% | jump rate {100 * evaluation.jump_rate:.2f}% | gates {gates}", flush=True)
    csv_path = output_dir / EVALUATION_CSV
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(["step", *CSV_HEADER])
        writer.writerow([step, *evaluation.csv_row()])


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint_{step:07d}.pt"


def _save_checkpoint(models: _Models, step: int, path_length_mean: float | None, config: dict, output_dir: Path) -> None:
    state = {key: component.state_dict() for key, component in models.checkpoint_components()}
    state |= {"step": step, "path_length_mean": path_length_mean, "config": config}
    torch.save(state, _checkpoint_path(output_dir, step))
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    for old_checkpoint in checkpoints[:-KEEP_CHECKPOINTS]:
        if int(old_checkpoint.stem.split("_")[1]) % MILESTONE_EVERY != 0:
            old_checkpoint.unlink()


def _train(models: _Models, images: torch.Tensor, monitor: _Monitor, step: int, path_length_mean: float | None, total_steps: int,
           config: dict, output_dir: Path) -> None:
    start_step, start_time = step, time.time()
    while step < total_steps:
        discriminator_loss, generator_loss, path_length_mean = _train_step(models, images, step, path_length_mean)
        step += 1
        if step % LOG_EVERY == 0 or step == total_steps:
            ms_per_step = (time.time() - start_time) / (step - start_step) * 1000
            print(f"step {step} ({step * BATCH / 1000:.1f} kimg) | D {discriminator_loss.item():.3f} G {generator_loss.item():.3f} "
                  f"PL {path_length_mean or 0:.3f} | {ms_per_step:.0f} ms/step", flush=True)
        if step % METRICS_EVERY == 0:
            print(f"metrics at step {step}: {_metrics_text(models, monitor)}", flush=True)
        if step % SAVE_EVERY == 0 or step == total_steps:
            # 評価を再開用のチェックポイントより先に済ませる(間で止まっても、再開したときにこの時点をもう一度評価する)
            snapshot = _save_snapshot(models, step, config, output_dir)
            if monitor.evaluator is not None:
                _evaluate_snapshot(monitor.evaluator, snapshot, step, output_dir)
            _save_checkpoint(models, step, path_length_mean, config, output_dir)
            start_step, start_time = step, time.time()  # 1歩の時間に、保存と評価の時間を含めない


def main() -> None:
    args = _parse_args()
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    if args.seed is not None:
        _fix_seed(args.seed)
    output_dir = DATA_DIR / "checkpoints" / "glyph_gan" / args.name
    # 結果を落としたあとは、podの上にチェックポイントがなく評価の値だけが残るので、それも見る
    if not args.resume and (any(output_dir.glob("*.pt")) or (output_dir / EVALUATION_CSV).exists()):
        # 別の学習のチェックポイント・スナップショット・評価の値が混ざらないように
        raise SystemExit(f"{output_dir} already has a run: continue it with --resume or use another --name")
    state = _load_latest_checkpoint(output_dir, device) if args.resume else None
    if state is not None:
        _restore_settings(state["config"])
    _apply_overrides(args.override)  # 再開のときに指定すると、保存した設定のうえから変える
    config = _checkpoint_config(args)
    if device.type == "mps" and BATCH * glyph_gan.MAX_FEATURES > glyph_gan.MPS_GROUPED_CONV_MAX_CHANNELS:
        raise SystemExit(f"BATCH {BATCH} is too large on Apple Silicon (MPS): at most {glyph_gan.GENERATION_CHUNK}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 学習データ(uint8、0=紙〜255=インク)を丸ごとデバイスに載せ、バッチは添字を選ぶだけで作る(読み込みの手間を減らす)
    images = torch.from_numpy(np.load(args.data)["images"]).to(device)
    image_size = images.shape[-1]
    print(f"device {device} | {len(images)} glyphs of {image_size}px from {args.data}", flush=True)
    models = _build_models(image_size, device)
    step, path_length_mean = _restore_models(models, state) if state is not None else (0, None)
    monitor = _build_monitor(args.classifier, images, device)
    _prepare_evaluation_csv(output_dir, step)

    _train(models, images, monitor, step, path_length_mean, math.ceil(args.kimg * 1000 / BATCH), config, output_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
