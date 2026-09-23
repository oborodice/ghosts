#!/usr/bin/env python3
# 混ぜ合わせ生成(attract_to_latent_prior)時のモーフィングで、ポインタ(始点・終点の参照先頂点)の
# 割り当てがフレーム間でどれだけ安定しているかを、ソフトポインタ版・ハードポインタ版のチェックポイントで
# 比較する。ハードポインタは離散選択のため、zが決定境界を跨いだ瞬間に接続先が「スナップ」する
# 可能性がある一方、ソフトポインタは連続的にブレンドされる分滑らかだが頂点共有の構造的保証が弱まる、
# というトレードオフを確認する。VAEの学習・推論(生成)には一切組み込まれない、独立した事後診断用のスクリプト
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from vae_checkpoint_v2 import Checkpoint, load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_eval_common import draw_curves, existence_mask_from_logits
from vae_eval_common_v2 import (
    attract_to_latent_prior,
    encode_batch,
    load_batch,
    reconstructed_strokes_real,
    stroke_curves,
    to_real_scale,
)
from vae_model_v2 import CHECKPOINT_PATH, VAE, DecoderOutput, ModelShape, SlotAttentionConfig, select_device
from vae_training_v2 import GUMBEL_TEMPERATURE

SOFT_CHECKPOINT_PATH = CHECKPOINT_PATH.parent / "vae_v2_soft_pointer.pt"
MORPH_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]  # 検証用に固定した、訓練データ先頭8件からの4組
MORPH_STEPS = 8  # 各ペアの補間フレーム数
PLOT_PAIR_INDEX = 0  # 目視確認の描画に使うペア(MORPH_PAIRSのインデックス)。定量集計は全ペア対象


class MorphFrames(NamedTuple):
    decoder_output: DecoderOutput
    existence_mask: np.ndarray  # (MORPH_STEPS, stroke_count)


def _load_legacy_soft_checkpoint(device: torch.device) -> Checkpoint:
    # vae_v2_soft_pointer.ptは、Checkpoint形式にgumbel_temperatureが追加される前に退避したファイルのため、
    # vae_checkpoint_v2.load_checkpointではKeyErrorになる。この値は推論(eval)時には参照されない
    # (学習時のGumbel-Softmaxの温度にのみ使う)ため、ダミー値を補ってこの診断専用に読み込む
    # (本番のload_checkpointには手を入れない)
    raw = torch.load(SOFT_CHECKPOINT_PATH, map_location=device)
    shape = ModelShape(*raw["shape"])
    slot_attention_config = SlotAttentionConfig(*raw["slot_attention_config"])
    model = VAE(shape, raw["hidden_dims"], raw["latent_dim"], slot_attention_config, GUMBEL_TEMPERATURE).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return Checkpoint(
        model, shape, raw["hidden_dims"], slot_attention_config, GUMBEL_TEMPERATURE,
        raw["vertex_mean"].to(device), raw["vertex_std"].to(device),
        raw["stroke_offset_mean"].to(device), raw["stroke_offset_std"].to(device), raw["latent_dim"],
    )


def _interpolated_z(mu_a: torch.Tensor, mu_b: torch.Tensor, steps: int) -> torch.Tensor:
    t = torch.linspace(0, 1, steps, device=mu_a.device).unsqueeze(1)
    return mu_a.unsqueeze(0) * (1 - t) + mu_b.unsqueeze(0) * t


def _decode_morph(checkpoint: Checkpoint, mu_a: torch.Tensor, mu_b: torch.Tensor, mu_real: torch.Tensor) -> MorphFrames:
    z_interp = _interpolated_z(mu_a, mu_b, MORPH_STEPS)
    with torch.no_grad():
        z, _ = attract_to_latent_prior(z_interp, mu_real)
        decoder_output = checkpoint.model.decode(z)
    existence_mask = existence_mask_from_logits(decoder_output.stroke_existence_logits)
    return MorphFrames(decoder_output, existence_mask)


def _pointer_flip_count(logits: torch.Tensor, existence_mask: np.ndarray) -> int:
    # フレームt→t+1で、実在するストロークのうちポインタの選択先(argmax)が変わった回数の総和
    selected = logits.argmax(dim=-1)
    changed = (selected[1:] != selected[:-1]).cpu().numpy()
    active_both_frames = existence_mask[1:] & existence_mask[:-1]
    return int((changed & active_both_frames).sum())


def _max_point_jump(points_real: np.ndarray, existence_mask: np.ndarray) -> float:
    # フレームt→t+1での、実在するストロークの座標移動量(実スケール)の最大値
    jump = np.linalg.norm(points_real[1:] - points_real[:-1], axis=-1)
    active_both_frames = existence_mask[1:] & existence_mask[:-1]
    return float(jump[active_both_frames].max()) if active_both_frames.any() else 0.0


def _print_morph_stability(label: str, checkpoint: Checkpoint, mu: torch.Tensor, mu_real: torch.Tensor) -> None:
    start_flips = end_flips = 0
    start_jumps: list[float] = []
    end_jumps: list[float] = []
    for a, b in MORPH_PAIRS:
        frames = _decode_morph(checkpoint, mu[a], mu[b], mu_real)
        start_flips += _pointer_flip_count(frames.decoder_output.start_pointer_logits, frames.existence_mask)
        end_flips += _pointer_flip_count(frames.decoder_output.end_pointer_logits, frames.existence_mask)
        start_real = to_real_scale(frames.decoder_output.start_points, checkpoint.vertex_mean, checkpoint.vertex_std)
        end_real = to_real_scale(frames.decoder_output.end_points, checkpoint.vertex_mean, checkpoint.vertex_std)
        start_jumps.append(_max_point_jump(start_real, frames.existence_mask))
        end_jumps.append(_max_point_jump(end_real, frames.existence_mask))

    print(f"--- {label} ({len(MORPH_PAIRS)} pairs x {MORPH_STEPS} frames) ---")
    print(f"pointer flips (start/end, total over all consecutive frame pairs): {start_flips} / {end_flips}")
    print(f"max per-stroke point jump between consecutive frames (start/end, real scale): "
          f"{max(start_jumps):.3f} / {max(end_jumps):.3f}")
    print()


def _plot_morph(ax_row: np.ndarray, checkpoint: Checkpoint, mu: torch.Tensor, mu_real: torch.Tensor) -> None:
    a, b = MORPH_PAIRS[PLOT_PAIR_INDEX]
    frames = _decode_morph(checkpoint, mu[a], mu[b], mu_real)
    reconstructed = reconstructed_strokes_real(checkpoint, frames.decoder_output)
    for col in range(MORPH_STEPS):
        curves = stroke_curves(
            reconstructed.start[col], reconstructed.end[col], reconstructed.offsets[col], frames.existence_mask[col]
        )
        draw_curves(ax_row[col], curves)


def main() -> None:
    device = select_device()
    train_batch = load_batch(prepare_datasets(), "train", device)

    hard_checkpoint = load_checkpoint(device)
    soft_checkpoint = _load_legacy_soft_checkpoint(device)

    _, axes = plt.subplots(nrows=2, ncols=MORPH_STEPS, figsize=(MORPH_STEPS * 1.5, 3))
    for row, (label, checkpoint) in enumerate([("soft pointer", soft_checkpoint), ("hard pointer", hard_checkpoint)]):
        with torch.no_grad():
            mu, _ = encode_batch(checkpoint, train_batch)
        # 訓練データ全件のmuを、補間ペアの取得元(mu)・attract_to_latent_priorの引き寄せ先(mu_real)の
        # 両方に使う(本番の生成経路と同じ使い方)
        _print_morph_stability(label, checkpoint, mu, mu)
        _plot_morph(axes[row], checkpoint, mu, mu)
        axes[row, 0].set_ylabel(label)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
