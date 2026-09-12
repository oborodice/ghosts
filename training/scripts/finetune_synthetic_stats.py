#!/usr/bin/env python3
from collections.abc import Iterator
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

from vae_checkpoint import load_checkpoint, save_checkpoint
from vae_data import SEED, prepare_datasets
from vae_eval_common import attract_to_latent_prior, existence_mask_from_logits
from vae_losses import AngleGMM, build_angle_gmm, compute_loss
from vae_model import VAE, ModelShape, flatten_input, select_device, unflatten_output
from vae_synthetic_losses import (
    compute_finetune_length_loss,
    compute_finetune_stroke_count_loss,
    compute_synthetic_grammar_loss,
    masked_mean_std,
)

BATCH_SIZE = 64
VAE_LR = 1e-4  # 収束済みチェックポイントの微調整のため、train_vae.pyの1e-3より小さくする
BETA = 1.0  # 微調整のみなのでKL annealingは行わない
EVAL_INTERVAL = 500  # このステップ数ごとに目標との乖離を測り、early stopping判定に使う
EVAL_SAMPLE_COUNT = 500  # 判定用サンプル数
PATIENCE = 6  # 目標乖離が改善しないままこの回数(×EVAL_INTERVALステップ)続いたら打ち切る。ストローク数・
# ストローク長の損失はいずれも長時間続けると目標を追い越してオーバーシュートすることが分かっているため、
# train_vae.pyのval_lossベースearly stoppingと同じ枠組みで、目標に最も近づいた時点のモデルを採用する
MAX_STEPS = 20000  # early stoppingが正常なら到達しない安全上限。この値まで崩壊しないことは検証済み


class _BestCheckpointTracker:
    # 目標乖離(呼び出し側が用意する複合スコア)が最小だった時点のモデルを保持し、それ以降PATIENCE回連続で
    # 改善しなければ停止すべきと判定する。update()の返り値がTrueになったら呼び出し側でループを打ち切る
    def __init__(self, model: VAE) -> None:
        self.best_deviation = float("inf")
        self.best_step = 0
        self.best_state_dict = {key: value.clone() for key, value in model.state_dict().items()}
        self._evals_without_improvement = 0

    def update(self, step: int, deviation: float, model: VAE) -> bool:
        if deviation < self.best_deviation:
            self.best_deviation = deviation
            self.best_step = step
            self.best_state_dict = {key: value.clone() for key, value in model.state_dict().items()}
            self._evals_without_improvement = 0
            return False
        self._evals_without_improvement += 1
        return self._evals_without_improvement >= PATIENCE


def _step_batches(
    loader: DataLoader, max_steps: int
) -> Iterator[tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    # loaderを(必要なら複数周)繰り返し辿り、通し番号のステップ数(1始まり)付きでバッチを返す。
    # max_stepsに達したら、周の途中でも打ち切る
    step = 0
    while step < max_steps:
        for batch in loader:
            step += 1
            yield step, batch
            if step >= max_steps:
                return


class _Targets(NamedTuple):
    # このスクリプトが同時に微調整する統計量の目標値と、その算出に使った実データ全件のmuをまとめて保持する
    mu_all: torch.Tensor
    real_count_all: torch.Tensor
    length_mean: torch.Tensor
    length_std: torch.Tensor


def _train_step(
    model: VAE,
    shape: ModelShape,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    mean: torch.Tensor,
    std: torch.Tensor,
    angle_gmm: AngleGMM,
    targets: _Targets,
    strokes_batch: torch.Tensor,
    existence_batch: torch.Tensor,
    connections_batch: torch.Tensor,
) -> torch.Tensor:
    strokes_batch = strokes_batch.to(device)
    existence_batch = existence_batch.to(device)
    connections_batch = connections_batch.to(device)

    model.train()
    recon, mu, logvar = model(flatten_input(strokes_batch, existence_batch))
    strokes_recon, existence_logits = unflatten_output(recon, shape)

    loss = compute_loss(
        strokes_batch,
        existence_batch,
        strokes_recon,
        existence_logits,
        mu,
        logvar,
        connections_batch,
        angle_gmm,
        BETA,
        mean,
        std,
    )
    loss = loss + compute_synthetic_grammar_loss(model, mu, shape, mean, std, angle_gmm)
    loss = loss + compute_finetune_stroke_count_loss(model, mu, shape, targets.mu_all, targets.real_count_all)
    loss = loss + compute_finetune_length_loss(model, mu, shape, mean, std, targets.length_mean, targets.length_std)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def _measure_population_stats(
    model: VAE, shape: ModelShape, targets: _Targets, mean: torch.Tensor, std: torch.Tensor
) -> tuple[float, float]:
    # report_generation_stats(検証用スクリプト)と同じ方法(事前分布からのサンプル+attract)で、
    # 母集団のストローク数平均・ストローク長平均を同じサンプルから算出する
    model.eval()
    mu_all = targets.mu_all
    z_raw = torch.randn(EVAL_SAMPLE_COUNT, mu_all.shape[1], device=mu_all.device)
    with torch.no_grad():
        z = attract_to_latent_prior(z_raw, mu_all)
        recon = model.decode(z)
        strokes_recon, existence_logits = unflatten_output(recon, shape)
        existence_pred = existence_mask_from_logits(existence_logits)
        existence_pred_tensor = torch.from_numpy(existence_pred).float().to(mu_all.device)
        length = strokes_recon[..., 3] * std[3] + mean[3]
    stroke_count_mean = existence_pred.sum(axis=1).mean()
    length_mean, _, _ = masked_mean_std(length, existence_pred_tensor)
    return stroke_count_mean, length_mean.mean().item()


def _compute_targets(
    model: VAE, train_strokes_standardized: torch.Tensor, train_existence: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> _Targets:
    with torch.no_grad():
        # 微調整開始時点で1回だけ計算し、以降固定して使う。_measure_population_statsに加え、
        # compute_finetune_stroke_count_loss内でのz_synthetic自体の引き寄せ先・目標選定用の
        # 近傍候補プール(いずれも実データ全件)としても使う。学習ミニバッチ(数十件)を候補にすると
        # 候補が少なすぎて目標値自体が偏り、また実際の生成時(attract_to_latent_prior)ともプール・
        # 帯域がずれてしまう(標準偏差の崩壊につながる)ことを確認済みのため、実データ全件で揃える。
        # 学習が進んでも再計算しない
        mu_all, _ = model.encode(flatten_input(train_strokes_standardized, train_existence))
        train_strokes_real = train_strokes_standardized * std + mean
        length_mean, length_std, length_count = masked_mean_std(train_strokes_real[..., 3], train_existence)
        # ストローク数1以下の字は標準偏差が定義できない(常に0になる)ため、目標値の算出からは除外する
        length_valid = length_count >= 2
    real_count_all = train_existence.sum(dim=1)
    return _Targets(mu_all, real_count_all, length_mean[length_valid].mean(), length_std[length_valid].mean())


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    checkpoint = load_checkpoint(device)
    model = checkpoint.model
    shape = checkpoint.shape
    mean, std = checkpoint.mean, checkpoint.std

    datasets = prepare_datasets()
    train_loader = DataLoader(
        datasets.train, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED)
    )
    angle_gmm = build_angle_gmm(datasets.angle_gmm_params, device)

    train_strokes_standardized, train_existence, _ = datasets.train.tensors
    train_existence = train_existence.to(device)
    train_strokes_standardized = train_strokes_standardized.to(device)
    targets = _compute_targets(model, train_strokes_standardized, train_existence, mean, std)
    target_count = targets.real_count_all.mean().item()
    target_length = targets.length_mean.item()
    print(f"target stroke count mean (train split average) = {target_count:.2f}")
    print(f"target stroke length mean (train split average) = {target_length:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_LR)
    tracker = _BestCheckpointTracker(model)

    for step, (strokes_batch, existence_batch, connections_batch) in _step_batches(train_loader, MAX_STEPS):
        loss = _train_step(
            model,
            shape,
            device,
            optimizer,
            mean,
            std,
            angle_gmm,
            targets,
            strokes_batch,
            existence_batch,
            connections_batch,
        )
        if step % EVAL_INTERVAL != 0:
            continue

        current_count, current_length = _measure_population_stats(model, shape, targets, mean, std)
        # 単位の異なる2つの乖離を、目標値に対する相対誤差の2乗和という無次元の1つの複合スコアにまとめる
        count_relative_deviation = (current_count - target_count) / target_count
        length_relative_deviation = (current_length - target_length) / target_length
        deviation = count_relative_deviation**2 + length_relative_deviation**2
        print(
            f"step {step}: loss={loss.item():.2f} stroke_count_mean={current_count:.2f} "
            f"stroke_length_mean={current_length:.2f} deviation={deviation:.4f}"
        )
        if tracker.update(step, deviation, model):
            print(f"Early stopping at step {step} (patience={PATIENCE}), best_step={tracker.best_step}")
            break

    model.load_state_dict(tracker.best_state_dict)
    print(f"Restored best checkpoint from step {tracker.best_step} (deviation={tracker.best_deviation:.4f})")

    save_checkpoint(
        model, shape, checkpoint.hidden_dims, checkpoint.latent_dim, checkpoint.slot_attention_config, mean, std
    )
    print("Saved checkpoint")


if __name__ == "__main__":
    main()
