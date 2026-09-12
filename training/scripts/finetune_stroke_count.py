#!/usr/bin/env python3
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader

from vae_checkpoint import load_checkpoint, save_checkpoint
from vae_data import SEED, prepare_datasets
from vae_eval_common import attract_to_latent_prior, existence_mask_from_logits
from vae_losses import (
    AngleGMM,
    build_angle_gmm,
    compute_finetune_stroke_count_loss,
    compute_loss,
    compute_synthetic_grammar_loss,
)
from vae_model import VAE, ModelShape, flatten_input, select_device, unflatten_output

BATCH_SIZE = 64
VAE_LR = 1e-4  # 収束済みチェックポイントの微調整のため、train_vae.pyの1e-3より小さくする
BETA = 1.0  # 微調整のみなのでKL annealingは行わない
EVAL_INTERVAL = 500  # このステップ数ごとに目標との乖離を測り、early stopping判定に使う
EVAL_SAMPLE_COUNT = 500  # 判定用サンプル数
PATIENCE = 6  # 目標乖離が改善しないままこの回数(×EVAL_INTERVALステップ)続いたら打ち切る。この損失は長時間続けると目標を追い越してオーバーシュートすることが分かっているため、train_vae.pyのval_lossベースearly stoppingと同じ枠組みで、目標に最も近づいた時点のモデルを採用する
MAX_STEPS = 20000  # early stoppingが正常なら到達しない安全上限。この値まで崩壊しないことは検証済み


class _BestCheckpointTracker:
    # 目標乖離が最小だった時点のモデルを保持し、それ以降PATIENCE回連続で改善しなければ
    # 停止すべきと判定する。update()の返り値がTrueになったら呼び出し側でループを打ち切る
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


def _train_step(
    model: VAE,
    shape: ModelShape,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    mean: torch.Tensor,
    std: torch.Tensor,
    angle_gmm: AngleGMM,
    target_pool_mu: torch.Tensor,
    target_pool_count: torch.Tensor,
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
    loss = loss + compute_finetune_stroke_count_loss(model, mu, shape, target_pool_mu, target_pool_count)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def _measure_stroke_count_mean(model: VAE, shape: ModelShape, mu_all: torch.Tensor) -> float:
    # report_generation_stats(検証用スクリプト)と同じ方法(事前分布からのサンプル+attract)で、
    # 母集団のストローク数平均だけを算出する
    model.eval()
    z_raw = torch.randn(EVAL_SAMPLE_COUNT, mu_all.shape[1], device=mu_all.device)
    with torch.no_grad():
        z = attract_to_latent_prior(z_raw, mu_all)
        recon = model.decode(z)
        _, existence_logits = unflatten_output(recon, shape)
        existence_pred = existence_mask_from_logits(existence_logits)
    return existence_pred.sum(axis=1).mean()


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
    with torch.no_grad():
        # 微調整開始時点で1回だけ計算し、以降固定して使う。_measure_stroke_count_meanに加え、
        # compute_finetune_stroke_count_loss内でのz_synthetic自体の引き寄せ先・目標選定用の
        # 近傍候補プール(いずれも実データ全件)としても使う。学習ミニバッチ(数十件)を候補にすると
        # 候補が少なすぎて目標値自体が偏り、また実際の生成時(attract_to_latent_prior)ともプール・
        # 帯域がずれてしまう(標準偏差の崩壊につながる)ことを確認済みのため、実データ全件で揃える。
        # 学習が進んでも再計算しない
        mu_all, _ = model.encode(flatten_input(train_strokes_standardized.to(device), train_existence))
    real_count_all = train_existence.sum(dim=1)

    target_mean = real_count_all.mean().item()
    print(f"target stroke count mean (train split average) = {target_mean:.2f}")

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
            mu_all,
            real_count_all,
            strokes_batch,
            existence_batch,
            connections_batch,
        )
        if step % EVAL_INTERVAL != 0:
            continue

        current_mean = _measure_stroke_count_mean(model, shape, mu_all)
        deviation = abs(current_mean - target_mean)
        print(f"step {step}: loss={loss.item():.2f} stroke_count_mean={current_mean:.2f} deviation={deviation:.3f}")
        if tracker.update(step, deviation, model):
            print(f"Early stopping at step {step} (patience={PATIENCE}), best_step={tracker.best_step}")
            break

    model.load_state_dict(tracker.best_state_dict)
    print(f"Restored best checkpoint from step {tracker.best_step} (deviation={tracker.best_deviation:.3f})")

    save_checkpoint(
        model, shape, checkpoint.hidden_dims, checkpoint.latent_dim, checkpoint.slot_attention_config, mean, std
    )
    print("Saved checkpoint")


if __name__ == "__main__":
    main()
