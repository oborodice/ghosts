#!/usr/bin/env python3
from collections.abc import Iterator
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

from vae_checkpoint import load_checkpoint, save_checkpoint
from vae_data import SEED, prepare_datasets
from vae_eval_common import attract_to_latent_prior, crossings_and_triple_junctions, existence_mask_from_logits
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
EVAL_INTERVAL = 100  # このステップ数ごとに目標との乖離を測り、early stopping判定に使う。500だった頃は
# 4軸拡張後の初回検証(サニティチェック)で、実際にはステップ100時点(deviation=1.245)の方がステップ500
# (本番実行で採用された値、deviation=2.25)より良かったにも関わらず、粒度が粗く見逃していたことが判明した
# ため、100に細かくした
EVAL_SAMPLE_COUNT = 500  # 判定用サンプル数
PATIENCE = 30  # 目標乖離が改善しないままこの回数(×EVAL_INTERVALステップ)続いたら打ち切る。EVAL_INTERVALを
# 500→100(5分の1)にした際、猶予するステップ数の実質量(PATIENCE×EVAL_INTERVAL=3000ステップ)は変えない
# よう、PATIENCEを6→30(5倍)にして揃えた。ストローク数・ストローク長の損失はいずれも長時間続けると
# 目標を追い越してオーバーシュートすることが分かっているため、train_vae.pyのval_lossベースearly stopping
# と同じ枠組みで、目標に最も近づいた時点のモデルを採用する
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
    # このスクリプトが同時に微調整する統計量の目標値と、その算出に使った実データ全件のmuをまとめて保持する。
    # crossings_std・triple_junction_stdは、目標値(平均)がゼロに近く相対誤差が発散するため、
    # 目標に対する相対誤差ではなく標準偏差基準(z-score的な発想)で乖離を測るために使う
    mu_all: torch.Tensor
    real_count_all: torch.Tensor
    length_mean: torch.Tensor
    length_std: torch.Tensor
    crossings_mean: float
    crossings_std: float
    triple_junction_mean: float
    triple_junction_std: float


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


class _PopulationStats(NamedTuple):
    stroke_count_mean: float
    length_mean: float
    crossings_mean: float
    triple_junction_mean: float


def _measure_population_stats(
    model: VAE, shape: ModelShape, targets: _Targets, mean: torch.Tensor, std: torch.Tensor
) -> _PopulationStats:
    # report_generation_stats(検証用スクリプト)と同じ方法(事前分布からのサンプル+attract)で、
    # 母集団のストローク数・長さ・交差数・3本以上合流の平均を同じサンプルから算出する
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
        strokes_recon_real = (strokes_recon * std + mean).cpu().numpy()
    stroke_count_mean = existence_pred.sum(axis=1).mean()
    length_mean, _, _ = masked_mean_std(length, existence_pred_tensor)
    crossings_per_char, _, triple_junction_per_char = crossings_and_triple_junctions(strokes_recon_real, existence_pred)
    return _PopulationStats(
        stroke_count_mean, length_mean.mean().item(), crossings_per_char.mean(), triple_junction_per_char.mean()
    )


def _relative_deviation(current: float, target: float) -> float:
    return (current - target) / target


def _zscore_deviation(current: float, target_mean: float, target_std: float) -> float:
    return (current - target_mean) / target_std


def _compute_deviation(stats: _PopulationStats, targets: _Targets, target_count: float, target_length: float) -> float:
    # 単位の異なる4つの乖離を、無次元の1つの複合スコアにまとめる。ストローク数・長さは目標値に対する
    # 相対誤差(値そのものが14・33程度で十分大きいため、この正規化が素直に機能する)。交差数・
    # 3本以上合流は目標値(平均)がゼロに近く、同じ相対誤差の定義だと少しの絶対誤差でも比率が
    # 発散してしまうため、代わりに実データの標準偏差を基準にした乖離(z-score的な発想)を使う。
    # 交差数・3本以上合流を追加したのは、ストローク長の微調整がこの2軸を悪化させる副作用を持つと
    # 判明したため。ストローク数・長さだけを見て早期終了すると、この副作用に気づかないまま
    # 「ストローク数・長さだけは良い」チェックポイントを採用してしまう
    count_deviation = _relative_deviation(stats.stroke_count_mean, target_count)
    length_deviation = _relative_deviation(stats.length_mean, target_length)
    crossings_deviation = _zscore_deviation(stats.crossings_mean, targets.crossings_mean, targets.crossings_std)
    triple_junction_deviation = _zscore_deviation(
        stats.triple_junction_mean, targets.triple_junction_mean, targets.triple_junction_std
    )
    return count_deviation**2 + length_deviation**2 + crossings_deviation**2 + triple_junction_deviation**2


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
        crossings_per_char, _, triple_junction_per_char = crossings_and_triple_junctions(
            train_strokes_real.cpu().numpy(), train_existence.cpu().numpy()
        )
    real_count_all = train_existence.sum(dim=1)
    return _Targets(
        mu_all,
        real_count_all,
        length_mean[length_valid].mean(),
        length_std[length_valid].mean(),
        crossings_per_char.mean(),
        crossings_per_char.std(),
        triple_junction_per_char.mean(),
        triple_junction_per_char.std(),
    )


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
    print(f"target crossings mean(std) = {targets.crossings_mean:.3f}({targets.crossings_std:.3f})")
    print(f"target triple_junction mean(std) = {targets.triple_junction_mean:.3f}({targets.triple_junction_std:.3f})")

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

        stats = _measure_population_stats(model, shape, targets, mean, std)
        deviation = _compute_deviation(stats, targets, target_count, target_length)
        print(
            f"step {step}: loss={loss.item():.2f} stroke_count_mean={stats.stroke_count_mean:.2f} "
            f"stroke_length_mean={stats.length_mean:.2f} crossings_mean={stats.crossings_mean:.3f} "
            f"triple_junction_mean={stats.triple_junction_mean:.3f} deviation={deviation:.4f}"
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
