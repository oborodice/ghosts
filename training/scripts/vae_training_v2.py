#!/usr/bin/env python3
# 既定値での単発学習CLI・複数候補を比較するsweep CLIの両方から呼ばれる学習ロジック本体
from pathlib import Path
from typing import Callable, NamedTuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from vae_checkpoint_v2 import save_checkpoint
from vae_data_v2 import SEED, Datasets
from vae_losses import AngleGMM, build_angle_gmm
from vae_losses_v2 import LossComponents, compute_loss
from vae_model_v2 import (
    SLOT_ATTENTION_FFN_DIM,
    SLOT_ATTENTION_HEADS,
    SLOT_ATTENTION_LAYERS,
    SLOT_DIM,
    VAE,
    ModelShape,
    SlotAttentionConfig,
    flatten_input,
)
from vae_resume_v2 import ResumeState, load_resume_state, resume_state_path, save_resume_state
from vae_synthetic_losses_v2 import SyntheticLossComponents, compute_synthetic_loss

SLOT_ATTENTION_CONFIG = SlotAttentionConfig(
    SLOT_DIM, SLOT_ATTENTION_HEADS, SLOT_ATTENTION_LAYERS, SLOT_ATTENTION_FFN_DIM
)

BETA = 0.1  # posterior collapse(潜在次元の大部分が死んで再構成精度が落ちる現象)を避けるため、
# 複数候補を比較して選んだ暫定値。容量・repulsion・βの組み合わせsweepで、この値がrepulsionの
# 高い方の水準と組み合わさったときに他の7候補を上回ることを確認済み(正式な最適値探しは今後別途行う)
KL_ANNEALING_EPOCHS = 60  # このepoch数をかけてβを0からBETAまで線形に引き上げる(warm-up)
GUMBEL_TEMPERATURE = 1.0  # Straight-Through Gumbel-Softmaxの温度。標準的な既定値を暫定採用(正式な調整は今後別途行う)
LEARNING_RATE = 2e-4  # パラメータ数が数倍に増えたモデルに対して、既存コードから引き継いだ1e-3のままでは
# 大きすぎ、ポインタ機構(始点・終点の頂点選択)の学習を明確に悪化させることを実測で確認済み。
# 軽量な継続学習での検証でポインタ分類精度・crossings・angle_naturalness・isolated_stroke_rateが
# いずれも回復した値を採用する
BATCH_SIZE = 64
PATIENCE = 20
MAX_EPOCHS = 1000  # early stoppingが正常なら到達しない安全上限


class LossContext(NamedTuple):
    # 損失計算に必要な、データセットから一度だけ導出される統計量一式(train/val・epochを通じて不変)。
    # 新しい合成z側損失を追加するたびに`_forward_and_compute_loss`等の引数リストが伸び続けるのを
    # 避けるため、1つのまとまりとして持ち回す
    vertex_mean: torch.Tensor
    vertex_std: torch.Tensor
    stroke_offset_mean: torch.Tensor
    stroke_offset_std: torch.Tensor
    angle_gmm: AngleGMM
    target_crossings_mean: float
    target_crossings_std: float


def _build_loss_context(datasets: Datasets, device: torch.device) -> LossContext:
    return LossContext(
        vertex_mean=torch.tensor(datasets.vertex_mean, dtype=torch.float32, device=device),
        vertex_std=torch.tensor(datasets.vertex_std, dtype=torch.float32, device=device),
        stroke_offset_mean=torch.tensor(datasets.stroke_offset_mean, dtype=torch.float32, device=device),
        stroke_offset_std=torch.tensor(datasets.stroke_offset_std, dtype=torch.float32, device=device),
        angle_gmm=build_angle_gmm(datasets.angle_gmm_params, device),
        target_crossings_mean=datasets.target_crossings_mean,
        target_crossings_std=datasets.target_crossings_std,
    )


def _compute_beta(epoch: int, beta: float, kl_annealing_epochs: int) -> float:
    # 学習序盤にKL項がフルに効くと一部の潜在次元が使われなくなる(posterior collapse)ため、線形にwarm-upする
    return beta * min(1.0, epoch / kl_annealing_epochs)


def _encode_train_mu_pool(model: VAE, datasets: Datasets, shape: ModelShape, device: torch.device) -> torch.Tensor:
    # 訓練データ全体を一度にエンコードし、合成z側の損失が引き寄せ先として使うmuの候補プールを作る。
    # 本番の生成(vae_eval_common.attract_to_latent_prior)が訓練データ全体を候補にするのと揃えるため。
    # 毎ステップ全データをエンコードするコストを避けるため、エポック単位(train呼び出し元で1回)で
    # キャッシュして使い回す近似にする(エポック内でのmuの変化は小さいと想定)。ここで計算した値は
    # 呼び出し元に戻さず引き寄せ先としてのみ使うため勾配は不要
    with torch.no_grad():
        tensors = tuple(t.to(device) for t in datasets.train.tensors)
        x = flatten_input(*tensors, shape)
        mu_pool, _ = model.encode(x)
    return mu_pool


def _forward_and_compute_loss(
    model: VAE,
    batch: tuple[torch.Tensor, ...],
    shape: ModelShape,
    device: torch.device,
    beta: float,
    ctx: LossContext,
    mu_pool: torch.Tensor,
) -> tuple[torch.Tensor, LossComponents, SyntheticLossComponents]:
    # 1バッチ分のforward計算と、再構成側(LossComponents)・混ぜ合わせz側(SyntheticLossComponents)
    # 両方の損失計算をまとめる。学習対象の合計は両者のtotalの和
    vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence = (
        t.to(device) for t in batch
    )
    x = flatten_input(vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence, shape)
    decoder_output, mu, logvar = model(x)
    recon_loss = compute_loss(
        vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence,
        decoder_output, mu, logvar, beta, ctx.vertex_mean, ctx.vertex_std,
        ctx.stroke_offset_mean, ctx.stroke_offset_std, ctx.angle_gmm,
    )
    synthetic_loss = compute_synthetic_loss(
        model, mu, mu_pool, ctx.vertex_std, ctx.stroke_offset_mean, ctx.stroke_offset_std,
        ctx.target_crossings_mean, ctx.target_crossings_std,
    )
    total = recon_loss.total + synthetic_loss.total
    return total, recon_loss, synthetic_loss


def _accumulate_losses(
    totals: dict[str, float], loss: LossComponents | SyntheticLossComponents, batch_size: int
) -> None:
    # LossComponents・SyntheticLossComponentsのどちらも「内訳をtotals辞書に集計する」処理は同じ形なので、
    # 呼び出し元(_run_epoch)で2回書き分けずにここへ共通化する。損失源が増えても呼び出しを1行足すだけで済む
    for field in loss._fields:
        totals[field] += getattr(loss, field).item() * batch_size


def _run_epoch(
    loader: DataLoader,
    model: VAE,
    shape: ModelShape,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    beta: float,
    ctx: LossContext,
    mu_pool: torch.Tensor,
) -> tuple[LossComponents, SyntheticLossComponents]:
    # optimizerがNoneのとき(validation時)は重み更新を行わないeval modeとして扱う
    is_training = optimizer is not None
    model.train(is_training)

    recon_totals = {field: 0.0 for field in LossComponents._fields}
    synthetic_totals = {field: 0.0 for field in SyntheticLossComponents._fields}
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            total, recon_loss, synthetic_loss = _forward_and_compute_loss(
                model, batch, shape, device, beta, ctx, mu_pool
            )

            if optimizer is not None:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            batch_size = batch[0].size(0)
            _accumulate_losses(recon_totals, recon_loss, batch_size)
            _accumulate_losses(synthetic_totals, synthetic_loss, batch_size)

    dataset_size = len(loader.dataset)
    recon_losses = LossComponents(**{field: recon_totals[field] / dataset_size for field in LossComponents._fields})
    synthetic_losses = SyntheticLossComponents(
        **{field: synthetic_totals[field] / dataset_size for field in SyntheticLossComponents._fields}
    )
    return recon_losses, synthetic_losses


class _TrainingState(NamedTuple):
    # trainのループ本体が必要とするもの一式。DataLoader構築・モデル構築・標準化定数の算出という、
    # ループの反復とは別の関心事(1回だけ行うセットアップ)をtrainから切り離すためのまとまり
    model: VAE
    optimizer: optim.Optimizer
    train_loader: DataLoader
    val_loader: DataLoader
    ctx: LossContext


def _assert_resume_matches_hyperparameters(
    resume_state: ResumeState,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
    gumbel_temperature: float,
) -> None:
    # 呼び出し元が渡したハイパーパラメータが、resume_fromに実際に保存されていたものと食い違ったまま
    # 進むと、以後保存するチェックポイント・再開状態のメタデータが実際のモデル構造とズレてしまう。
    # 数時間かけた学習の終盤でこれに気づくことがないよう、再開直後に検証して早期に失敗させる。
    # SlotAttentionConfigはNamedTuple(=tupleのサブクラス)なので、値が同じであれば要素ごとの
    # tuple変換なしでもそのまま比較できる
    expected = (hidden_dims, latent_dim, slot_attention_config, gumbel_temperature)
    actual = (
        resume_state.hidden_dims, resume_state.latent_dim,
        resume_state.slot_attention_config, resume_state.gumbel_temperature,
    )
    assert expected == actual, f"hyperparameters saved in resume_from {actual} do not match the caller's {expected}"


def _build_loaders(datasets: Datasets, batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        datasets.train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(datasets.val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _build_training_state(
    datasets: Datasets,
    device: torch.device,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
    gumbel_temperature: float,
    learning_rate: float,
    batch_size: int,
) -> _TrainingState:
    train_loader, val_loader = _build_loaders(datasets, batch_size)
    model = VAE(datasets.shape, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ctx = _build_loss_context(datasets, device)
    return _TrainingState(model, optimizer, train_loader, val_loader, ctx)


class _RunInit(NamedTuple):
    state: _TrainingState
    start_epoch: int
    best_val_loss: float
    epochs_without_improvement: int


def _initialize_run(
    datasets: Datasets,
    device: torch.device,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
    gumbel_temperature: float,
    learning_rate: float,
    batch_size: int,
    resume_from: Path | None,
) -> _RunInit:
    # resume_fromが指定された場合、そこに保存された学習状態(モデル重み・optimizer状態・epoch数)
    # から続きを学習する。長時間のフルスケール学習が途中で落ちた場合に、ゼロからのやり直しを避けるため
    if resume_from is not None:
        resume_state = load_resume_state(datasets.shape, device, learning_rate, resume_from)
        _assert_resume_matches_hyperparameters(
            resume_state, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature
        )
        train_loader, val_loader = _build_loaders(datasets, batch_size)
        ctx = _build_loss_context(datasets, device)
        state = _TrainingState(resume_state.model, resume_state.optimizer, train_loader, val_loader, ctx)
        start_epoch = resume_state.epoch + 1
        print(f"Resumed from {resume_from} at epoch {start_epoch} (best_val_loss={resume_state.best_val_loss:.4f})")
        return _RunInit(state, start_epoch, resume_state.best_val_loss, resume_state.epochs_without_improvement)

    state = _build_training_state(
        datasets, device, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature, learning_rate, batch_size
    )
    return _RunInit(state, 1, float("inf"), 0)


def _format_loss_breakdown(*components: LossComponents | SyntheticLossComponents) -> str:
    # 各componentsの_fieldsを列挙して組み立てる(totalは呼び出し元で別途表示済みのため除く)。
    # 損失を追加・削除してもここを手で編集する必要がない
    return " ".join(
        f"{name}={getattr(component, name):.4f}"
        for component in components
        for name in component._fields
        if name != "total"
    )


def train(
    datasets: Datasets,
    device: torch.device,
    *,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
    gumbel_temperature: float,
    beta: float,
    kl_annealing_epochs: int,
    checkpoint_path: Path,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
    max_epochs: int = MAX_EPOCHS,
    on_epoch_end: Callable[[int, VAE, LossContext], None] | None = None,
    resume_from: Path | None = None,
) -> float:
    # ハイパーパラメータを引数として受け取ることで、既定値での単発学習・候補ごとの比較学習(sweep)の
    # 両方が同じ学習ロジックを呼び出せるようにしている。戻り値はearly stopping時点のbest validation loss。
    # on_epoch_endは、long runの安全網としてbest val loss更新とは無関係にエポックのスナップショットを
    # 残したい呼び出し元向けのオプションのフック(既定Noneなら本番の挙動に一切影響しない)。
    state, start_epoch, best_val_loss, epochs_without_improvement = _initialize_run(
        datasets, device, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature,
        learning_rate, batch_size, resume_from,
    )
    resume_path = resume_state_path(checkpoint_path)

    for epoch in range(start_epoch, max_epochs + 1):
        beta_epoch = _compute_beta(epoch, beta, kl_annealing_epochs)
        # 合成z側の損失が引き寄せ先として使うmuの候補プールをエポック単位でキャッシュする。
        # train・val両方のこのエポック中の合成損失計算で使い回す(val側もわずかに古いmu_poolになるが、
        # 毎エポック1回の全データエンコードで十分という前提は_encode_train_mu_pool参照)
        mu_pool = _encode_train_mu_pool(state.model, datasets, datasets.shape, device)
        train_losses, train_synthetic = _run_epoch(
            state.train_loader, state.model, datasets.shape, device, state.optimizer, beta_epoch, state.ctx, mu_pool
        )
        # 早期終了・チェックポイント選定はannealing中でも比較可能にするため、常に最終的なβ(=beta)で評価する
        val_losses, val_synthetic = _run_epoch(
            state.val_loader, state.model, datasets.shape, device, None, beta, state.ctx, mu_pool
        )
        # 学習対象・early stopping判定に使う実際の合計は、再構成側・混ぜ合わせz側それぞれのtotalの和
        train_total = train_losses.total + train_synthetic.total
        val_total = val_losses.total + val_synthetic.total
        print(
            f"Epoch {epoch}: train_loss={train_total:.4f} val_loss={val_total:.4f} beta={beta_epoch:.4f} | "
            f"train breakdown: {_format_loss_breakdown(train_losses, train_synthetic)}"
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_without_improvement = 0
            save_checkpoint(
                state.model, datasets.shape, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature,
                state.ctx.vertex_mean, state.ctx.vertex_std, state.ctx.stroke_offset_mean, state.ctx.stroke_offset_std,
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        save_resume_state(
            state.model, state.optimizer, epoch, best_val_loss, epochs_without_improvement,
            hidden_dims, latent_dim, slot_attention_config, gumbel_temperature, resume_path,
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

        if on_epoch_end is not None:
            on_epoch_end(epoch, state.model, state.ctx)

    return best_val_loss
