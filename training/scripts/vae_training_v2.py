#!/usr/bin/env python3
# 既定値での単発学習CLI・複数候補を比較するsweep CLIの両方から呼ばれる学習ロジック本体
from pathlib import Path
from typing import NamedTuple

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
from vae_synthetic_losses_v2 import SyntheticLossComponents, compute_synthetic_loss

SLOT_ATTENTION_CONFIG = SlotAttentionConfig(
    SLOT_DIM, SLOT_ATTENTION_HEADS, SLOT_ATTENTION_LAYERS, SLOT_ATTENTION_FFN_DIM
)

BETA = 0.25  # posterior collapse(潜在次元の大部分が死んで再構成精度が落ちる現象)を避けるため、
# 複数候補を比較して選んだ暫定値(正式な最適値探しは今後別途行う)
KL_ANNEALING_EPOCHS = 60  # このepoch数をかけてβを0からBETAまで線形に引き上げる(warm-up)
GUMBEL_TEMPERATURE = 1.0  # Straight-Through Gumbel-Softmaxの温度。標準的な既定値を暫定採用(正式な調整は今後別途行う)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
PATIENCE = 20
MAX_EPOCHS = 1000  # early stoppingが正常なら到達しない安全上限


def _compute_beta(epoch: int, beta: float, kl_annealing_epochs: int) -> float:
    # 学習序盤にKL項がフルに効くと一部の潜在次元が使われなくなる(posterior collapse)ため、線形にwarm-upする
    return beta * min(1.0, epoch / kl_annealing_epochs)


def _forward_and_compute_loss(
    model: VAE,
    batch: tuple[torch.Tensor, ...],
    shape: ModelShape,
    device: torch.device,
    beta: float,
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
    angle_gmm: AngleGMM,
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
        decoder_output, mu, logvar, beta, vertex_mean, vertex_std, angle_gmm,
    )
    synthetic_loss = compute_synthetic_loss(model, mu)
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
    vertex_mean: torch.Tensor,
    vertex_std: torch.Tensor,
    angle_gmm: AngleGMM,
) -> tuple[LossComponents, SyntheticLossComponents]:
    # optimizerがNoneのとき(validation時)は重み更新を行わないeval modeとして扱う
    is_training = optimizer is not None
    model.train(is_training)

    recon_totals = {field: 0.0 for field in LossComponents._fields}
    synthetic_totals = {field: 0.0 for field in SyntheticLossComponents._fields}
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            total, recon_loss, synthetic_loss = _forward_and_compute_loss(
                model, batch, shape, device, beta, vertex_mean, vertex_std, angle_gmm
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
    vertex_mean: torch.Tensor
    vertex_std: torch.Tensor
    stroke_offset_mean: torch.Tensor
    stroke_offset_std: torch.Tensor
    angle_gmm: AngleGMM


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
    train_loader = DataLoader(
        datasets.train,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(datasets.val, batch_size=batch_size, shuffle=False)

    model = VAE(datasets.shape, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    vertex_mean = torch.tensor(datasets.vertex_mean, dtype=torch.float32, device=device)
    vertex_std = torch.tensor(datasets.vertex_std, dtype=torch.float32, device=device)
    stroke_offset_mean = torch.tensor(datasets.stroke_offset_mean, dtype=torch.float32, device=device)
    stroke_offset_std = torch.tensor(datasets.stroke_offset_std, dtype=torch.float32, device=device)
    angle_gmm = build_angle_gmm(datasets.angle_gmm_params, device)
    return _TrainingState(
        model, optimizer, train_loader, val_loader,
        vertex_mean, vertex_std, stroke_offset_mean, stroke_offset_std, angle_gmm,
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
) -> float:
    # ハイパーパラメータを引数として受け取ることで、既定値での単発学習・候補ごとの比較学習(sweep)の
    # 両方が同じ学習ロジックを呼び出せるようにしている。戻り値はearly stopping時点のbest validation loss
    state = _build_training_state(
        datasets, device, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature,
        learning_rate, batch_size,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        beta_epoch = _compute_beta(epoch, beta, kl_annealing_epochs)
        train_losses, train_synthetic = _run_epoch(
            state.train_loader, state.model, datasets.shape, device, state.optimizer, beta_epoch,
            state.vertex_mean, state.vertex_std, state.angle_gmm,
        )
        # 早期終了・チェックポイント選定はannealing中でも比較可能にするため、常に最終的なβ(=beta)で評価する
        val_losses, val_synthetic = _run_epoch(
            state.val_loader, state.model, datasets.shape, device, None, beta,
            state.vertex_mean, state.vertex_std, state.angle_gmm,
        )
        # 学習対象・early stopping判定に使う実際の合計は、再構成側・混ぜ合わせz側それぞれのtotalの和
        train_total = train_losses.total + train_synthetic.total
        val_total = val_losses.total + val_synthetic.total
        print(
            f"Epoch {epoch}: train_loss={train_total:.4f} val_loss={val_total:.4f} beta={beta_epoch:.4f} | "
            f"train breakdown: vertex={train_losses.vertex_loss:.4f} kl={train_losses.kl_divergence:.4f} "
            f"crossing={train_losses.crossing_loss:.4f} angle={train_losses.angle_naturalness_loss:.4f} "
            f"min_length={train_losses.min_length_loss:.4f} repulsion={train_losses.vertex_repulsion_loss:.4f} "
            f"self_loop={train_synthetic.self_loop_loss:.4f}"
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_without_improvement = 0
            save_checkpoint(
                state.model, datasets.shape, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature,
                state.vertex_mean, state.vertex_std, state.stroke_offset_mean, state.stroke_offset_std,
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    return best_val_loss
