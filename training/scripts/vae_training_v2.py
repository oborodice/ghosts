#!/usr/bin/env python3
# 既定値での単発学習CLI・複数候補を比較するsweep CLIの両方から呼ばれる学習ロジック本体
from pathlib import Path

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
) -> LossComponents:
    # optimizerがNoneのとき(validation時)は重み更新を行わないeval modeとして扱う
    is_training = optimizer is not None
    model.train(is_training)

    totals = {field: 0.0 for field in LossComponents._fields}
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence = (
                t.to(device) for t in batch
            )

            x = flatten_input(
                vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence, shape
            )
            decoder_output, mu, logvar = model(x)
            loss = compute_loss(
                vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence,
                decoder_output, mu, logvar, beta, vertex_mean, vertex_std, angle_gmm,
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.total.backward()
                optimizer.step()

            batch_size = vertices.size(0)
            for field in LossComponents._fields:
                totals[field] += getattr(loss, field).item() * batch_size

    dataset_size = len(loader.dataset)
    return LossComponents(**{field: totals[field] / dataset_size for field in LossComponents._fields})


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

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        beta_epoch = _compute_beta(epoch, beta, kl_annealing_epochs)
        train_losses = _run_epoch(
            train_loader, model, datasets.shape, device, optimizer, beta_epoch, vertex_mean, vertex_std, angle_gmm
        )
        # 早期終了・チェックポイント選定はannealing中でも比較可能にするため、常に最終的なβ(=beta)で評価する
        val_losses = _run_epoch(
            val_loader, model, datasets.shape, device, None, beta, vertex_mean, vertex_std, angle_gmm
        )
        print(
            f"Epoch {epoch}: train_loss={train_losses.total:.4f} val_loss={val_losses.total:.4f} beta={beta_epoch:.4f} | "
            f"train breakdown: vertex={train_losses.vertex_loss:.4f} kl={train_losses.kl_divergence:.4f} "
            f"crossing={train_losses.crossing_loss:.4f} angle={train_losses.angle_naturalness_loss:.4f} "
            f"min_length={train_losses.min_length_loss:.4f} repulsion={train_losses.vertex_repulsion_loss:.4f}"
        )

        if val_losses.total < best_val_loss:
            best_val_loss = val_losses.total
            epochs_without_improvement = 0
            save_checkpoint(
                model, datasets.shape, hidden_dims, latent_dim, slot_attention_config, gumbel_temperature,
                vertex_mean, vertex_std, stroke_offset_mean, stroke_offset_std, checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    return best_val_loss
