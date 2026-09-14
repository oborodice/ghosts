#!/usr/bin/env python3
# 既定値での単発学習CLI・複数候補を比較するsweep CLIの両方から呼ばれる学習ロジック本体
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from vae_checkpoint_v2 import save_checkpoint
from vae_data_v2 import SEED, Datasets
from vae_losses_v2 import compute_loss
from vae_model_v2 import (
    SLOT_ATTENTION_FFN_DIM,
    SLOT_ATTENTION_HEADS,
    SLOT_ATTENTION_LAYERS,
    SLOT_DIM,
    VAE,
    ModelShape,
    SlotAttentionConfig,
    flatten_input,
    unflatten_output,
)

SLOT_ATTENTION_CONFIG = SlotAttentionConfig(
    SLOT_DIM, SLOT_ATTENTION_HEADS, SLOT_ATTENTION_LAYERS, SLOT_ATTENTION_FFN_DIM
)

BETA = 0.25  # posterior collapse(潜在次元の大部分が死んで再構成精度が落ちる現象)を避けるため、
# 複数候補を比較して選んだ暫定値(正式な最適値探しは今後別途行う)
KL_ANNEALING_EPOCHS = 60  # このepoch数をかけてβを0からBETAまで線形に引き上げる(warm-up)
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
) -> float:
    # optimizerがNoneのとき(validation時)は重み更新を行わないeval modeとして扱う
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    with torch.set_grad_enabled(is_training):
        for vertices_batch, existence_batch in loader:
            vertices_batch = vertices_batch.to(device)
            existence_batch = existence_batch.to(device)
            recon, mu, logvar = model(flatten_input(vertices_batch, existence_batch))
            vertices_recon, existence_logits = unflatten_output(recon, shape)
            loss = compute_loss(vertices_batch, existence_batch, vertices_recon, existence_logits, mu, logvar, beta)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * vertices_batch.size(0)
    return total_loss / len(loader.dataset)


def train(
    datasets: Datasets,
    device: torch.device,
    *,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
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

    model = VAE(datasets.shape, hidden_dims, latent_dim, slot_attention_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mean_tensor = torch.tensor(datasets.mean, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(datasets.std, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        beta_epoch = _compute_beta(epoch, beta, kl_annealing_epochs)
        train_loss = _run_epoch(train_loader, model, datasets.shape, device, optimizer, beta_epoch)
        # 早期終了・チェックポイント選定はannealing中でも比較可能にするため、常に最終的なβ(=beta)で評価する
        val_loss = _run_epoch(val_loader, model, datasets.shape, device, None, beta)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} beta={beta_epoch:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model, datasets.shape, hidden_dims, latent_dim, slot_attention_config,
                mean_tensor, std_tensor, checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    return best_val_loss
