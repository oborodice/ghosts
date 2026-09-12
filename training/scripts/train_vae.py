#!/usr/bin/env python3
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from vae_checkpoint import save_checkpoint
from vae_data import SEED, prepare_datasets
from vae_losses import AngleGMM, build_angle_gmm, compute_loss
from vae_model import (
    CHECKPOINT_PATH,
    HIDDEN_DIMS,
    LATENT_DIM,
    SLOT_ATTENTION_FFN_DIM,
    SLOT_ATTENTION_HEADS,
    SLOT_ATTENTION_LAYERS,
    SLOT_DIM,
    VAE,
    ModelShape,
    SlotAttentionConfig,
    flatten_input,
    select_device,
    unflatten_output,
)
from vae_synthetic_losses import compute_synthetic_grammar_loss

SLOT_ATTENTION_CONFIG = SlotAttentionConfig(
    SLOT_DIM, SLOT_ATTENTION_HEADS, SLOT_ATTENTION_LAYERS, SLOT_ATTENTION_FFN_DIM
)

BETA = 1.0
KL_ANNEALING_EPOCHS = 60  # このepoch数をかけてβを0からBETAまで線形に引き上げる(warm-up)。30から60への延長でdead dimensionsが減ることを検証済み
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
PATIENCE = 20
MAX_EPOCHS = 1000  # early stoppingが正常なら到達しない安全上限


def _compute_beta(epoch: int) -> float:
    # 学習序盤にKL項がフルに効くと一部の潜在次元が使われなくなる(posterior collapse)ため、
    # KL_ANNEALING_EPOCHSかけてβを0からBETAまで線形に引き上げる
    return BETA * min(1.0, epoch / KL_ANNEALING_EPOCHS)


def _run_epoch(
    loader: DataLoader,
    model: VAE,
    shape: ModelShape,
    device: torch.device,
    optimizer: optim.Optimizer | None,
    beta: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    angle_gmm: AngleGMM,
) -> float:
    # optimizerがNoneのとき(validation時)は重み更新を行わないeval modeとして扱う
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    with torch.set_grad_enabled(is_training):
        for strokes_batch, existence_batch, connections_batch in loader:
            strokes_batch = strokes_batch.to(device)
            existence_batch = existence_batch.to(device)
            connections_batch = connections_batch.to(device)
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
                beta,
                mean,
                std,
            )
            loss = loss + compute_synthetic_grammar_loss(model, mu, shape, mean, std, angle_gmm)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * strokes_batch.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    print(f"Using device: {device}")

    datasets = prepare_datasets()
    train_loader = DataLoader(
        datasets.train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(datasets.val, batch_size=BATCH_SIZE, shuffle=False)

    model = VAE(datasets.shape, HIDDEN_DIMS, LATENT_DIM, SLOT_ATTENTION_CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mean_tensor = torch.tensor(datasets.mean, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(datasets.std, dtype=torch.float32, device=device)
    angle_gmm = build_angle_gmm(datasets.angle_gmm_params, device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        beta = _compute_beta(epoch)
        train_loss = _run_epoch(
            train_loader, model, datasets.shape, device, optimizer, beta, mean_tensor, std_tensor, angle_gmm
        )
        # 早期終了・チェックポイント選定はannealing中でも比較可能にするため、常に最終的なβ(=BETA)で評価する
        val_loss = _run_epoch(
            val_loader, model, datasets.shape, device, None, BETA, mean_tensor, std_tensor, angle_gmm
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} beta={beta:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model, datasets.shape, HIDDEN_DIMS, LATENT_DIM, SLOT_ATTENTION_CONFIG, mean_tensor, std_tensor
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
