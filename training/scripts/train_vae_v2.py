#!/usr/bin/env python3
import torch

from vae_data_v2 import SEED, prepare_datasets
from vae_model_v2 import CHECKPOINT_PATH, HIDDEN_DIMS, LATENT_DIM, select_device
from vae_training_v2 import BETA, KL_ANNEALING_EPOCHS, SLOT_ATTENTION_CONFIG, train


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    print(f"Using device: {device}")

    datasets = prepare_datasets()
    best_val_loss = train(
        datasets,
        device,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        slot_attention_config=SLOT_ATTENTION_CONFIG,
        beta=BETA,
        kl_annealing_epochs=KL_ANNEALING_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
