#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

import torch

from vae_data_v2 import SEED, prepare_datasets
from vae_model_v2 import CHECKPOINT_PATH, HIDDEN_DIMS, LATENT_DIM, select_device
from vae_resume_v2 import checkpoint_path_from_resume_state
from vae_training_v2 import BETA, GUMBEL_TEMPERATURE, KL_ANNEALING_EPOCHS, SLOT_ATTENTION_CONFIG, train


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a resume-state file (<checkpoint>_resume.pt) to continue an interrupted run",
    )
    return parser.parse_args()


def _timestamped_checkpoint_path() -> Path:
    # 固定パス(CHECKPOINT_PATH)へそのまま保存すると、実行のたびに既存のチェックポイントを
    # 上書きしてしまう。起動時刻をsuffixにしたパスをデフォルトにすることで、既存ファイルや
    # 複数回の実行(smoke test・再試行等)同士の衝突を両方避ける。ベース名はCHECKPOINT_PATHの
    # stem("vae_v2")をそのまま使わず固定の"vae"にする(頂点+辺構造とそれ以前の構造とで
    # チェックポイントの互換性は元々無く、ファイル名で区別する必要がない)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CHECKPOINT_PATH.with_name(f"vae_{timestamp}{CHECKPOINT_PATH.suffix}")


def main() -> None:
    args = _parse_args()

    torch.manual_seed(SEED)
    device = select_device()
    print(f"Using device: {device}")

    datasets = prepare_datasets()
    checkpoint_path = (
        checkpoint_path_from_resume_state(args.resume) if args.resume is not None else _timestamped_checkpoint_path()
    )
    best_val_loss = train(
        datasets,
        device,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM,
        slot_attention_config=SLOT_ATTENTION_CONFIG,
        gumbel_temperature=GUMBEL_TEMPERATURE,
        beta=BETA,
        kl_annealing_epochs=KL_ANNEALING_EPOCHS,
        checkpoint_path=checkpoint_path,
        resume_from=args.resume,
    )

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
