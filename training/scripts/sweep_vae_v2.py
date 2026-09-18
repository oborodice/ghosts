#!/usr/bin/env python3
# train()が返すbest validation loss(betaで重み付けされた合成損失)は、候補ごとにbetaが異なると
# 直接比較できない(betaが小さい候補ほどKL項が軽くなり、総合損失だけ見かけ上小さくなるため)。
# そのため候補間の比較は、betaに依存しない指標(実座標スケールの頂点距離・活性潜在次元数)で行う
from pathlib import Path
from typing import NamedTuple

import torch

from vae_checkpoint_v2 import load_checkpoint
from vae_data_v2 import SEED, Datasets, prepare_datasets
from vae_eval_common_v2 import ACTIVE_UNIT_THRESHOLD, kl_per_dim, vertex_distance_real
from vae_model_v2 import HIDDEN_DIMS, LATENT_DIM, flatten_input, select_device
from vae_training_v2 import BETA, GUMBEL_TEMPERATURE, KL_ANNEALING_EPOCHS, SLOT_ATTENTION_CONFIG, train

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "checkpoints"


class Candidate(NamedTuple):
    name: str
    hidden_dims: tuple[int, int]
    latent_dim: int
    beta: float
    kl_annealing_epochs: int


# 既定値(BETA=1.0・KL_ANNEALING_EPOCHS=60・LATENT_DIM=48)のまま学習したところ、潜在次元48個中38個が
# deadになり、実座標スケールの頂点距離誤差がデータ前処理のクラスタリング誤差(平均1.23)の11倍以上
# (平均14.36)になる、強いposterior collapseが確認された。この3候補はそれぞれ異なる仮説(再構成損失に
# 対してKLの重みが強すぎる/warm-upが速すぎる/潜在次元数がこのタスクの実質的な複雑さに対して過大)を
# 切り分けるためのもの
CANDIDATES = [
    Candidate("low_beta", HIDDEN_DIMS, LATENT_DIM, BETA / 4, KL_ANNEALING_EPOCHS),  # KL圧力を1/4に弱める
    Candidate("slow_kl_annealing", HIDDEN_DIMS, LATENT_DIM, BETA, int(KL_ANNEALING_EPOCHS * 2.5)),  # warm-upを2.5倍緩やかにする
    Candidate("small_latent", HIDDEN_DIMS, LATENT_DIM // 3, BETA, KL_ANNEALING_EPOCHS),  # 潜在次元をこのタスクの実質的な複雑さに近づける
]


def _candidate_checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"vae_v2_sweep_{name}.pt"


def _evaluate_candidate(checkpoint_path: Path, datasets: Datasets, device: torch.device) -> tuple[float, int, int]:
    # 戻り値: (実座標スケールの頂点距離の平均, dead次元数, 潜在次元数)。train()が返すbest_val_lossは
    # beta依存のため使わず、保存済みチェックポイントを読み込んでbeta非依存の指標を計算し直す
    checkpoint = load_checkpoint(device, checkpoint_path)
    vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence = (
        t.to(device) for t in datasets.val.tensors
    )

    with torch.no_grad():
        x = flatten_input(
            vertices, vertex_existence, stroke_vertex_indices, stroke_offsets, stroke_existence, checkpoint.shape
        )
        mu, logvar = checkpoint.model.encode(x)
        decoder_output = checkpoint.model.decode(mu)

    kl = kl_per_dim(mu, logvar)
    dead_dims = int((kl < ACTIVE_UNIT_THRESHOLD).sum().item())
    distance = vertex_distance_real(
        vertices, vertex_existence, decoder_output.vertex_features, checkpoint.vertex_mean, checkpoint.vertex_std
    )
    return float(distance.mean()), dead_dims, len(kl)


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    print(f"Using device: {device}")

    datasets = prepare_datasets()

    print("=== baseline (already trained, not re-run) ===")
    print("beta=1.0 kl_annealing_epochs=60 latent_dim=48")
    print("vertex distance (real scale) mean=14.3636 max=83.8420, dead dims=38/48\n")

    results: list[tuple[str, float, int]] = []
    for candidate in CANDIDATES:
        print(f"=== {candidate.name} (beta={candidate.beta}, kl_annealing_epochs={candidate.kl_annealing_epochs}, "
              f"latent_dim={candidate.latent_dim}) ===")
        checkpoint_path = _candidate_checkpoint_path(candidate.name)
        train(
            datasets,
            device,
            hidden_dims=candidate.hidden_dims,
            latent_dim=candidate.latent_dim,
            slot_attention_config=SLOT_ATTENTION_CONFIG,
            gumbel_temperature=GUMBEL_TEMPERATURE,
            beta=candidate.beta,
            kl_annealing_epochs=candidate.kl_annealing_epochs,
            checkpoint_path=checkpoint_path,
        )

        mean_distance, dead_dims, latent_dim = _evaluate_candidate(checkpoint_path, datasets, device)
        results.append((candidate.name, mean_distance, dead_dims))
        print(
            f"{candidate.name}: vertex distance (real scale) mean={mean_distance:.4f}, "
            f"dead dims={dead_dims}/{latent_dim}\n"
        )

    print("=== Summary (sorted by vertex distance mean, lower is better) ===")
    for name, mean_distance, dead_dims in sorted(results, key=lambda r: r[1]):
        print(f"{name}: mean_distance={mean_distance:.4f} dead_dims={dead_dims}")


if __name__ == "__main__":
    main()
