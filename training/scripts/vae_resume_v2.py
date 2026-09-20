#!/usr/bin/env python3
# train()の学習再開機能が使う、学習再開用の状態(モデル重み・optimizer状態・epoch数)の保存形式。
# vae_checkpoint_v2.pyの推論用チェックポイントとは別ファイル・別フォーマットで持つ(推論用チェックポイントは
# 学習再開に必要なoptimizer状態・epoch数を含まない設計のため)
from pathlib import Path
from typing import NamedTuple

import torch
import torch.optim as optim

from vae_model_v2 import VAE, ModelShape, SlotAttentionConfig

_RESUME_STATE_SUFFIX = "_resume"  # チェックポイントのファイル名にこの文字列を挟んで学習再開状態のパスを作る


def resume_state_path(checkpoint_path: Path) -> Path:
    # 学習再開用の状態は、save_checkpointが書く推論用のチェックポイントとは別ファイルに保存する。
    # ファイル名は対応するcheckpoint_pathから機械的に導出できるようにし、
    # checkpoint_path_from_resume_stateで逆変換できる
    return checkpoint_path.with_name(f"{checkpoint_path.stem}{_RESUME_STATE_SUFFIX}{checkpoint_path.suffix}")


def checkpoint_path_from_resume_state(resume_path: Path) -> Path:
    # --resumeに指定された状態ファイルから、対応する(再開後も書き続けるべき)チェックポイントの
    # 保存先を逆算する。resume_state_pathの命名規則の逆変換
    assert resume_path.stem.endswith(_RESUME_STATE_SUFFIX), (
        f"{resume_path} does not match the resume-state naming convention "
        f"(*{_RESUME_STATE_SUFFIX}{resume_path.suffix})"
    )
    stem = resume_path.stem[: -len(_RESUME_STATE_SUFFIX)]
    return resume_path.with_name(f"{stem}{resume_path.suffix}")


class ResumeState(NamedTuple):
    model: VAE
    optimizer: optim.Optimizer
    epoch: int
    best_val_loss: float
    epochs_without_improvement: int
    # 呼び出し元が使おうとしているハイパーパラメータが、実際に保存されていたものと一致するかを
    # 検証するために返す(不一致のまま進むと、以後保存するチェックポイント・再開状態のメタデータが
    # 実際のモデル構造と食い違ったまま上書きされてしまう)
    hidden_dims: tuple[int, int]
    latent_dim: int
    slot_attention_config: SlotAttentionConfig
    gumbel_temperature: float


def load_resume_state(shape: ModelShape, device: torch.device, learning_rate: float, resume_path: Path) -> ResumeState:
    # DataLoaderのシャッフル順・optimizerのモーメンタム等、学習の完全な再現までは保証しない
    # (再開のたびに同じ乱数列を辿り直す設計はしていない)。目的は中断による手戻りを避けることであり、
    # 中断が無かった場合とビット単位で一致する学習経過を再現することではない
    saved = torch.load(resume_path, map_location=device, weights_only=False)
    slot_attention_config = SlotAttentionConfig(*saved["slot_attention_config"])
    model = VAE(
        shape, saved["hidden_dims"], saved["latent_dim"], slot_attention_config, saved["gumbel_temperature"]
    ).to(device)
    model.load_state_dict(saved["model_state_dict"])
    # learning_rateはoptimizerを構築するためだけに使い、直後のload_state_dictで保存時の学習率に
    # 上書きされる(optimizerのparam_groups自体がstate_dictに含まれるため)。つまり再開時は
    # 常に元の学習率が引き継がれ、呼び出し元のlearning_rateは実質無視される
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    return ResumeState(
        model, optimizer, saved["epoch"], saved["best_val_loss"], saved["epochs_without_improvement"],
        saved["hidden_dims"], saved["latent_dim"], slot_attention_config, saved["gumbel_temperature"],
    )


def save_resume_state(
    model: VAE,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    epochs_without_improvement: int,
    hidden_dims: tuple[int, int],
    latent_dim: int,
    slot_attention_config: SlotAttentionConfig,
    gumbel_temperature: float,
    resume_path: Path,
) -> None:
    # 同じファイルを毎エポック上書きするため、書き込み中にプロセスが落ちると再開状態自体が
    # 壊れてしまいかねない。一時ファイルに書いてからPath.replace(POSIXでは原子的)で置き換えることで、
    # 書き込みが完全に終わったファイルだけが常にresume_pathに存在するようにする
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resume_path.with_name(f"{resume_path.name}.tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "slot_attention_config": tuple(slot_attention_config),
            "gumbel_temperature": gumbel_temperature,
        },
        tmp_path,
    )
    tmp_path.replace(resume_path)
