#!/usr/bin/env python3
# ストロークの曲線(vae_eval_common.draw_curves相当)はまだ無い段階のため、頂点を点群として描画する
import matplotlib.pyplot as plt
import numpy as np
import torch

from vae_checkpoint_v2 import load_checkpoint
from vae_data_v2 import prepare_datasets
from vae_model_v2 import flatten_input, select_device, unflatten_output

TYPICAL_INDICES = [0, 1, 2, 3]  # 典型的なサンプルとして固定で表示する検証データの先頭4件
POINT_SIZE = 20  # 正解・再構成の各点を見分けやすい大きさとして目視で選んだ値
DISPLACEMENT_LINE_WIDTH = 0.6  # 正解と再構成を結ぶ線が点より目立ちすぎないよう目視で選んだ値


def _sample_mean_distance(true_real: np.ndarray, existence: np.ndarray, recon_real: np.ndarray) -> np.ndarray:
    # サンプルごとの平均距離(存在する頂点のみ)。ワースト字を選ぶために使う
    distance = np.linalg.norm(recon_real - true_real, axis=-1)
    return (distance * existence).sum(axis=1) / existence.sum(axis=1)


def _draw_vertices(ax: plt.Axes, true_points: np.ndarray, recon_points: np.ndarray, title: str) -> None:
    # SVGはy軸が下向きのため、他のレンダリングスクリプトと同様上向きに合わせて反転する
    ax.scatter(true_points[:, 0], -true_points[:, 1], c="black", s=POINT_SIZE, label="true", zorder=3)
    ax.scatter(recon_points[:, 0], -recon_points[:, 1], c="red", s=POINT_SIZE, marker="x", label="recon", zorder=3)
    for true_p, recon_p in zip(true_points, recon_points):
        ax.plot(
            [true_p[0], recon_p[0]], [-true_p[1], -recon_p[1]], c="gray", linewidth=DISPLACEMENT_LINE_WIDTH, zorder=1
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    device = select_device()
    checkpoint = load_checkpoint(device)
    datasets = prepare_datasets()

    val_vertices_std, val_existence = (t.to(device) for t in datasets.val.tensors)

    with torch.no_grad():
        # 再パラメータ化のサンプリングεによるランダム性を排除するため、muをそのままdecodeする
        mu, _ = checkpoint.model.encode(flatten_input(val_vertices_std, val_existence))
        vertices_recon, _ = unflatten_output(checkpoint.model.decode(mu), checkpoint.shape)

    true_real = (val_vertices_std * checkpoint.std + checkpoint.mean).cpu().numpy()
    recon_real = (vertices_recon * checkpoint.std + checkpoint.mean).cpu().numpy()
    existence_np = val_existence.cpu().numpy().astype(bool)

    worst_index = int(np.argmax(_sample_mean_distance(true_real, existence_np, recon_real)))
    indices = [*TYPICAL_INDICES, worst_index]

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    for col, index in enumerate(indices):
        mask = existence_np[index]
        title = f"idx {index}" + (" (worst)" if index == worst_index else "")
        _draw_vertices(axes[col], true_real[index][mask], recon_real[index][mask], title)
    # 凡例をプロット領域の外(図の上部)に出す。内側に置くと点や線と被ることがあるため
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    plt.show()


if __name__ == "__main__":
    main()
