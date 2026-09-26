#!/usr/bin/env python3
# 生成(ノイズからデコーダに渡すzを作る処理)の仕組みをまとめる。本番の生成・診断スクリプト・学習中の
# 合成zの損失が、同じ実装を使う
import torch

# 生成時にzを実データへ引き寄せるカーネル幅。次元数が多いほど同じbandwidthでもsoftmax重みが均一化する
# (次元の呪い)ため、LATENT_DIMを変える場合は再較正が必要。1.5はLATENT_DIM=48のときに較正した値で、
# 現在のLATENT_DIM(96)向けの再較正はまだ行っていない
KERNEL_BANDWIDTH = 1.5

# 生成時のデコードで、ポインタ選択・頂点座標のビン選択を確率加重平均にするsoftmaxの温度(model.decode
# のsoft_temperature)。argmaxでは、zをわずかに動かしただけで選択が切り替わり、字が飛ぶ。
# 小さいほどargmaxに近づき、ジャンプが増える。大きいほどジャンプは減るが、字が小さく、ストロークが短くなる。
# 1.0は、この釣り合いで暫定的に選んだ値で、正式な調整は今後別途行う
GENERATION_SOFT_TEMPERATURE = 1.0

CORRECTION_FIT_SAMPLE_COUNT = 4000  # LatentSamplerの補正を推定するために引き寄せる点の数
CORRECTION_FIT_SEED = 1  # 補正を、構築のたびに同じ値にするため固定する
CORRECTION_MIN_VARIANCE = 1e-9  # 分散がほぼ0の主成分で、標準偏差の比を取るときのゼロ除算を避ける下限


def attract_to_pool(z_raw: torch.Tensor, pool: torch.Tensor, bandwidth: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Nadaraya-Watson推定量(重み付き平均)でz_rawをpoolへ引き寄せる。学習時の合成zの構築と、生成時
    # (LatentSampler。引き寄せた後に補正もかける)の引き寄せが、この同じ実装を経由するようにして、食い違いを防ぐ。
    # 重み(どのpool要素がどれだけ引き寄せに寄与したか)も返す。呼び出し元の大半は結果だけを使う
    # (`z_synthetic, _ = attract_to_pool(...)`)が、寄与元の実在字を特定したい診断用途では
    # この重みをそのまま使える
    dist_sq = torch.cdist(z_raw, pool) ** 2
    weights = torch.softmax(-dist_sq / (2 * bandwidth * bandwidth), dim=1)
    return weights @ pool, weights


def _fit_correction(real: torch.Tensor, blend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 引き寄せた点(blend)の分布を、実在字のz(real)の分布に合わせる線形の補正を求める。引き寄せた点の平均を
    # 実在字の平均へ移し、実在字の主成分ごとに、引き寄せた点の標準偏差を実在字にそろえる。
    # 返り値(matrix, offset)は、補正後 = blend_z @ matrix + offset となる行列とベクトル
    center = real.mean(dim=0)
    eigenvalues, components = torch.linalg.eigh(torch.cov((real - center).T))
    blend_mean = blend.mean(dim=0)
    blend_variance = (components.T @ torch.cov((blend - blend_mean).T) @ components).diagonal()
    std_ratio = (
        blend_variance.clamp(min=CORRECTION_MIN_VARIANCE) / eigenvalues.clamp(min=CORRECTION_MIN_VARIANCE)
    ).sqrt()
    matrix = components @ torch.diag(1.0 / std_ratio) @ components.T
    return matrix, center - blend_mean @ matrix


class LatentSampler:
    # 生成時に、ノイズ(z_raw)を、デコーダに渡すzへ写す。まず実在字のzへ引き寄せ(attract_to_pool)、
    # 次に、引き寄せた点の分布を、実在字のzの分布に合わせる線形の補正をかける。引き寄せは、密度で重み付けした
    # 平均なので、実在字の多い所(ストローク数の少ない字)へ偏り、実在字の重心の方へ縮む。補正は、この偏り
    # (平均のずれ)と縮み(主成分ごとの標準偏差の縮小)を直す。
    # 補正は、実在字のzと引き寄せの幅だけで決まるため、固定の乱数で(標準正規分布から)引き寄せた点から、
    # 構築時に一度だけ推定する(モデルや引き寄せの幅を変えれば、構築し直すだけで推定し直される)
    def __init__(self, mu_real: torch.Tensor) -> None:
        self.mu_real = mu_real
        generator = torch.Generator().manual_seed(CORRECTION_FIT_SEED)
        fit_raw = torch.randn(CORRECTION_FIT_SAMPLE_COUNT, mu_real.shape[1], generator=generator)
        with torch.no_grad():
            fit_blend, _ = attract_to_pool(fit_raw.to(mu_real.device), mu_real, KERNEL_BANDWIDTH)
        # 固有値分解は、MPSが対応していないためCPUで行う
        matrix, offset = _fit_correction(mu_real.cpu(), fit_blend.cpu())
        self.correction_matrix, self.correction_offset = matrix.to(mu_real.device), offset.to(mu_real.device)

    @torch.no_grad()
    def sample(self, z_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 補正したzと、引き寄せの重み(どの実在字がどれだけ寄与したか。診断用)を返す
        z_blend, weights = attract_to_pool(z_raw, self.mu_real, KERNEL_BANDWIDTH)
        return z_blend @ self.correction_matrix + self.correction_offset, weights
