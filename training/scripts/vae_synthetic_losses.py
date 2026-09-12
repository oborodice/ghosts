#!/usr/bin/env python3
import torch

from vae_losses import AngleGMM, angle_log_density, off_diagonal_exist_pairs, weighted_crossing_penalty
from vae_model import VAE, ModelShape, unflatten_output

# 実在漢字への反発力。生成結果が特定の実在字とほぼ一致しやすい問題への対応。複数の実在字のmuを混ぜた
# 合成z(学習が一度も見ない領域)をdecodeし、交差抑制・角度自然さの2つの「文法」損失を適用できるようにする。
# 正解データを必要としない絶対評価であり、対応する正解ペアが存在しないため接続点一致損失・近傍点間隔一致損失
# (いずれも正解データの実測値が目標値になる、vae_losses.py参照)は適用できない
SYNTHETIC_BANDWIDTH = 0.6  # weight sweepの結果、0.6が最良だった。1.0〜2.0はバッチ内mu(数十件、疎)を対象に混ぜすぎになり、交差・斜め関与の改善幅は0.6よりやや大きいものの3本以上合流とstrokes_mseが悪化する非単調な副作用が出たため、副作用が無い、または最小の0.6を採用した
SYNTHETIC_CROSSING_WEIGHT = 0.5  # weight sweepの結果、0.3・0.5はベースラインに対し全指標で悪化ゼロだったが、0.5の方が交差抑制効果が強くstrokes_mseの悪化も小さいため採用した(1.0はさらに強力だが合流とstrokes_mseの悪化が大きく、悪化ゼロの条件を満たさなかった)
SYNTHETIC_ANGLE_WEIGHT = 0.0  # 角度損失は無効化した。対数密度をそのまま最大化する設計のため、重みを上げるほど生成結果の角度が最頻値(横画)に偏り、実データの角度分布(斜めが最多)から大きく外れる副作用が確認された。SYNTHETIC_CROSSING_WEIGHT単体でも交差数・3本以上合流はベースラインを下回っており、角度損失なしでも文法面の改善は十分に得られている。分布の形を保つ設計に作り直せない限り再度有効化しない
SYNTHETIC_EXISTENCE_THRESHOLD = 0.5  # 合成データの予測existenceをマスク化する閾値。vae_eval_common.EXISTENCE_THRESHOLDと同じ考え方
MASKED_STD_EPS = 1e-6  # masked_mean_stdの標準偏差計算で、平方根が0付近で微分不能になるのを避けるための微小値

# 2段階目の微調整(finetune_synthetic_stats.py)専用のストローク数損失。生成結果は実データよりストローク数が
# 少なく(平均10.9画 vs 14.6画)多様性も乏しい(std2.4 vs 5.1)偏りがあり、GANによる検証でこの軸が改善対象
# として意味のあるものだと裏付け済み。この損失は潜在空間が「近い字ほど似た性質を持つ」という構造に整理
# されて初めて意味のある目標値になり、scratchからの学習序盤ではこの前提が満たされず効果が弱いため、
# compute_synthetic_grammar_lossには組み込まず収束済みチェックポイントへの微調整としてのみ適用する
FINETUNE_STROKE_COUNT_WEIGHT = 1.0  # 2.0でも3000ステップの範囲では改善が見られず(平均12〜13で頭打ち)、weightの大小では平均の頭打ちは解消しないことを確認済み。以前の設計(z_synthetic構築が学習ミニバッチ・SYNTHETIC_BANDWIDTH依存だった頃)では、weight=1.0でも長時間続けると目標を追い越してオーバーシュートすることを確認しており(sigmoidの非対称性によりexistenceを増やす方向は勾配が乗りやすく、減らす方向は乗りにくいため)、現在の設計でも同じリスクが完全には否定できない。呼び出し側で目標乖離が最小の時点を選ぶearly stoppingと組み合わせて使うことが前提
FINETUNE_SYNTHETIC_BANDWIDTH = 1.5  # zの引き寄せに使う帯域。SYNTHETIC_BANDWIDTH(0.6)は1段階目の文法損失
# (交差抑制・角度自然さという、場所に依存しない普遍的なルール)用に調整された値であり、そのまま流用すると
# 学習時に合成zが実際に配置される位置と、実際の生成時(vae_eval_common.KERNEL_BANDWIDTH=1.5)の位置がずれる。
# ストローク数損失は場所に紐づいた個別の対応関係(この位置はこの近傍のストローク数であるべき)を学習する
# 設計のため、この位置ずれが致命的(生成時に多様性が失われて見える)になることを確認済み。
# vae_eval_common.KERNEL_BANDWIDTHと同じ値にする(直接importはしない。値を変える際は両方を確認すること)
FINETUNE_TARGET_BANDWIDTH = 0.1  # 目標ストローク数を選ぶための重みの帯域。FINETUNE_SYNTHETIC_BANDWIDTHと
# 同じ(1.5)にすると複数の実在字の加重平均で目標が平滑化され過ぎ、平均・標準偏差ともむしろ悪化することを
# 確認済みのため、最近傍1〜2字に近い値を目標にできるよう分離したまま維持する

# 2段階目の微調整専用のストローク長損失。生成結果は実データよりストロークが短く(平均20.3 vs 33.0)、
# 長さのばらつきも乏しい(std8.7 vs 16.9)偏りがあり、分類器診断でこの軸が正答率のほとんどを説明する
# 主要因だと判明済み。ストローク数と違い、目標値は
# 「実データ全体で固定の平均・標準偏差」という位置に依存しない値のため、この損失自体は交差抑制と同じ
# 普遍的なルールとして1段階目のscratch学習に組み込めるはずだった。しかし実際にはバッチ全体の統計を
# 目標に合わせるこの設計に、目標付近で自然に釣り合う固定点が無く、重みの大小に関わらず時間とともに
# 目標を追い越して高止まりすることを確認済み(ストローク数の集計統計版が繰り返しドリフトしたのと同じ
# 構造)。そのため、損失の形はそのまま(個別サンプル目標ではなくバッチ全体の統計マッチング)に、適用の
# 仕方だけをストローク数と同じ「収束済みチェックポイントへの2段階目微調整+目標乖離ベースのearly stopping」
# に変更して使う
FINETUNE_LENGTH_WEIGHT = 0.5  # weight sweepの結果、1.0は数百ステップで目標を追い越すのに対し、0.5は
# 目標付近を通過するまでの時間が長く、early stoppingで捉えやすいため採用した(0.3はさらに緩やかだが、
# 4000ステップ時点でまだ目標に届いておらず収束が遅すぎた)


def _kernel_weights(z_raw: torch.Tensor, mu_batch: torch.Tensor, bandwidth: float) -> torch.Tensor:
    # Nadaraya-Watson推定量のカーネル重み。_attract_batch(zの引き寄せ)と
    # compute_finetune_stroke_count_loss(目標ストローク数の選定)で共有する
    dist_sq = torch.cdist(z_raw, mu_batch) ** 2
    return torch.softmax(-dist_sq / (2 * bandwidth * bandwidth), dim=1)


def _attract_batch(z_raw: torch.Tensor, mu_batch: torch.Tensor, bandwidth: float) -> torch.Tensor:
    # vae_eval_common.attract_to_latent_priorと同じNadaraya-Watson推定量だが、decoderへ勾配を
    # 通す必要があるため@torch.no_gradにはしない(mu_batch側は呼び出し元で事前にdetachする)
    return _kernel_weights(z_raw, mu_batch, bandwidth) @ mu_batch


def _synthetic_existence_mask(existence_logits: torch.Tensor) -> torch.Tensor:
    # 合成データには正解のexistenceが存在しないため、モデル自身の予測値をマスクとして使う。
    # マスクは離散的な採用判定であり勾配は不要なため、vae_losses._crossing_pair_weightsのnot_crossingと
    # 同じくdetachする
    with torch.no_grad():
        return (torch.sigmoid(existence_logits) > SYNTHETIC_EXISTENCE_THRESHOLD).float()


def _decode_synthetic_batch(
    model: VAE, mu: torch.Tensor, shape: ModelShape
) -> tuple[torch.Tensor, torch.Tensor]:
    # バッチ内の実データのmu(引き寄せ先の代わり、勾配を通さないようdetach)へ、ランダムなz_rawを
    # 引き寄せて複数の実在字を混ぜた合成zを作りdecodeする。compute_synthetic_grammar_loss・
    # compute_finetune_length_lossの両方から使う共有ヘルパー(いずれも「位置に依存しない一律のルール」
    # という同じ性質の損失で、z_synthetic自体の作り方も同じでよいため)
    z_raw = torch.randn_like(mu)
    z_synthetic = _attract_batch(z_raw, mu.detach(), SYNTHETIC_BANDWIDTH)
    recon_synthetic = model.decode(z_synthetic)
    strokes_recon_synthetic, existence_logits_synthetic = unflatten_output(recon_synthetic, shape)
    existence_synthetic = _synthetic_existence_mask(existence_logits_synthetic)
    return strokes_recon_synthetic, existence_synthetic


def masked_mean_std(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # マスクされた要素だけを対象に、行(サンプルまたは字)ごとの平均・標準偏差・有効要素数を求める。
    # finetune_synthetic_stats.pyでの目標値(実データ全体)算出と、_length_moment_matching_lossの
    # 両方から使う共有ヘルパー
    count = mask.sum(dim=1)
    safe_count = count.clamp(min=1)
    mean = (values * mask).sum(dim=1) / safe_count
    var = (((values - mean.unsqueeze(1)) ** 2) * mask).sum(dim=1) / safe_count
    std = torch.sqrt(var + MASKED_STD_EPS)
    return mean, std, count


def _compute_synthetic_crossing_loss(
    strokes_recon: torch.Tensor, existence: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    # 正解データがなく「意図した交差」を除外できないため、存在するペア全てが交差しないのが正しいという
    # 前提を置く(実データでも交差は少数派のため、多くの場合で妥当な近似になる)
    weights = off_diagonal_exist_pairs(existence)
    return weighted_crossing_penalty(strokes_recon, weights, mean, std)


def _compute_synthetic_angle_loss(
    strokes_recon: torch.Tensor, existence: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, angle_gmm: AngleGMM
) -> torch.Tensor:
    # 比較対象となる正解の角度が存在しないため、vae_losses._compute_angle_naturalness_lossのような
    # 相対評価ではなく、GMMの対数密度をそのまま最大化する絶対評価を行う
    # (合成データは常に典型的な角度に近づけるべきという前提)
    recon_angle = strokes_recon[..., 2] * std[2] + mean[2]
    recon_log_density = angle_log_density(recon_angle, angle_gmm)
    # 対数密度は大きいほど良いため、損失としては符号を反転する
    return (-recon_log_density * existence).sum(dim=1).mean()


def compute_synthetic_grammar_loss(
    model: VAE,
    mu: torch.Tensor,
    shape: ModelShape,
    mean: torch.Tensor,
    std: torch.Tensor,
    angle_gmm: AngleGMM,
) -> torch.Tensor:
    # decodeした出力に文法(交差抑制・角度自然さ)損失を課す
    strokes_recon_synthetic, existence_synthetic = _decode_synthetic_batch(model, mu, shape)

    crossing_loss = _compute_synthetic_crossing_loss(strokes_recon_synthetic, existence_synthetic, mean, std)
    angle_loss = _compute_synthetic_angle_loss(strokes_recon_synthetic, existence_synthetic, mean, std, angle_gmm)
    return SYNTHETIC_CROSSING_WEIGHT * crossing_loss + SYNTHETIC_ANGLE_WEIGHT * angle_loss


def _per_sample_stroke_count_loss(
    existence_logits: torch.Tensor, weights: torch.Tensor, real_count: torch.Tensor
) -> torch.Tensor:
    # 各合成サンプルの目標ストローク数を、そのサンプルの合成に使った実在字群(weights)のストローク数の
    # 加重平均とする。ハード閾値ではなくsigmoid確率の合計(連続値)を「期待ストローク数」として使うことで、
    # existence_logitsまで勾配を通す
    expected_count = torch.sigmoid(existence_logits).sum(dim=1)
    target_count = weights @ real_count
    return ((expected_count - target_count) ** 2).mean()


def compute_finetune_stroke_count_loss(
    model: VAE,
    mu: torch.Tensor,
    shape: ModelShape,
    target_pool_mu: torch.Tensor,
    target_pool_count: torch.Tensor,
) -> torch.Tensor:
    # z_synthetic自体もtarget_pool_mu(実データ全件)・FINETUNE_SYNTHETIC_BANDWIDTHで引き寄せる
    # (理由はFINETUNE_SYNTHETIC_BANDWIDTHの定義部分を参照)
    z_raw = torch.randn(mu.shape[0], target_pool_mu.shape[1], device=mu.device)
    z_synthetic = _attract_batch(z_raw, target_pool_mu.detach(), FINETUNE_SYNTHETIC_BANDWIDTH)
    recon_synthetic = model.decode(z_synthetic)
    _, existence_logits_synthetic = unflatten_output(recon_synthetic, shape)
    target_weights = _kernel_weights(z_raw, target_pool_mu.detach(), FINETUNE_TARGET_BANDWIDTH)
    return FINETUNE_STROKE_COUNT_WEIGHT * _per_sample_stroke_count_loss(
        existence_logits_synthetic, target_weights, target_pool_count
    )


def _length_moment_matching_loss(
    strokes_recon: torch.Tensor,
    existence: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_length_mean: torch.Tensor,
    target_length_std: torch.Tensor,
) -> torch.Tensor:
    # 各合成サンプルの長さの平均・標準偏差を求め、バッチ全体で平均したものを実データ全体の固定値に
    # 近づける。ストローク数の個別サンプル目標と異なり、目標値がサンプルの位置(どの実在字群を混ぜたか)に
    # 依存しない一律の値であるため、対応関係がバッチ内のどのサンプルにも等しく当てはまる
    length = strokes_recon[..., 3] * std[3] + mean[3]
    per_sample_mean, per_sample_std, count = masked_mean_std(length, existence)
    valid = count >= 2  # 標準偏差は2点以上ないと定義できない
    if valid.sum() == 0:
        return torch.zeros((), device=strokes_recon.device)
    batch_mean = per_sample_mean[valid].mean()
    batch_std = per_sample_std[valid].mean()
    return (batch_mean - target_length_mean) ** 2 + (batch_std - target_length_std) ** 2


def compute_finetune_length_loss(
    model: VAE,
    mu: torch.Tensor,
    shape: ModelShape,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_length_mean: torch.Tensor,
    target_length_std: torch.Tensor,
) -> torch.Tensor:
    # 目標値がバッチ内のどのサンプルにも等しく当てはまる一律の値のため、ストローク数と違いzの引き寄せ先を
    # 実データ全件にする必要はなく、compute_synthetic_grammar_lossと同じ構築方法(_decode_synthetic_batch)を使う
    strokes_recon_synthetic, existence_synthetic = _decode_synthetic_batch(model, mu, shape)
    return FINETUNE_LENGTH_WEIGHT * _length_moment_matching_loss(
        strokes_recon_synthetic, existence_synthetic, mean, std, target_length_mean, target_length_std
    )
