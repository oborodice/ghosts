#!/usr/bin/env python3
from typing import NamedTuple

import torch
import torch.nn.functional as F

from vae_crossing_geometry_v2 import folded_crossing_per_sample_total
from vae_eval_common import EXISTENCE_THRESHOLD
from vae_eval_common_v2 import KERNEL_BANDWIDTH, attract_to_pool
from vae_losses_v2 import MIN_DIRECTION_NORM
from vae_model_v2 import VAE, DecoderOutput


class SyntheticLossComponents(NamedTuple):
    # vae_losses_v2.LossComponentsと同じ構成(合計+内訳)を、混ぜ合わせz側の損失について持つ。
    # 個々の損失を追加するたびに(a)このNamedTupleへのフィールド追加(b)compute_synthetic_lossの
    # lossesへの1行(c)重みが1.0以外ならweightsへの1行、で済むようにする(vae_losses_v2.compute_lossと
    # 同じ設計)
    total: torch.Tensor
    self_loop_loss: torch.Tensor
    phantom_reference_loss: torch.Tensor
    synthetic_crossing_loss: torch.Tensor


SELF_LOOP_LOSS_WEIGHT = 1.0  # 1本のストロークの始点・終点ポインタが同じ頂点を指してしまう自己ループ
# (実データでは常に0%、混ぜ合わせ生成時に特有の現象)を抑制する。既存コードに対応物が存在しない
# 新規の損失のため暫定値とする
PHANTOM_REFERENCE_LOSS_WEIGHT = 1.0  # ストロークのポインタが、existenceヘッドが「存在しない」と判定した
# 頂点を指してしまう現象(実データでは常に0%)を抑制する。self_loop_lossと同じ理由で新規の損失のため暫定値とする
SYNTHETIC_CROSSING_WEIGHT = 1.0  # 生成側crossings頻度を実データの頻度分布に近づけるmoment matching損失。
# 既存コードに対応物が存在しない新規の損失のため暫定値とする


def _synthetic_existence_mask(existence_logits: torch.Tensor) -> torch.Tensor:
    # 合成データには正解のexistenceが存在しないため、モデル自身の予測値をマスクとして使う。
    # マスクは離散的な採用判定であり勾配は不要なためdetachする
    with torch.no_grad():
        return (torch.sigmoid(existence_logits) > EXISTENCE_THRESHOLD).float()


def _decode_synthetic_batch(model: VAE, mu: torch.Tensor, mu_pool: torch.Tensor, detach_slots: bool) -> DecoderOutput:
    # 訓練データ全体のmu(mu_pool、呼び出し元でエポック単位にキャッシュ・detach済み)へランダムな
    # z_rawを引き寄せた合成zを構築してdecodeする。合成z側の損失(self_loop・phantom_reference・crossing)
    # がいずれも最初に行う共通処理。z_rawの件数はmu(このバッチの実データ数)に合わせるが、引き寄せ先は
    # mu_poolでありこのバッチ自体ではない。decoderへ勾配を通す必要があるためno_gradにはしない
    z_raw = torch.randn(mu.shape[0], mu_pool.shape[1], device=mu_pool.device)
    z_synthetic, _ = attract_to_pool(z_raw, mu_pool, KERNEL_BANDWIDTH)
    return model.decode(z_synthetic, detach_slots=detach_slots)


def _compute_self_loop_penalty(
    start_logits: torch.Tensor, end_logits: torch.Tensor, existence: torch.Tensor
) -> torch.Tensor:
    # 始点・終点ポインタのsoftmax分布の内積を「衝突確率」(2つの分布から独立にサンプリングした場合に
    # 同じ頂点を選んでしまう確率)として使う。コサイン類似度と異なり正規化しないため、分布の鋭さ
    # (自信度)も反映した、実際に防ぎたい事象(自己ループ)に直接対応する量になる
    p_start = F.softmax(start_logits, dim=-1)
    p_end = F.softmax(end_logits, dim=-1)
    collision_probability = (p_start * p_end).sum(dim=-1)
    return (collision_probability * existence).sum(dim=1).mean()


def _compute_phantom_reference_penalty(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    vertex_existence_logits: torch.Tensor,
    existence: torch.Tensor,
) -> torch.Tensor:
    # start/endそれぞれのポインタのsoftmax分布と、頂点ごとの「存在しない確率」(1 - sigmoid(existence
    # logit))の内積を、「存在しないと判定された頂点を指してしまう期待確率」として使う。self_loop_loss
    # (2つの分布の衝突確率)と同じ発想。existence logitはdetachする(この損失がポインタ側だけを動かし、
    # existenceヘッド側を「全部存在するとみなす」ことで安く損失を消す抜け道を防ぐため)
    non_existence_probability = 1.0 - torch.sigmoid(vertex_existence_logits.detach())
    p_start = F.softmax(start_logits, dim=-1)
    p_end = F.softmax(end_logits, dim=-1)
    phantom_probability = (p_start + p_end) @ non_existence_probability.unsqueeze(-1)
    return (phantom_probability.squeeze(-1) * existence).sum(dim=1).mean()


def _compute_pointer_penalty_losses(
    model: VAE, mu: torch.Tensor, mu_pool: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # self_loop_loss・phantom_reference_lossはいずれも、教師すべきなのはポインタの各headの重みだけであり、
    # vertex_features等と共有されているTransformer出力(slots)自体を変える必要はない(detach_slots=True)。
    # self_loop_lossでは、detachしないと混ぜ合わせz(実データより自由度が高い領域)での学習を通じて、
    # 共有されたslotsを経由し頂点座標側に意図しない副作用(過剰接続の悪化)が漏れることを実測で確認済み。
    # phantom_reference_lossは同じ構造(ポインタのみを教師する合成z上の損失)のため同じ設計を予防的に
    # 適用しているが、こちら単体でdetach_slots=Falseにした場合の副作用は個別には検証していない。
    # 両損失とも同じdetach_slots=Trueで済むため、合成zの構築・decodeを1回で共有する(2回計算する無駄を避ける)
    decoder_output = _decode_synthetic_batch(model, mu, mu_pool, detach_slots=True)
    existence = _synthetic_existence_mask(decoder_output.stroke_existence_logits)
    self_loop = _compute_self_loop_penalty(
        decoder_output.start_pointer_logits, decoder_output.end_pointer_logits, existence
    )
    phantom_reference = _compute_phantom_reference_penalty(
        decoder_output.start_pointer_logits,
        decoder_output.end_pointer_logits,
        decoder_output.vertex_existence_logits,
        existence,
    )
    return self_loop, phantom_reference


def _huber(residual: torch.Tensor, delta: float) -> torch.Tensor:
    # 残差がdelta以内ならL2(滑らかで目標付近での精密な収束を維持)、delta超ならL1
    # (勾配の大きさがdeltaで頭打ちになり暴走を防ぐ)に切り替わる、外れ値に頑健な標準的な損失。
    # スクラッチ学習(ランダム初期化直後)ではバッチ集計統計が目標から大きく乖離しうるため、
    # 素朴な二乗誤差だと乖離の大きさに応じて勾配が際限なく増幅し、共有Transformer全体を
    # 経由して学習全体を破壊することを実測で確認済み(detach_slots=Falseのため)
    abs_residual = residual.abs()
    quadratic = torch.clamp(abs_residual, max=delta)
    linear = abs_residual - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _compute_synthetic_crossing_loss(
    model: VAE,
    mu: torch.Tensor,
    mu_pool: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> torch.Tensor:
    # 合成サンプル1つあたりの交差数(offsetを使った折れ線近似。vae_data_v2._compute_crossing_targets
    # と同じ判定方法)を求め、バッチ内の平均・標準偏差を実データ全体の固定目標値に近づけるmoment
    # matching損失(旧`vae_synthetic_losses.py`の`_length_moment_matching_loss`と発想は同じだが、
    # 二乗誤差ではなくHuber損失を使う。理由は_huberのコメントを参照)。
    # start/endをdetachしoffsetのみ勾配を通すのは、offsetレバー設計: 頂点座標(start/end)は
    # 再構成側のvertex_lossが強く教師する量であり、crossing_lossにも動かせてしまうと頂点配置を
    # 経由して他ストロークとの接続関係を壊す副作用が大きいことを実測で確認済み。offset(曲がり具合)
    # は頂点配置と独立に交差を作れるレバーであり、この損失専用に使わせても副作用が小さい
    decoder_output = _decode_synthetic_batch(model, mu, mu_pool, detach_slots=False)
    existence = _synthetic_existence_mask(decoder_output.stroke_existence_logits)
    start, end = decoder_output.start_points.detach(), decoder_output.end_points.detach()
    offset = decoder_output.stroke_offsets

    well_defined = (((end - start) * vertex_std).norm(dim=-1) >= MIN_DIRECTION_NORM).float()
    per_sample_total = folded_crossing_per_sample_total(
        start, end, offset, existence, well_defined, vertex_std, stroke_offset_mean, stroke_offset_std
    )
    batch_mean = per_sample_total.mean()
    batch_std = per_sample_total.std()
    return _huber(batch_mean - target_mean, target_std) + _huber(batch_std - target_std, target_std)


def compute_synthetic_loss(
    model: VAE,
    mu: torch.Tensor,
    mu_pool: torch.Tensor,
    vertex_std: torch.Tensor,
    stroke_offset_mean: torch.Tensor,
    stroke_offset_std: torch.Tensor,
    target_crossings_mean: float,
    target_crossings_std: float,
) -> SyntheticLossComponents:
    # model自体を使って混ぜ合わせzをdecodeする必要があり、vae_losses_v2.compute_lossが受け取る
    # decoder_output(再構成側のdecode結果)だけでは完結しないため別枠にしている。mu_poolは訓練データ
    # 全体のmu(呼び出し元でエポック単位にキャッシュ・detach済み)で、本番の生成が引き寄せ先として
    # 使うattract_to_latent_priorの候補プールと同じもの。muはこのバッチの実データのmuで、合成zを
    # 何件作るか(z_rawの件数)を揃えるためだけに使う。
    # lossesは重み乗算前の生の値。個々の損失を追加するたびに(a)このdictへの1行(b)重みが1.0以外
    # ならweightsへの1行、の2箇所を触るだけで済む(vae_losses_v2.compute_lossと同じ設計)。
    # self_loop_loss・phantom_reference_loss(detach_slots=True)とcrossing_loss(detach_slots=False)は
    # 必要なdetach設定が異なるため、crossing_lossだけ独立に合成zを構築・decodeする
    losses: dict[str, torch.Tensor] = {}
    losses["self_loop_loss"], losses["phantom_reference_loss"] = _compute_pointer_penalty_losses(model, mu, mu_pool)
    losses["synthetic_crossing_loss"] = _compute_synthetic_crossing_loss(
        model, mu, mu_pool, vertex_std, stroke_offset_mean, stroke_offset_std,
        target_crossings_mean, target_crossings_std,
    )

    # ここに列挙のない損失は暗黙的に重み1.0として扱う
    weights = {
        "self_loop_loss": SELF_LOOP_LOSS_WEIGHT,
        "phantom_reference_loss": PHANTOM_REFERENCE_LOSS_WEIGHT,
        "synthetic_crossing_loss": SYNTHETIC_CROSSING_WEIGHT,
    }
    total = sum(weights.get(name, 1.0) * value for name, value in losses.items())
    return SyntheticLossComponents(total=total, **losses)
