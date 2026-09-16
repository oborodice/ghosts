#!/usr/bin/env python3
from typing import NamedTuple

import torch
import torch.nn.functional as F

from vae_model_v2 import VAE


class SyntheticLossComponents(NamedTuple):
    # vae_losses_v2.LossComponentsと同じ構成(合計+内訳)を、混ぜ合わせz側の損失について持つ。
    # 個々の損失を追加するたびに(a)このNamedTupleへのフィールド追加(b)compute_synthetic_lossの
    # lossesへの1行(c)重みが1.0以外ならweightsへの1行、で済むようにする(vae_losses_v2.compute_lossと
    # 同じ設計)
    total: torch.Tensor
    self_loop_loss: torch.Tensor


SYNTHETIC_BANDWIDTH = 0.6  # Nadaraya-Watson推定量の帯域。正式な値は今後のsweepで決める暫定値
SYNTHETIC_EXISTENCE_THRESHOLD = 0.5  # 合成データの予測existenceをマスク化する閾値
SELF_LOOP_LOSS_WEIGHT = 1.0  # 1本のストロークの始点・終点ポインタが同じ頂点を指してしまう自己ループ
# (実データでは常に0%、混ぜ合わせ生成時に特有の現象)を抑制する。既存コードに対応物が存在しない
# 新規の損失のため暫定値とする


def _attract_batch(z_raw: torch.Tensor, mu_batch: torch.Tensor, bandwidth: float) -> torch.Tensor:
    # Nadaraya-Watson推定量でz_rawをmu_batch(バッチ内の実データのmu、呼び出し元でdetach済み)へ
    # 引き寄せ、複数の実在字の潜在表現を混ぜた合成zを作る。decoderへ勾配を通す必要があるため
    # no_gradにはしない
    dist_sq = torch.cdist(z_raw, mu_batch) ** 2
    weights = torch.softmax(-dist_sq / (2 * bandwidth * bandwidth), dim=1)
    return weights @ mu_batch


def _synthetic_existence_mask(existence_logits: torch.Tensor) -> torch.Tensor:
    # 合成データには正解のexistenceが存在しないため、モデル自身の予測値をマスクとして使う。
    # マスクは離散的な採用判定であり勾配は不要なためdetachする
    with torch.no_grad():
        return (torch.sigmoid(existence_logits) > SYNTHETIC_EXISTENCE_THRESHOLD).float()


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


def _compute_self_loop_loss(model: VAE, mu: torch.Tensor) -> torch.Tensor:
    # バッチ内の実データのmuへランダムなz_rawを引き寄せた合成zをdecodeし、自己ループへの
    # ペナルティを計算する。重み乗算前の生の値を返す。
    # detach_slots=True: この損失が本来教師すべきなのはポインタの各headの重みだけであり、
    # vertex_features等と共有されているTransformer出力(slots)自体を変える必要はない。
    # detachしないと、混ぜ合わせz(実データより自由度が高い領域)での学習を通じて、
    # 共有されたslotsを経由し頂点座標側に意図しない副作用(過剰接続の悪化)が漏れることを実測で確認済み
    z_raw = torch.randn_like(mu)
    z_synthetic = _attract_batch(z_raw, mu.detach(), SYNTHETIC_BANDWIDTH)
    decoder_output = model.decode(z_synthetic, detach_slots=True)
    existence = _synthetic_existence_mask(decoder_output.stroke_existence_logits)
    return _compute_self_loop_penalty(
        decoder_output.start_pointer_logits, decoder_output.end_pointer_logits, existence
    )


def compute_synthetic_loss(model: VAE, mu: torch.Tensor) -> SyntheticLossComponents:
    # model自体を使って混ぜ合わせzをdecodeする必要があり、vae_losses_v2.compute_lossが受け取る
    # decoder_output(再構成側のdecode結果)だけでは完結しないため別枠にしている。
    # lossesは重み乗算前の生の値。個々の損失を追加するたびに(a)このdictへの1行(b)重みが1.0以外
    # ならweightsへの1行、の2箇所を触るだけで済む(vae_losses_v2.compute_lossと同じ設計)
    losses: dict[str, torch.Tensor] = {}
    losses["self_loop_loss"] = _compute_self_loop_loss(model, mu)

    # ここに列挙のない損失は暗黙的に重み1.0として扱う
    weights = {
        "self_loop_loss": SELF_LOOP_LOSS_WEIGHT,
    }
    total = sum(weights.get(name, 1.0) * value for name, value in losses.items())
    return SyntheticLossComponents(total=total, **losses)
