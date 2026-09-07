import { LATENT_DIM } from "./model";

// training/scripts/export_latent_prior.pyが出力する、学習データ全件のencode結果(mu)
const LATENT_PRIOR_URL = "/latent_prior.bin";

// 実データ同士の最近傍距離(中央値0.609)を目安に選んだカーネル幅。値を上げるほど複数の実データを混ぜやすくなるが、
// 上げすぎると多数の実データの平均に寄ってしまい特徴がぼやける
// (幽霊文字の生成モデル(モデル設計・フェーズ2).md の「生成品質(潜在空間の構造)の検証」参照)
const KERNEL_BANDWIDTH = 0.6;

export async function loadLatentPrior(): Promise<Float32Array> {
  const response = await fetch(LATENT_PRIOR_URL);
  const buffer = await response.arrayBuffer();
  return new Float32Array(buffer);
}

// z_rawから全実データ点までの距離をガウシアンカーネルで重み付けし、加重平均を返す(Nadaraya-Watson推定量)。
// 生成用のzを常に実データの近くへ引き寄せることで、z_raw自体が実データから遠い場所にあっても、
// Decoderが訓練中に見たことのない座標をそのまま渡さずに済む
export function attractToLatentPrior(zRaw: Float32Array, priorPoints: Float32Array): Float32Array {
  const pointCount = priorPoints.length / LATENT_DIM;
  const invTwoHH = 1 / (2 * KERNEL_BANDWIDTH * KERNEL_BANDWIDTH);

  // softmaxの数値安定化のため、各点の対数重み(-距離^2 / 2h^2)を求めつつ最大値を記録する
  const logits = new Float32Array(pointCount);
  let maxLogit = -Infinity;
  for (let i = 0; i < pointCount; i++) {
    let distSq = 0;
    const base = i * LATENT_DIM;
    for (let dim = 0; dim < LATENT_DIM; dim++) {
      const diff = zRaw[dim] - priorPoints[base + dim];
      distSq += diff * diff;
    }
    const logit = -distSq * invTwoHH;
    logits[i] = logit;
    if (logit > maxLogit) maxLogit = logit;
  }

  let weightSum = 0;
  for (let i = 0; i < pointCount; i++) {
    const weight = Math.exp(logits[i] - maxLogit);
    logits[i] = weight; // 以降はexp後の重みとして使い回す
    weightSum += weight;
  }

  const zOut = new Float32Array(LATENT_DIM);
  for (let i = 0; i < pointCount; i++) {
    const weight = logits[i] / weightSum;
    const base = i * LATENT_DIM;
    for (let dim = 0; dim < LATENT_DIM; dim++) {
      zOut[dim] += weight * priorPoints[base + dim];
    }
  }
  return zOut;
}
