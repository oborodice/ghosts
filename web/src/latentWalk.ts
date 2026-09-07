import { createNoise2D } from "simplex-noise";

import { LATENT_DIM } from "./model";

// 潜在次元ごとに独立したノイズ場を使うことで、各次元が互いに無関係に変化するようにする
const noises = Array.from({ length: LATENT_DIM }, () => createNoise2D());

// 1文字がおよそ5秒で変容する速さを初期値とする(幽霊文字の生成モデル(モデル設計・フェーズ2).mdの「パーリンノイズ」参照)
const SPEED = 1 / 5;

// simplex-noiseの出力はおおよそ[-1, 1]だが、学習時の事前分布はN(0, 1)であり本来もっと広い範囲を取りうるため、
// 潜在空間をある程度の範囲まで探索できるようこの倍率で拡大する
const Z_SCALE = 2;

export function latentAt(timeSeconds: number): Float32Array {
  const z = new Float32Array(LATENT_DIM);
  for (let dim = 0; dim < LATENT_DIM; dim++) {
    // simplex-noiseは1次元のノイズを提供していないため、2次元目を固定して1次元分の滑らかな連続値として使う
    z[dim] = noises[dim](timeSeconds * SPEED, 0) * Z_SCALE;
  }
  return z;
}
