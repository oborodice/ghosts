// 字の変わり方を決める、潜在空間の軌跡(simplex noiseの値)。学習側の training/glyph/walk.py と同じ作り方で、
// 潜在の次元ごとにノイズ場の別の行を、時間とともにx方向へ進みながらたどる。
// ノイズは既存のライブラリ(noise クレートの OpenSimplex)を使う。学習側の opensimplex と同じ種類のノイズで、模様の大きさ
// (近い位置どうしの相関)は同じだが、値の大きさをそろえる定数が違い、値の幅が約0.62倍になる。生成器のONNXの中の正規分布への
// 変換の表は学習側の値の分布に合わせてあるので、倍率を掛けて学習側の分布にそろえる
use noise::{NoiseFn, OpenSimplex};

pub const DISPLAY_FPS: f64 = 30.0;
const DIMENSION_SPACING: f64 = 10.0; // 潜在の次元ごとに、ノイズ場のy方向にこれだけ離した行を使い、次元どうしを無関係にする
// ノイズ場の中を進む速さ(1秒あたり)。学習側で、この速さで字の移り変わりがなめらかなことを確かめた
const NOISE_SPEED: f64 = 0.0633;
// 学習側の値の分布にそろえる倍率。ばらばらの位置の値の分位点を、両方で100万点ずつ測って最小二乗で合わせた
// (合わせたあとの分位点の差は最大0.007。値の範囲は約±0.87)
const SCALE_TO_TRAINING: f64 = 1.6016;

pub struct LatentWalk {
    noise: OpenSimplex,
    latent_dim: usize,
}

impl LatentWalk {
    pub fn new(seed: u32, latent_dim: usize) -> Self {
        Self {
            noise: OpenSimplex::new(seed),
            latent_dim,
        }
    }

    pub fn values(&self, frame: u64) -> Vec<f32> {
        // frame 番目のフレームの、潜在の次元の数ぶんの値
        let x = frame as f64 / DISPLAY_FPS * NOISE_SPEED;
        (0..self.latent_dim)
            .map(|dimension| (SCALE_TO_TRAINING * self.noise.get([x, dimension as f64 * DIMENSION_SPACING])) as f32)
            .collect()
    }
}
