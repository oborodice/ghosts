// 生成器のONNX(学習側の training/scripts/export_onnx.py で書き出したもの)をONNX Runtime(CPU)で動かし、
// simplex noiseの値から字のインクの画像を作る。採用した生成器は実行ファイルに埋め込み、実行ファイル1つで配れるようにする
use std::path::Path;

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::value::{Tensor, ValueType};

// 学習側で書き出し直したら、組み立て直すだけで新しい生成器になる
const EMBEDDED_MODEL: &[u8] = include_bytes!("../../training/data/onnx/glyph_generator.onnx");

pub struct Generator {
    session: Session,
    pub latent_dim: usize,
    pub image_size: usize,
}

fn tensor_shape(value_type: &ValueType) -> Result<Vec<i64>> {
    match value_type {
        ValueType::Tensor { shape, .. } => Ok(shape.to_vec()),
        other => bail!("expected a tensor, got {other:?}"),
    }
}

impl Generator {
    pub fn load(model: Option<&Path>) -> Result<Self> {
        // modelを指定しないときは、埋め込んだ生成器を使う
        let session = match model {
            Some(model) => Session::builder()?
                .commit_from_file(model)
                .with_context(|| format!("loading {}", model.display()))?,
            None => Session::builder()?
                .commit_from_memory(EMBEDDED_MODEL)
                .context("loading the embedded model")?,
        };
        // 入力は (1, 潜在の次元の数)、出力は (1, 1, 解像度, 解像度)
        let latent_dim = *tensor_shape(session.inputs()[0].dtype())?
            .get(1)
            .context("input has no latent dimension")? as usize;
        let image_size = *tensor_shape(session.outputs()[0].dtype())?.last().context("output has no dimensions")? as usize;
        Ok(Self {
            session,
            latent_dim,
            image_size,
        })
    }

    pub fn ink(&mut self, simplex_values: Vec<f32>) -> Result<Vec<f32>> {
        // 返り値は 解像度 x 解像度 のインクの濃さ(0=紙〜1=インク、上の行から)
        let input = Tensor::from_array(([1, self.latent_dim], simplex_values))?;
        let outputs = self.session.run(ort::inputs!["simplex_values" => input])?;
        let (_, ink) = outputs["ink"].try_extract_tensor::<f32>()?;
        Ok(ink.to_vec())
    }
}
