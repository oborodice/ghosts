import * as ort from "onnxruntime-web";

// training/scripts/export_onnx.pyが出力するVAEデコーダのONNXモデル
const MODEL_URL = "/vae.onnx";

export const LATENT_DIM = 32; // training/scripts/vae_model.pyのLATENT_DIMと同じ値

export interface GenerationResult {
  strokes: Float32Array;
  existenceProb: Float32Array;
}

export function loadModel(): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(MODEL_URL);
}

export async function generate(
  session: ort.InferenceSession,
  z: Float32Array,
): Promise<GenerationResult> {
  // 1文字ずつ生成する用途のため、バッチサイズは常に1とする(モデル自体は可変バッチに対応している)
  const zTensor = new ort.Tensor("float32", z, [1, LATENT_DIM]);
  const outputs = await session.run({ z: zTensor });
  return {
    strokes: outputs.strokes.data as Float32Array,
    existenceProb: outputs.existence_prob.data as Float32Array,
  };
}
