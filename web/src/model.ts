import * as ort from "onnxruntime-web";

// training/scripts/export_onnx.pyが出力するVAEデコーダのONNXモデル
const MODEL_URL = "/vae_phase1.onnx";

export const LATENT_DIM = 32;

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
  const outputs = await session.run({ z: new ort.Tensor("float32", z, [1, LATENT_DIM]) });
  return {
    strokes: outputs.strokes.data as Float32Array,
    existenceProb: outputs.existence_prob.data as Float32Array,
  };
}
