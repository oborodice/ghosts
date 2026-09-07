import "./style.css";
import { LATENT_DIM, generate, loadModel } from "./model";

async function main(): Promise<void> {
  const session = await loadModel();
  console.log("Loaded ONNX model.", { inputs: session.inputNames, outputs: session.outputNames });

  // 読み込んだモデルが実際に推論できるか確認するため、ゼロベクトルを1回流してみる
  const { strokes, existenceProb } = await generate(session, new Float32Array(LATENT_DIM));
  console.log("strokes:", strokes);
  console.log("existence_prob:", existenceProb);
}

main();
