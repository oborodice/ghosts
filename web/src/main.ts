import "./style.css";
import { LATENT_DIM, generate, loadModel } from "./model";
import { drawStrokes } from "./render";

function getCanvasContext(): CanvasRenderingContext2D {
  const canvas = document.querySelector<HTMLCanvasElement>("#kanji-canvas")!;
  return canvas.getContext("2d")!;
}

async function main(): Promise<void> {
  const session = await loadModel();
  const ctx = getCanvasContext();

  const result = await generate(session, new Float32Array(LATENT_DIM));
  drawStrokes(ctx, result);
}

main();
