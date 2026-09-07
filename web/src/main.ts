import "./style.css";
import { attractToLatentPrior, loadLatentPrior } from "./latentPrior";
import { latentAt } from "./latentWalk";
import { generate, loadModel } from "./model";
import { drawStrokes } from "./render";

function getCanvasContext(): CanvasRenderingContext2D {
  const canvas = document.querySelector<HTMLCanvasElement>("#kanji-canvas")!;
  return canvas.getContext("2d")!;
}

async function main(): Promise<void> {
  const [session, latentPrior] = await Promise.all([loadModel(), loadLatentPrior()]);
  const ctx = getCanvasContext();

  // 前フレームの推論が終わるまで次のrequestAnimationFrameを呼ばないため、フレームが重なって溜まることはない
  async function renderFrame(timeMs: number): Promise<void> {
    const z = attractToLatentPrior(latentAt(timeMs / 1000), latentPrior);
    const result = await generate(session, z);
    drawStrokes(ctx, result);
    requestAnimationFrame(renderFrame);
  }
  requestAnimationFrame(renderFrame);
}

main();
