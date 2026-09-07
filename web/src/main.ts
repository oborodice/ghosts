import "./style.css";
import { latentAt } from "./latentWalk";
import { generate, loadModel } from "./model";
import { drawStrokes } from "./render";

function getCanvasContext(): CanvasRenderingContext2D {
  const canvas = document.querySelector<HTMLCanvasElement>("#kanji-canvas")!;
  return canvas.getContext("2d")!;
}

async function main(): Promise<void> {
  const session = await loadModel();
  const ctx = getCanvasContext();

  // 前フレームの推論が終わるまで次のrequestAnimationFrameを呼ばないため、フレームが重なって溜まることはない
  async function renderFrame(timeMs: number): Promise<void> {
    const result = await generate(session, latentAt(timeMs / 1000));
    drawStrokes(ctx, result);
    requestAnimationFrame(renderFrame);
  }
  requestAnimationFrame(renderFrame);
}

main();
