import type { GenerationResult } from "./model";

const EXISTENCE_THRESHOLD = 0.5; // training/scripts/vae_eval_common.pyのEXISTENCE_THRESHOLDと同じ値
const FEATURE_COUNT = 6; // start_x, start_y, angle, length, offset_x, offset_y
const KANJI_VIEWBOX_SIZE = 109; // KanjiVGのSVGのviewBoxサイズ(training/data/kanjivg/*.svg参照)

interface StrokeCurve {
  startX: number;
  startY: number;
  controlX: number;
  controlY: number;
  endX: number;
  endY: number;
}

function strokesToCurves(strokes: Float32Array, existenceProb: Float32Array): StrokeCurve[] {
  const curves: StrokeCurve[] = [];
  for (let slot = 0; slot < existenceProb.length; slot++) {
    if (existenceProb[slot] <= EXISTENCE_THRESHOLD) continue;

    const offset = slot * FEATURE_COUNT;
    const startX = strokes[offset];
    const startY = strokes[offset + 1];
    const angle = strokes[offset + 2];
    const length = strokes[offset + 3];
    const offsetX = strokes[offset + 4];
    const offsetY = strokes[offset + 5];
    const endX = startX + length * Math.cos(angle);
    const endY = startY + length * Math.sin(angle);

    curves.push({
      startX,
      startY,
      controlX: (startX + endX) / 2 + offsetX,
      controlY: (startY + endY) / 2 + offsetY,
      endX,
      endY,
    });
  }
  return curves;
}

function drawCurves(ctx: CanvasRenderingContext2D, curves: StrokeCurve[]): void {
  const scale = ctx.canvas.width / KANJI_VIEWBOX_SIZE;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;

  // KanjiVGのSVGはy軸が下向きで、canvasも同じ向きのため反転は不要(view_kanji.pyのmatplotlib向け反転とは対照的)
  for (const { startX, startY, controlX, controlY, endX, endY } of curves) {
    ctx.beginPath();
    ctx.moveTo(startX * scale, startY * scale);
    ctx.quadraticCurveTo(controlX * scale, controlY * scale, endX * scale, endY * scale);
    ctx.stroke();
  }
}

export function drawStrokes(
  ctx: CanvasRenderingContext2D,
  { strokes, existenceProb }: GenerationResult,
): void {
  drawCurves(ctx, strokesToCurves(strokes, existenceProb));
}
