import type { GenerationResult } from "./model";

const EXISTENCE_THRESHOLD = 0.5; // training/scripts/vae_eval_common.pyのEXISTENCE_THRESHOLDと同じ値
const FEATURE_COUNT = 5; // start_x, start_y, angle, curvature, length
const KANJI_VIEWBOX_SIZE = 109; // KanjiVGのSVGのviewBoxサイズ(training/data/kanjivg/*.svg参照)

interface StrokeSegment {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

function strokesToSegments(strokes: Float32Array, existenceProb: Float32Array): StrokeSegment[] {
  const segments: StrokeSegment[] = [];
  for (let slot = 0; slot < existenceProb.length; slot++) {
    if (existenceProb[slot] <= EXISTENCE_THRESHOLD) continue;

    const offset = slot * FEATURE_COUNT;
    const startX = strokes[offset];
    const startY = strokes[offset + 1];
    const angle = strokes[offset + 2];
    const curvature = strokes[offset + 3];
    const length = strokes[offset + 4];
    const radius = length - curvature;

    segments.push({
      startX,
      startY,
      endX: startX + radius * Math.cos(angle),
      endY: startY + radius * Math.sin(angle),
    });
  }
  return segments;
}

function drawSegments(ctx: CanvasRenderingContext2D, segments: StrokeSegment[]): void {
  const scale = ctx.canvas.width / KANJI_VIEWBOX_SIZE;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "black";
  ctx.lineWidth = 2;

  // KanjiVGのSVGはy軸が下向きで、canvasも同じ向きのため反転は不要(view_kanji.pyのmatplotlib向け反転とは対照的)
  for (const { startX, startY, endX, endY } of segments) {
    ctx.beginPath();
    ctx.moveTo(startX * scale, startY * scale);
    ctx.lineTo(endX * scale, endY * scale);
    ctx.stroke();
  }
}

export function drawStrokes(
  ctx: CanvasRenderingContext2D,
  { strokes, existenceProb }: GenerationResult,
): void {
  drawSegments(ctx, strokesToSegments(strokes, existenceProb));
}
