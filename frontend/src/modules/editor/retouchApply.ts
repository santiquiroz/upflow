import { createMaskCanvas } from "./editorMaskCanvas";
import { drawStrokes, type BrushStroke } from "./maskCanvas";
import { cloneWithMask, restoreWithMask } from "./retouch";

export type RetouchKind = "restore" | "clone";

export interface RetouchOffset {
  dx: number;
  dy: number;
}

function loadImage(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("could not decode image"));
    };
    image.src = url;
  });
}

function contextOf(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  return ctx;
}

async function pixelsOf(blob: Blob): Promise<ImageData> {
  const image = await loadImage(blob);
  const canvas = createMaskCanvas(image.naturalWidth, image.naturalHeight);
  const ctx = contextOf(canvas);
  ctx.drawImage(image, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// Pincel suave: el mismo trazo de siempre, con el borde difuminado. Su alfa es
// el peso con que se mezcla cada píxel.
function strokeWeights(stroke: BrushStroke, width: number, height: number): ImageData {
  const canvas = createMaskCanvas(width, height);
  const ctx = contextOf(canvas);
  ctx.filter = `blur(${Math.max(1, stroke.radius * 0.25)}px)`;
  drawStrokes(ctx, [stroke], "#ffffff");
  return ctx.getImageData(0, 0, width, height);
}

function encodePng(pixels: ImageData): Promise<Blob> {
  const canvas = createMaskCanvas(pixels.width, pixels.height);
  contextOf(canvas).putImageData(pixels, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("could not encode the image"));
    }, "image/png");
  });
}

function retouchPixels(
  kind: RetouchKind,
  current: ImageData,
  original: ImageData,
  weights: ImageData,
  offset: RetouchOffset,
): ImageData {
  return kind === "restore"
    ? restoreWithMask(current, original, weights)
    : cloneWithMask(current, weights, offset.dx, offset.dy);
}

// Aplica un trazo sobre la base actual y devuelve el PNG nuevo. No sube nada.
export async function applyRetouch(
  kind: RetouchKind,
  stroke: BrushStroke,
  currentBlob: Blob,
  originalBlob: Blob,
  offset: RetouchOffset,
): Promise<Blob> {
  const current = await pixelsOf(currentBlob);
  const original = kind === "restore" ? await pixelsOf(originalBlob) : current;
  const weights = strokeWeights(stroke, current.width, current.height);
  return encodePng(retouchPixels(kind, current, original, weights, offset));
}
