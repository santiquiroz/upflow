import { createMaskCanvas, type PixelRect } from "./editorMaskCanvas";
import { drawStrokes, type BrushStroke } from "./maskCanvas";
import { cloneFromSource, restoreWithMask } from "./retouch";

export type RetouchKind = "restore" | "clone";

export interface RetouchOffset {
  dx: number;
  dy: number;
}

// Una base decodificada se reutiliza en todos los trazos de su versión: el blob es la clave y el GC la suelta.
const decodedBases = new WeakMap<Blob, Promise<ImageBitmap>>();

function bitmapOf(blob: Blob): Promise<ImageBitmap> {
  let decoded = decodedBases.get(blob);
  if (!decoded) {
    decoded = createImageBitmap(blob).catch(() => {
      throw new Error("could not decode image");
    });
    decodedBases.set(blob, decoded);
  }
  return decoded;
}

function contextOf(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  return ctx;
}

function clipRect(rect: PixelRect, width: number, height: number): PixelRect {
  const left = Math.max(0, rect.x);
  const top = Math.max(0, rect.y);
  const right = Math.min(width, rect.x + rect.width);
  const bottom = Math.min(height, rect.y + rect.height);
  return { x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

// Caja del trazo dilatada por el radio del pincel: lo único que se procesa de la foto.
function strokeBox(stroke: BrushStroke, width: number, height: number): PixelRect {
  const margin = Math.ceil(stroke.radius);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const { x, y } of stroke.points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const left = Math.floor(minX - margin);
  const top = Math.floor(minY - margin);
  return clipRect({ x: left, y: top, width: Math.ceil(maxX + margin) - left, height: Math.ceil(maxY + margin) - top }, width, height);
}

function regionPixels(bitmap: ImageBitmap, rect: PixelRect): ImageData {
  const canvas = createMaskCanvas(rect.width, rect.height);
  const ctx = contextOf(canvas);
  ctx.drawImage(bitmap, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
  return ctx.getImageData(0, 0, rect.width, rect.height);
}

function shiftedStroke(stroke: BrushStroke, rect: PixelRect): BrushStroke {
  return { ...stroke, points: stroke.points.map((point) => ({ x: point.x - rect.x, y: point.y - rect.y })) };
}

// Pincel suave: el mismo trazo de siempre, con el borde difuminado. Su alfa es el peso con que se mezcla cada píxel.
function strokeWeights(stroke: BrushStroke, rect: PixelRect): ImageData {
  const canvas = createMaskCanvas(rect.width, rect.height);
  const ctx = contextOf(canvas);
  ctx.filter = `blur(${Math.max(1, stroke.radius * 0.25)}px)`;
  drawStrokes(ctx, [shiftedStroke(stroke, rect)], "#ffffff");
  return ctx.getImageData(0, 0, rect.width, rect.height);
}

function restoreRegion(current: ImageBitmap, original: ImageBitmap, rect: PixelRect, weights: ImageData): ImageData {
  return restoreWithMask(regionPixels(current, rect), regionPixels(original, rect), weights);
}

function cloneRegion(current: ImageBitmap, rect: PixelRect, weights: ImageData, offset: RetouchOffset): ImageData {
  const dx = Math.round(offset.dx);
  const dy = Math.round(offset.dy);
  const target = regionPixels(current, rect);
  const sourceRect = clipRect({ x: rect.x + dx, y: rect.y + dy, width: rect.width, height: rect.height }, current.width, current.height);
  if (sourceRect.width === 0 || sourceRect.height === 0) return target;
  const source = regionPixels(current, sourceRect);
  return cloneFromSource(target, weights, source, { x: rect.x + dx - sourceRect.x, y: rect.y + dy - sourceRect.y });
}

// La foto completa se copia una vez, se le pega el recorte y se codifica: el único trabajo por píxel es el recorte.
function encodeComposite(base: ImageBitmap, rect: PixelRect, pixels: ImageData): Promise<Blob> {
  const canvas = createMaskCanvas(base.width, base.height);
  const ctx = contextOf(canvas);
  ctx.drawImage(base, 0, 0);
  ctx.putImageData(pixels, rect.x, rect.y);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("could not encode the image"));
    }, "image/png");
  });
}

// Aplica un trazo sobre la base actual y devuelve el PNG nuevo. No sube nada.
export async function applyRetouch(
  kind: RetouchKind,
  stroke: BrushStroke,
  currentBlob: Blob,
  originalBlob: Blob,
  offset: RetouchOffset,
): Promise<Blob> {
  const current = await bitmapOf(currentBlob);
  const original = kind === "restore" ? await bitmapOf(originalBlob) : current;
  const rect = strokeBox(stroke, current.width, current.height);
  const weights = strokeWeights(stroke, rect);
  const pixels = kind === "restore"
    ? restoreRegion(current, original, rect, weights)
    : cloneRegion(current, rect, weights, offset);
  return encodeComposite(current, rect, pixels);
}
