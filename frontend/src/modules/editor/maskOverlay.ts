import { createMaskCanvas, type CommittedMasks, type EdgeAdjust } from "./editorMaskCanvas";
import { drawStrokes, type BrushStroke } from "./maskCanvas";

const OVERLAY_RED = [239, 68, 68];
const GROW_ALPHA = 56;
const SHRINK_ALPHA = 200;
const EDIT_TINT = "#ef4444";
const PROTECT_TINT = "#22c55e";

export interface OverlayLayers {
  width: number;
  height: number;
  edit: HTMLCanvasElement;
  protect: HTMLCanvasElement;
}

function contextOf(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  return canvas.getContext("2d");
}

function copyCanvas(source: HTMLCanvasElement, width: number, height: number): HTMLCanvasElement {
  const copy = createMaskCanvas(width, height);
  contextOf(copy)?.drawImage(source, 0, 0, width, height);
  return copy;
}

function blurCanvas(source: HTMLCanvasElement, radiusPx: number): HTMLCanvasElement {
  const blurred = createMaskCanvas(source.width, source.height);
  const ctx = contextOf(blurred);
  if (ctx) {
    ctx.filter = `blur(${radiusPx}px)`;
    ctx.drawImage(source, 0, 0);
  }
  return blurred;
}

function thresholdAlpha(source: HTMLCanvasElement, minAlpha: number): HTMLCanvasElement {
  const out = createMaskCanvas(source.width, source.height);
  const outCtx = contextOf(out);
  const pixels = contextOf(source)?.getImageData(0, 0, source.width, source.height);
  if (!outCtx || !pixels) return out;
  for (let i = 0; i < pixels.data.length; i += 4) {
    pixels.data.set(OVERLAY_RED, i);
    pixels.data[i + 3] = pixels.data[i + 3] >= minAlpha ? 255 : 0;
  }
  outCtx.putImageData(pixels, 0, 0);
  return out;
}

function usableScale(scale: number): number | null {
  return Number.isFinite(scale) && scale > 0 ? Math.min(scale, 1) : null;
}

export function previewAdjust(canvas: HTMLCanvasElement, growPx: number, featherPx: number, rawScale: number): void {
  if (growPx === 0 && featherPx === 0) return;
  const scale = usableScale(rawScale);
  if (scale === null) return;
  const width = Math.round(canvas.width * scale);
  const height = Math.round(canvas.height * scale);
  if (width <= 0 || height <= 0) return;
  const ctx = contextOf(canvas);
  if (!ctx || !("filter" in ctx)) return;
  let shape = copyCanvas(canvas, width, height);
  if (growPx !== 0) {
    shape = thresholdAlpha(blurCanvas(shape, Math.abs(growPx) * scale), growPx > 0 ? GROW_ALPHA : SHRINK_ALPHA);
  }
  if (featherPx !== 0) shape = blurCanvas(shape, featherPx * scale);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(shape, 0, 0, canvas.width, canvas.height);
}

function tintMask(canvas: HTMLCanvasElement, color: string): void {
  const ctx = contextOf(canvas);
  if (!ctx) return;
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "source-over";
}

// Tamaño del overlay en píxeles de pantalla: el ancho visible por el devicePixelRatio, nunca más que la foto.
export function displaySize(element: HTMLElement, photoWidth: number, photoHeight: number): { width: number; height: number } {
  const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;
  const cssWidth = element.clientWidth;
  const width = cssWidth > 0 ? Math.min(photoWidth, Math.max(1, Math.round(cssWidth * dpr))) : photoWidth;
  return { width, height: Math.max(1, Math.round((width * photoHeight) / photoWidth)) };
}

// Las capas de pantalla se calculan una vez por historial: pintar un trazo no vuelve a escalar 24 MP.
export function overlayLayers(masks: CommittedMasks, width: number, height: number): OverlayLayers {
  return {
    width,
    height,
    edit: copyCanvas(masks.edit, width, height),
    protect: thresholdAlpha(copyCanvas(masks.protect, width, height), 1),
  };
}

function sizeCanvas(canvas: HTMLCanvasElement, layers: OverlayLayers): void {
  if (canvas.width !== layers.width) canvas.width = layers.width;
  if (canvas.height !== layers.height) canvas.height = layers.height;
}

function paintLiveStroke(ctx: CanvasRenderingContext2D, stroke: BrushStroke, scale: number): void {
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  drawStrokes(ctx, [stroke], "#ffffff");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function paintEditLayer(red: HTMLCanvasElement, layers: OverlayLayers, live: BrushStroke | null, scale: number): void {
  const ctx = contextOf(red);
  if (!ctx) return;
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, layers.width, layers.height);
  ctx.drawImage(layers.edit, 0, 0);
  if (live) paintLiveStroke(ctx, live, scale);
  ctx.globalCompositeOperation = "destination-out";
  ctx.drawImage(layers.protect, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  tintMask(red, EDIT_TINT);
}

function paintProtectLayer(green: HTMLCanvasElement, layers: OverlayLayers): void {
  const ctx = contextOf(green);
  if (!ctx) return;
  ctx.clearRect(0, 0, layers.width, layers.height);
  ctx.drawImage(layers.protect, 0, 0);
  tintMask(green, PROTECT_TINT);
}

export function paintOverlay(
  red: HTMLCanvasElement,
  green: HTMLCanvasElement,
  layers: OverlayLayers,
  live: BrushStroke | null,
  photoWidth: number,
  edge: EdgeAdjust | null,
): void {
  sizeCanvas(red, layers);
  sizeCanvas(green, layers);
  const scale = layers.width / photoWidth;
  paintEditLayer(red, layers, live, scale);
  if (edge) previewAdjust(red, edge.growPx * scale, edge.featherPx * scale, 1);
  paintProtectLayer(green, layers);
}
