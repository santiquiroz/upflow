import { drawStrokes, strokeAddsEditableArea, type BrushStroke } from "./maskCanvas";
import { maskHasPixels, maskToOpaquePixels, subtractProtectedPixels } from "./protectMask";

export interface HistoryEntry {
  kind: "stroke" | "segment" | "protect" | "clearProtection" | "batch";
  stroke?: BrushStroke;
  mask?: HTMLCanvasElement;
  point?: { x: number; y: number };
  entries?: HistoryEntry[];
  personIds?: number[];
}

export function flattenHistory(history: HistoryEntry[]): HistoryEntry[] {
  return history.flatMap((entry) => entry.kind === "batch" ? flattenHistory(entry.entries ?? []) : [entry]);
}

export function withoutEdits(history: HistoryEntry[]): HistoryEntry[] {
  return flattenHistory(history).filter((entry) => entry.kind === "protect" || entry.kind === "clearProtection");
}

export function createMaskCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function pngBlobToAlphaCanvas(blob: Blob, width: number, height: number): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = createMaskCanvas(width, height);
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("canvas 2d context unavailable"));
        return;
      }
      ctx.drawImage(image, 0, 0, width, height);
      const pixels = ctx.getImageData(0, 0, width, height);
      for (let i = 0; i < pixels.data.length; i += 4) {
        pixels.data[i + 3] = pixels.data[i];
        pixels.data[i] = 255;
        pixels.data[i + 1] = 255;
        pixels.data[i + 2] = 255;
      }
      ctx.putImageData(pixels, 0, 0);
      resolve(canvas);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("could not decode mask"));
    };
    image.src = url;
  });
}

export function replayMasks(
  edit: HTMLCanvasElement,
  protect: HTMLCanvasElement,
  history: HistoryEntry[],
  live: BrushStroke | null,
): void {
  const editCtx = edit.getContext("2d");
  const protectCtx = protect.getContext("2d");
  if (!editCtx || !protectCtx) return;
  for (const ctx of [editCtx, protectCtx]) {
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, edit.width, edit.height);
  }
  const flat = flattenHistory(history);
  for (const entry of flat) {
    if (entry.kind === "clearProtection") {
      protectCtx.clearRect(0, 0, protect.width, protect.height);
      continue;
    }
    if (entry.kind === "protect" && entry.mask) {
      protectCtx.drawImage(entry.mask, 0, 0);
      continue;
    }
    if (entry.kind === "segment" && entry.mask) {
      editCtx.drawImage(entry.mask, 0, 0);
      continue;
    }
    if (entry.stroke) drawStrokes(editCtx, [entry.stroke], "#ffffff");
  }
  resolveOverlap(edit, editCtx, protectCtx, activeProtections(flat));
  if (live) drawStrokes(editCtx, [live], "#ffffff");
}

function activeProtections(history: HistoryEntry[]): HistoryEntry[] {
  const lastClear = history.map((entry) => entry.kind).lastIndexOf("clearProtection");
  return history.slice(lastClear + 1).filter((entry) => entry.kind === "protect" && entry.mask);
}

function coversPoint(ctx: CanvasRenderingContext2D, point: { x: number; y: number } | undefined): boolean {
  if (!point) return false;
  return ctx.getImageData(Math.floor(point.x), Math.floor(point.y), 1, 1).data[3] > 0;
}

function eraseWith(ctx: CanvasRenderingContext2D, mask: HTMLCanvasElement): void {
  ctx.globalCompositeOperation = "destination-out";
  ctx.drawImage(mask, 0, 0);
  ctx.globalCompositeOperation = "source-over";
}

// Medido en 59 fotos de grado: un segmento que contiene el toque de una protección se tragó a ese
// sujeto (se le devuelve); en el resto gana el borrado, porque una protección que se pasa de largo
// se traga a alguien del fondo y con "la protección siempre gana" esa persona no se podía borrar.
function resolveOverlap(
  edit: HTMLCanvasElement,
  editCtx: CanvasRenderingContext2D,
  protectCtx: CanvasRenderingContext2D,
  protections: HistoryEntry[],
): void {
  if (protections.length === 0) return;
  protections
    .filter((entry) => coversPoint(editCtx, entry.point))
    .forEach((entry) => eraseWith(editCtx, entry.mask!));
  eraseWith(protectCtx, edit);
}

export function maskHasContent(history: HistoryEntry[]): boolean {
  return flattenHistory(history).some((entry) => entry.kind === "segment" || strokeAddsEditableArea(entry.stroke));
}

export function hasProtection(history: HistoryEntry[]): boolean {
  const flat = flattenHistory(history);
  const lastClear = flat.map((entry) => entry.kind).lastIndexOf("clearProtection");
  return flat.slice(lastClear + 1).some((entry) => entry.kind === "protect");
}

export function canvasHasPixels(canvas: HTMLCanvasElement): boolean {
  const ctx = canvas.getContext("2d");
  return Boolean(ctx && maskHasPixels(ctx.getImageData(0, 0, canvas.width, canvas.height).data));
}

function tintMask(canvas: HTMLCanvasElement, color: string): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "source-over";
}

export function renderMaskOverlays(
  overlay: HTMLCanvasElement,
  protectionOverlay: HTMLCanvasElement,
  edit: HTMLCanvasElement,
  protect: HTMLCanvasElement,
): void {
  const ctx = overlay.getContext("2d");
  const protectCtx = protect.getContext("2d");
  const editCtx = edit.getContext("2d");
  const greenCtx = protectionOverlay.getContext("2d");
  if (!ctx || !protectCtx || !editCtx || !greenCtx) return;
  const pixels = editCtx.getImageData(0, 0, edit.width, edit.height);
  const protectedPixels = protectCtx.getImageData(0, 0, protect.width, protect.height);
  pixels.data.set(subtractProtectedPixels(pixels.data, protectedPixels.data));
  ctx.putImageData(pixels, 0, 0);
  tintMask(overlay, "#ef4444");
  greenCtx.clearRect(0, 0, protect.width, protect.height);
  greenCtx.drawImage(protect, 0, 0);
  tintMask(protectionOverlay, "#22c55e");
}

export function exportMaskBlob(maskCanvas: HTMLCanvasElement): Promise<Blob> {
  const exportCanvas = createMaskCanvas(maskCanvas.width, maskCanvas.height);
  const ctx = exportCanvas.getContext("2d");
  if (!ctx) return Promise.reject(new Error("canvas 2d context unavailable"));
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
  ctx.drawImage(maskCanvas, 0, 0);
  return encodeMaskPng(exportCanvas);
}

export function exportProtectMaskBlob(maskCanvas: HTMLCanvasElement): Promise<Blob> {
  const exportCanvas = createMaskCanvas(maskCanvas.width, maskCanvas.height);
  const sourceCtx = maskCanvas.getContext("2d");
  const ctx = exportCanvas.getContext("2d");
  if (!ctx || !sourceCtx) return Promise.reject(new Error("canvas 2d context unavailable"));
  const pixels = sourceCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  pixels.data.set(maskToOpaquePixels(pixels.data));
  ctx.putImageData(pixels, 0, 0);
  return encodeMaskPng(exportCanvas);
}

export function readMaskImageData(canvas: HTMLCanvasElement): ImageData | null {
  return canvas.getContext("2d")?.getImageData(0, 0, canvas.width, canvas.height) ?? null;
}

export function exportImageDataMaskBlob(mask: ImageData): Promise<Blob> {
  const canvas = createMaskCanvas(mask.width, mask.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.reject(new Error("canvas 2d context unavailable"));
  ctx.putImageData(mask, 0, 0);
  return exportMaskBlob(canvas);
}

function encodeMaskPng(exportCanvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    exportCanvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("could not export the mask"));
    }, "image/png");
  });
}
