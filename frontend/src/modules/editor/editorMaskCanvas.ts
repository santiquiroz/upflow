import { drawStrokes, strokeAddsEditableArea, type BrushPoint, type BrushStroke } from "./maskCanvas";
import { maskHasPixels, maskToOpaquePixels } from "./protectMask";

export interface HistoryEntry {
  id?: number;
  kind: "stroke" | "segment" | "protect" | "clearProtection" | "batch";
  stroke?: BrushStroke;
  mask?: HTMLCanvasElement;
  origin?: BrushPoint;
  point?: BrushPoint;
  points?: BrushPoint[];
  entries?: HistoryEntry[];
  personIds?: number[];
  failed?: number;
}

export interface PixelRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface CropMask {
  mask: HTMLCanvasElement;
  origin: BrushPoint;
}

export interface CommittedMasks {
  edit: HTMLCanvasElement;
  protect: HTMLCanvasElement;
}

export function flattenHistory(history: HistoryEntry[]): HistoryEntry[] {
  return history.flatMap((entry) => entry.kind === "batch" ? flattenHistory(entry.entries ?? []) : [entry]);
}

export function withoutEdits(history: HistoryEntry[]): HistoryEntry[] {
  return flattenHistory(history).filter((entry) => entry.kind === "protect" || entry.kind === "clearProtection");
}

export interface EdgeAdjust {
  growPx: number;
  featherPx: number;
}

export const NO_EDGE: EdgeAdjust = { growPx: 0, featherPx: 0 };

export function createMaskCanvas(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function contextOf(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  return ctx;
}

function loadMaskImage(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("could not decode mask"));
    };
    image.src = url;
  });
}

function toAlphaPixels(pixels: ImageData): ImageData {
  for (let i = 0; i < pixels.data.length; i += 4) {
    pixels.data[i + 3] = pixels.data[i];
    pixels.data[i] = 255;
    pixels.data[i + 1] = 255;
    pixels.data[i + 2] = 255;
  }
  return pixels;
}

// Decodifica solo el recorte: la máscara de una persona no ocupa el lienzo completo de la foto.
export async function pngBlobToCropMask(
  blob: Blob,
  width: number,
  height: number,
  rect: PixelRect,
): Promise<CropMask> {
  const image = await loadMaskImage(blob);
  const mask = createMaskCanvas(rect.width, rect.height);
  const ctx = contextOf(mask);
  ctx.drawImage(image, -rect.x, -rect.y, width, height);
  ctx.putImageData(toAlphaPixels(ctx.getImageData(0, 0, rect.width, rect.height)), 0, 0);
  return { mask, origin: { x: rect.x, y: rect.y } };
}

export async function pngBlobToAlphaCanvas(blob: Blob, width: number, height: number): Promise<HTMLCanvasElement> {
  const { mask } = await pngBlobToCropMask(blob, width, height, { x: 0, y: 0, width, height });
  return mask;
}

function drawEntryMask(ctx: CanvasRenderingContext2D, entry: HistoryEntry): void {
  ctx.drawImage(entry.mask!, entry.origin?.x ?? 0, entry.origin?.y ?? 0);
}

function eraseEntry(ctx: CanvasRenderingContext2D, entry: HistoryEntry): void {
  ctx.globalCompositeOperation = "destination-out";
  drawEntryMask(ctx, entry);
  ctx.globalCompositeOperation = "source-over";
}

export function replayMasks(edit: HTMLCanvasElement, protect: HTMLCanvasElement, history: HistoryEntry[]): void {
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
      drawEntryMask(protectCtx, entry);
      continue;
    }
    if (entry.kind === "segment" && entry.mask) {
      drawEntryMask(editCtx, entry);
      continue;
    }
    if (entry.stroke) drawStrokes(editCtx, [entry.stroke], "#ffffff");
  }
  resolveOverlap(edit, editCtx, protectCtx, activeProtections(flat));
}

export function composeCommittedMasks(history: HistoryEntry[], width: number, height: number): CommittedMasks {
  const masks = { edit: createMaskCanvas(width, height), protect: createMaskCanvas(width, height) };
  replayMasks(masks.edit, masks.protect, history);
  return masks;
}

function activeProtections(history: HistoryEntry[]): HistoryEntry[] {
  const lastClear = history.map((entry) => entry.kind).lastIndexOf("clearProtection");
  return history.slice(lastClear + 1).filter((entry) => entry.kind === "protect" && entry.mask);
}

function touchesEntry(ctx: CanvasRenderingContext2D, entry: HistoryEntry): boolean {
  const points = entry.points ?? (entry.point ? [entry.point] : []);
  return points.some((point) => ctx.getImageData(Math.floor(point.x), Math.floor(point.y), 1, 1).data[3] > 0);
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
    .filter((entry) => touchesEntry(editCtx, entry))
    .forEach((entry) => eraseEntry(editCtx, entry));
  protectCtx.globalCompositeOperation = "destination-out";
  protectCtx.drawImage(edit, 0, 0);
  protectCtx.globalCompositeOperation = "source-over";
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
