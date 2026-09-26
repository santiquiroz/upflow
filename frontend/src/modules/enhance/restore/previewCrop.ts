import type { JobResponse } from "../../../lib/apiTypes";
import type { CropBox, RestoreOptions } from "../../../lib/restoreApiTypes";
import type { BoxRect, FractionRect, Point, Size } from "./geometryMath";

// Mismo tope que app/services/photo_restore_pipeline.py (PREVIEW_MAX_SIDE).
export const PREVIEW_MAX_SIDE = 512;
export const MIN_PREVIEW_PX = 16;
// Lado mayor de preview.jpg en app/services/restore_session.py (PREVIEW_MAX_SIDE).
const SESSION_PREVIEW_MAX_SIDE = 2048;
const CLICK_TOLERANCE_PX = 4;
const CROP_LENGTH = 4;

export interface PreviewResult {
  crop: CropBox;
}

export interface SourceRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toWorking(point: Point, box: BoxRect, working: Size): Point {
  const x = clamp((point.x - box.left) / box.width, 0, 1);
  const y = clamp((point.y - box.top) / box.height, 0, 1);
  return { x: Math.round(x * working.width), y: Math.round(y * working.height) };
}

function centeredSpan(center: number, total: number): [start: number, length: number] {
  const length = Math.min(PREVIEW_MAX_SIDE, total);
  return [clamp(Math.round(center - length / 2), 0, total - length), length];
}

function anchoredSpan(anchor: number, end: number): [start: number, length: number] {
  const delta = clamp(end - anchor, -PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE);
  return [Math.min(anchor, anchor + delta), Math.abs(delta)];
}

function isClick(start: Point, end: Point): boolean {
  return Math.hypot(end.x - start.x, end.y - start.y) < CLICK_TOLERANCE_PX;
}

export function centeredPreviewCrop(working: Size): CropBox {
  const [x, width] = centeredSpan(working.width / 2, working.width);
  const [y, height] = centeredSpan(working.height / 2, working.height);
  return [x, y, width, height];
}

function centeredAt(point: Point, working: Size): CropBox {
  const [x, width] = centeredSpan(point.x, working.width);
  const [y, height] = centeredSpan(point.y, working.height);
  return [x, y, width, height];
}

// Un clic pone el area mas grande permitida alrededor; arrastrar la dibuja desde el ancla.
function isMeasurable(start: Point, end: Point, box: BoxRect): boolean {
  const values = [start.x, start.y, end.x, end.y];
  return box.width > 0 && box.height > 0 && values.every(Number.isFinite);
}

export function previewCropFromPoints(start: Point, end: Point, box: BoxRect, working: Size): CropBox | null {
  if (!isMeasurable(start, end, box)) {
    return null;
  }
  const from = toWorking(start, box, working);
  if (isClick(start, end)) {
    return centeredAt(from, working);
  }
  const to = toWorking(end, box, working);
  const [x, width] = anchoredSpan(from.x, to.x);
  const [y, height] = anchoredSpan(from.y, to.y);
  return width < MIN_PREVIEW_PX || height < MIN_PREVIEW_PX ? null : [x, y, width, height];
}

export function cropFraction([x, y, width, height]: CropBox, working: Size): FractionRect {
  return {
    x: x / working.width,
    y: y / working.height,
    width: width / working.width,
    height: height / working.height,
  };
}

// La vista previa de la sesion puede ser mas chica que la copia de trabajo (lado mayor 2048).
export function previewSourceRect([x, y, width, height]: CropBox, working: Size, preview: Size): SourceRect {
  const scaleX = preview.width / working.width;
  const scaleY = preview.height / working.height;
  return { x: x * scaleX, y: y * scaleY, width: width * scaleX, height: height * scaleY };
}

// El "antes" se corta de preview.jpg: solo es de resolucion completa si la sesion no la achico.
export function beforeAtFullResolution(working: Size): boolean {
  return Math.max(working.width, working.height) <= SESSION_PREVIEW_MAX_SIDE;
}

export function withPreviewCrop(options: RestoreOptions, crop: CropBox): RestoreOptions {
  return { ...options, preview_crop: crop };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readCrop(value: unknown): CropBox | null {
  if (!Array.isArray(value) || value.length !== CROP_LENGTH || !value.every((item) => typeof item === "number")) {
    return null;
  }
  const [x, y, width, height] = value as number[];
  return [x, y, width, height];
}

// Una corrida de "Preview this area" deja solo el artefacto preview y el recorte en la metadata.
export function readPreviewResult(job: JobResponse): PreviewResult | null {
  if (job.status !== "completed") {
    return null;
  }
  const restore: unknown = job.metadata?.restore;
  if (!isRecord(restore)) {
    return null;
  }
  const artifacts = Array.isArray(restore.artifacts) ? restore.artifacts : [];
  const crop = readCrop(restore.previewCrop);
  if (!artifacts.includes("preview") || artifacts.includes("view") || crop === null) {
    return null;
  }
  return { crop };
}
