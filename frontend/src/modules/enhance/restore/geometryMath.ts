import type { CropBox, RestoreGeometry } from "../../../lib/restoreApiTypes";

// Mismo tope que app/services/photo_geometry.py (MAX_STRAIGHTEN_DEG).
export const MAX_STRAIGHTEN_DEG = 45;
export const MIN_CROP_PX = 16;
const ANGLE_STEP = 10;

export interface FractionRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface BoxRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface Size {
  width: number;
  height: number;
}

const EMPTY_SELECTION: FractionRect = { x: 0, y: 0, width: 0, height: 0 };

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function rotateClockwise(geometry: RestoreGeometry): RestoreGeometry {
  return { ...geometry, rotate90: (geometry.rotate90 + 1) % 4, crop: null };
}

export function withAngle(geometry: RestoreGeometry, angle: number): RestoreGeometry {
  const clamped = clamp(angle, -MAX_STRAIGHTEN_DEG, MAX_STRAIGHTEN_DEG);
  return { ...geometry, angle: Math.round(clamped * ANGLE_STEP) / ANGLE_STEP };
}

export function withoutCrop(geometry: RestoreGeometry): RestoreGeometry {
  return { ...geometry, crop: null };
}

export function isNeutralGeometry(geometry: RestoreGeometry): boolean {
  return geometry.rotate90 === 0 && geometry.crop === null && geometry.angle === 0;
}

function fraction(value: number, start: number, length: number): number {
  return clamp((value - start) / length, 0, 1);
}

export function selectionFromPoints(start: Point, end: Point, box: BoxRect): FractionRect {
  if (box.width <= 0 || box.height <= 0) {
    return EMPTY_SELECTION;
  }
  const x0 = fraction(Math.min(start.x, end.x), box.left, box.width);
  const x1 = fraction(Math.max(start.x, end.x), box.left, box.width);
  const y0 = fraction(Math.min(start.y, end.y), box.top, box.height);
  const y1 = fraction(Math.max(start.y, end.y), box.top, box.height);
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

function pixelSpan(start: number, length: number, total: number): [number, number] {
  const from = Math.round(start * total);
  const to = Math.min(total, Math.round((start + length) * total));
  return [from, to - from];
}

// El recorte se mide en la imagen ya girada; la vista previa muestra la copia de
// trabajo, que puede venir recortada, asi que se suma el recorte vigente.
export function cropFromSelection(
  geometry: RestoreGeometry,
  selection: FractionRect,
  working: Size,
): RestoreGeometry | null {
  const [x, width] = pixelSpan(selection.x, selection.width, working.width);
  const [y, height] = pixelSpan(selection.y, selection.height, working.height);
  if (width < MIN_CROP_PX || height < MIN_CROP_PX) {
    return null;
  }
  const [offsetX, offsetY] = geometry.crop ?? [0, 0];
  const crop: CropBox = [offsetX + x, offsetY + y, width, height];
  return { ...geometry, crop };
}

// cv2 gira en sentido antihorario con angulos positivos; CSS lo hace al reves.
export function previewRotationDeg(appliedAngle: number, draftAngle: number): number {
  return appliedAngle - draftAngle;
}
