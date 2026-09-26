import type { CctvBox } from "../../../services/cctv";

export interface FrameSize {
  width: number;
  height: number;
}

export interface FramePoint {
  x: number;
  y: number;
}

export interface SurfaceRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface OsdCheckResult {
  box: readonly number[];
  looksLikeText: boolean;
}

// Igual que osd_check.MAX_OSD_BOXES del backend.
export const MAX_OSD_BOXES = 8;
export const MAX_ROI_BOXES = 1;
const MIN_SIDE = 2;
const STEP_PX = 2;
// Posiciones por defecto de Hikvision: fecha y hora arriba a la izquierda, camara abajo a la derecha.
const OSD_MARGIN = 0.02;
const OSD_HEIGHT = 0.06;
const OSD_CLOCK_WIDTH = 0.4;
const OSD_CAMERA_WIDTH = 0.25;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function evenFloor(value: number): number {
  return Math.floor(value / 2) * 2;
}

function evenSide(fraction: number, total: number): number {
  return Math.max(MIN_SIDE, evenFloor(fraction * total));
}

export function suggestedOsdBoxes(frame: FrameSize): CctvBox[] {
  const marginX = Math.round(frame.width * OSD_MARGIN);
  const marginY = Math.round(frame.height * OSD_MARGIN);
  const height = evenSide(OSD_HEIGHT, frame.height);
  const clockWidth = evenSide(OSD_CLOCK_WIDTH, frame.width);
  const cameraWidth = evenSide(OSD_CAMERA_WIDTH, frame.width);
  return [
    [marginX, marginY, clockWidth, height],
    [frame.width - marginX - cameraWidth, frame.height - marginY - height, cameraWidth, height],
  ];
}

export function pointOnFrame(clientX: number, clientY: number, rect: SurfaceRect, frame: FrameSize): FramePoint {
  const x = rect.width > 0 ? ((clientX - rect.left) / rect.width) * frame.width : 0;
  const y = rect.height > 0 ? ((clientY - rect.top) / rect.height) * frame.height : 0;
  return { x: clamp(x, 0, frame.width), y: clamp(y, 0, frame.height) };
}

function evenSpan(from: number, to: number, total: number): readonly [number, number] {
  const start = Math.floor(Math.min(from, to));
  const end = Math.min(total, Math.ceil(Math.max(from, to)));
  return [start, evenFloor(end - start)];
}

// El backend exige ancho y alto pares y la caja entera dentro del cuadro guardado.
export function boxFromCorners(a: FramePoint, b: FramePoint, frame: FrameSize): CctvBox | null {
  const [x, width] = evenSpan(a.x, b.x, frame.width);
  const [y, height] = evenSpan(a.y, b.y, frame.height);
  return width >= MIN_SIDE && height >= MIN_SIDE ? [x, y, width, height] : null;
}

export function movedBox(box: CctvBox, dx: number, dy: number, frame: FrameSize): CctvBox {
  const [x, y, width, height] = box;
  return [clamp(x + dx, 0, frame.width - width), clamp(y + dy, 0, frame.height - height), width, height];
}

export function resizedBox(box: CctvBox, dw: number, dh: number, frame: FrameSize): CctvBox {
  const [x, y, width, height] = box;
  return [x, y, clamp(width + dw, MIN_SIDE, evenFloor(frame.width - x)), clamp(height + dh, MIN_SIDE, evenFloor(frame.height - y))];
}

const ARROW_DELTAS: Readonly<Record<string, readonly [number, number]>> = {
  ArrowLeft: [-STEP_PX, 0],
  ArrowRight: [STEP_PX, 0],
  ArrowUp: [0, -STEP_PX],
  ArrowDown: [0, STEP_PX],
};

export function boxAfterKey(box: CctvBox, key: string, resize: boolean, frame: FrameSize): CctvBox | null {
  const delta = ARROW_DELTAS[key];
  if (!delta) {
    return null;
  }
  return resize ? resizedBox(box, delta[0], delta[1], frame) : movedBox(box, delta[0], delta[1], frame);
}

export function withBoxAdded(boxes: readonly CctvBox[], box: CctvBox, maxBoxes: number): CctvBox[] {
  // Con una sola caja (ROI) dibujar otra la reemplaza; con varias, el tope deja las que hay.
  if (maxBoxes === 1) {
    return [box];
  }
  return boxes.length >= maxBoxes ? [...boxes] : [...boxes, box];
}

export function withBoxReplaced(boxes: readonly CctvBox[], index: number, box: CctvBox): CctvBox[] {
  return boxes.map((current, position) => (position === index ? box : current));
}

export function withBoxRemoved(boxes: readonly CctvBox[], index: number): CctvBox[] {
  return boxes.filter((_, position) => position !== index);
}

function sameBox(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

export function notTextIndices(boxes: readonly CctvBox[], checks: readonly OsdCheckResult[]): number[] {
  return boxes.flatMap((box, index) =>
    checks.some((check) => !check.looksLikeText && sameBox(check.box, box)) ? [index] : [],
  );
}

export function boxPercentStyle(box: CctvBox, frame: FrameSize): Record<string, string> {
  const [x, y, width, height] = box;
  return {
    left: `${(x / frame.width) * 100}%`,
    top: `${(y / frame.height) * 100}%`,
    width: `${(width / frame.width) * 100}%`,
    height: `${(height / frame.height) * 100}%`,
  };
}
