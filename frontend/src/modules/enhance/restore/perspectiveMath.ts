import type { Corner, Quad, RestoreCapture, RestoreGeometry } from "../../../lib/restoreApiTypes";
import type { Point, Size } from "./geometryMath";

// Margen de las esquinas iniciales cuando no hay una sugerencia del analisis.
export const DEFAULT_HANDLE_INSET = 0.08;
export const NUDGE_STEP = 0.005;
export const NUDGE_STEP_LARGE = 0.02;
const CORNER_DECIMALS = 10;

export type Handles = [Point, Point, Point, Point];

const DEFAULT_HANDLES: Handles = [
  { x: DEFAULT_HANDLE_INSET, y: DEFAULT_HANDLE_INSET },
  { x: 1 - DEFAULT_HANDLE_INSET, y: DEFAULT_HANDLE_INSET },
  { x: 1 - DEFAULT_HANDLE_INSET, y: 1 - DEFAULT_HANDLE_INSET },
  { x: DEFAULT_HANDLE_INSET, y: 1 - DEFAULT_HANDLE_INSET },
];

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

// cv2.getRotationMatrix2D gira el contenido en sentido antihorario sobre el centro
// de la foto girada; en coordenadas de borde ese centro es (ancho/2, alto/2).
function rotateAboutCenter(point: Point, angleDeg: number, frame: Size, inverse: boolean): Point {
  const radians = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = inverse ? -Math.sin(radians) : Math.sin(radians);
  const dx = point.x - frame.width / 2;
  const dy = point.y - frame.height / 2;
  return { x: cos * dx + sin * dy + frame.width / 2, y: -sin * dx + cos * dy + frame.height / 2 };
}

export function workingToFrame(point: Point, geometry: RestoreGeometry, frame: Size): Point {
  const [offsetX, offsetY] = geometry.crop ?? [0, 0];
  return rotateAboutCenter({ x: point.x + offsetX, y: point.y + offsetY }, geometry.angle, frame, true);
}

export function frameToWorking(point: Point, geometry: RestoreGeometry, frame: Size): Point {
  const [offsetX, offsetY] = geometry.crop ?? [0, 0];
  const straightened = rotateAboutCenter(point, geometry.angle, frame, false);
  return { x: straightened.x - offsetX, y: straightened.y - offsetY };
}

// Sin tamano del marco (API vieja) solo se puede mapear si no hay recorte: enderezar no cambia el tamano.
export function perspectiveFrame(
  capture: RestoreCapture | undefined,
  geometry: RestoreGeometry,
  working: Size,
): Size | null {
  if (capture && capture.frameWidth > 0 && capture.frameHeight > 0) {
    return { width: capture.frameWidth, height: capture.frameHeight };
  }
  return geometry.crop === null ? working : null;
}

function roundedCorner(point: Point, frame: Size): Corner {
  const x = clamp(point.x, 0, frame.width);
  const y = clamp(point.y, 0, frame.height);
  return [Math.round(x * CORNER_DECIMALS) / CORNER_DECIMALS, Math.round(y * CORNER_DECIMALS) / CORNER_DECIMALS];
}

export function perspectiveFromHandles(
  geometry: RestoreGeometry,
  handles: Handles,
  working: Size,
  frame: Size,
): RestoreGeometry {
  const corners = handles.map((handle) =>
    roundedCorner(workingToFrame({ x: handle.x * working.width, y: handle.y * working.height }, geometry, frame), frame),
  ) as Quad;
  return { rotate90: geometry.rotate90, crop: null, angle: 0, corners };
}

function handleFromFrame(corner: Corner, geometry: RestoreGeometry, working: Size, frame: Size): Point {
  const point = frameToWorking({ x: corner[0], y: corner[1] }, geometry, frame);
  return { x: clamp(point.x / working.width, 0, 1), y: clamp(point.y / working.height, 0, 1) };
}

export function initialHandles(
  geometry: RestoreGeometry,
  suggestion: RestoreGeometry | null | undefined,
  working: Size,
  frame: Size,
): Handles {
  const corners = suggestion?.corners;
  if (!corners || suggestion.rotate90 !== geometry.rotate90) {
    return DEFAULT_HANDLES;
  }
  return corners.map((corner) => handleFromFrame(corner, geometry, working, frame)) as Handles;
}

export function isConvexQuad(points: readonly Point[]): boolean {
  const turns = points.map((point, index) => {
    const next = points[(index + 1) % points.length];
    const after = points[(index + 2) % points.length];
    return (next.x - point.x) * (after.y - next.y) - (next.y - point.y) * (after.x - next.x);
  });
  return turns.every((turn) => turn > 0) || turns.every((turn) => turn < 0);
}

export function movedHandle(handles: Handles, index: number, to: Point): Handles {
  const clamped = { x: clamp(to.x, 0, 1), y: clamp(to.y, 0, 1) };
  return handles.map((handle, position) => (position === index ? clamped : handle)) as Handles;
}

const NUDGES: Record<string, Point> = {
  ArrowLeft: { x: -1, y: 0 },
  ArrowRight: { x: 1, y: 0 },
  ArrowUp: { x: 0, y: -1 },
  ArrowDown: { x: 0, y: 1 },
};

export function nudgedHandle(handles: Handles, index: number, key: string, large: boolean): Handles | null {
  const direction = NUDGES[key];
  if (!direction) {
    return null;
  }
  const step = large ? NUDGE_STEP_LARGE : NUDGE_STEP;
  const current = handles[index];
  return movedHandle(handles, index, { x: current.x + direction.x * step, y: current.y + direction.y * step });
}

function cropCorners(crop: NonNullable<RestoreGeometry["crop"]>): Point[] {
  const [x, y, width, height] = crop;
  return [
    { x, y },
    { x: x + width, y },
    { x: x + width, y: y + height },
    { x, y: y + height },
  ];
}

// Contorno de una foto sugerida (recorte nivelado) dibujado sobre la copia de trabajo actual.
export function suggestionOutline(
  suggestion: RestoreGeometry,
  current: RestoreGeometry,
  working: Size,
  frame: Size,
): Point[] | null {
  if (!suggestion.crop || suggestion.rotate90 !== current.rotate90 || current.corners) {
    return null;
  }
  const leveled = { rotate90: suggestion.rotate90, crop: null, angle: suggestion.angle };
  return cropCorners(suggestion.crop).map((corner) => {
    const inFrame = workingToFrame(corner, leveled, frame);
    const point = frameToWorking(inFrame, current, frame);
    return { x: point.x / working.width, y: point.y / working.height };
  });
}
