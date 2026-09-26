// Lógica pura del canvas de máscara del Editor. El contrato del backend
// (generation_onnx._load_mask_image) es: PNG en escala de grises,
// BLANCO = editar, NEGRO = conservar. Todo lo que toca DOM/canvas real vive
// en EditorPanel; esto se testea sin jsdom-canvas.

export interface BrushPoint {
  x: number;
  y: number;
}

export interface BrushStroke {
  // "lasso": polígono a mano alzada que se cierra y rellena al soltar — pinta
  // área editable igual que "paint", pero por contorno en vez de por trazo.
  mode: "paint" | "erase" | "lasso";
  radius: number;
  points: BrushPoint[];
}

export const MIN_POINT_DISTANCE_PX = 2;
// Con menos de 3 puntos no hay polígono: el lazo se descarta al soltar.
export const MIN_LASSO_POINTS = 3;

export function isCommittableStroke(stroke: BrushStroke): boolean {
  if (stroke.mode === "lasso") {
    return stroke.points.length >= MIN_LASSO_POINTS;
  }
  return stroke.points.length > 0;
}

export function strokeAddsEditableArea(stroke: BrushStroke | undefined): boolean {
  if (!stroke) {
    return false;
  }
  if (stroke.mode === "lasso") {
    return stroke.points.length >= MIN_LASSO_POINTS;
  }
  return stroke.mode === "paint" && stroke.points.length > 0;
}

export function startStroke(mode: BrushStroke["mode"], radius: number, point: BrushPoint): BrushStroke {
  return { mode, radius, points: [point] };
}

export function appendPoint(stroke: BrushStroke, point: BrushPoint): BrushStroke {
  const last = stroke.points[stroke.points.length - 1];
  const distance = Math.hypot(point.x - last.x, point.y - last.y);
  if (distance < MIN_POINT_DISTANCE_PX) {
    return stroke;
  }
  return { ...stroke, points: [...stroke.points, point] };
}

export function undoLastStroke(strokes: BrushStroke[]): BrushStroke[] {
  return strokes.slice(0, -1);
}

export function hasEditableArea(strokes: BrushStroke[]): boolean {
  // Un trazo de goma no agrega área editable; con solo gomas la máscara queda negra.
  return strokes.some((stroke) => strokeAddsEditableArea(stroke));
}

export function toImagePoint(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  imageWidth: number,
  imageHeight: number,
): BrushPoint {
  if (rect.width <= 0 || rect.height <= 0) {
    return { x: 0, y: 0 };
  }
  const x = ((clientX - rect.left) / rect.width) * imageWidth;
  const y = ((clientY - rect.top) / rect.height) * imageHeight;
  return {
    x: Math.min(Math.max(x, 0), imageWidth),
    y: Math.min(Math.max(y, 0), imageHeight),
  };
}

export function rectanglePoints(origin: BrushPoint, corner: BrushPoint): BrushPoint[] {
  return [
    { x: origin.x, y: origin.y },
    { x: corner.x, y: origin.y },
    { x: corner.x, y: corner.y },
    { x: origin.x, y: corner.y },
  ];
}


// El job de generación exige dimensiones múltiplo de 64 entre 64 y 1024; el
// engine redimensiona base y máscara a ese tamaño y el resultado sale así.
export function fitGenerationSize(
  imageWidth: number,
  imageHeight: number,
  maxSide = 1024,
): { width: number; height: number } {
  const scale = Math.min(1, maxSide / Math.max(imageWidth, imageHeight, 1));
  const snap = (value: number) =>
    Math.min(maxSide, Math.max(64, Math.round((value * scale) / 64) * 64));
  return { width: snap(imageWidth), height: snap(imageHeight) };
}

interface StrokeContext {
  lineWidth: number;
  lineCap: string;
  lineJoin: string;
  strokeStyle: unknown;
  fillStyle: unknown;
  globalCompositeOperation: string;
  beginPath(): void;
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  closePath(): void;
  stroke(): void;
  arc(x: number, y: number, radius: number, start: number, end: number): void;
  fill(): void;
}

function drawLasso(ctx: StrokeContext, stroke: BrushStroke, paintColor: string): void {
  // Con 2 puntos todavía no hay polígono, pero el preview en vivo del arrastre
  // ya se dibuja relleno: muestra exactamente lo que va a quedar al soltar.
  if (stroke.points.length < 2) {
    return;
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = paintColor;
  const [first, ...rest] = stroke.points;
  ctx.beginPath();
  ctx.moveTo(first.x, first.y);
  for (const point of rest) {
    ctx.lineTo(point.x, point.y);
  }
  ctx.closePath();
  ctx.fill();
}

export function drawStrokes(ctx: StrokeContext, strokes: BrushStroke[], paintColor: string): void {
  for (const stroke of strokes) {
    if (stroke.points.length === 0) {
      continue;
    }
    if (stroke.mode === "lasso") {
      drawLasso(ctx, stroke, paintColor);
      continue;
    }
    // La goma borra píxeles del overlay (destination-out) en vez de pintar
    // negro encima: así el overlay de la UI queda transparente donde no se edita.
    ctx.globalCompositeOperation = stroke.mode === "erase" ? "destination-out" : "source-over";
    ctx.strokeStyle = paintColor;
    ctx.fillStyle = paintColor;
    const [first, ...rest] = stroke.points;
    if (rest.length === 0) {
      ctx.beginPath();
      ctx.arc(first.x, first.y, stroke.radius, 0, Math.PI * 2);
      ctx.fill();
      continue;
    }
    ctx.lineWidth = stroke.radius * 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(first.x, first.y);
    for (const point of rest) {
      ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
  }
  ctx.globalCompositeOperation = "source-over";
}

// Máscara binaria por píxel (1 = marcado), fila por fila. Es la versión en
// memoria de lo que se sube: sin canvas, así se compone y se mide en los tests.
export interface BinaryMask {
  width: number;
  height: number;
  data: Uint8Array;
}

interface PixelBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function strokeValue(stroke: BrushStroke): number {
  return stroke.mode === "erase" ? 0 : 1;
}

function clippedBounds(points: BrushPoint[], margin: number, width: number, height: number): PixelBounds {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: Math.max(0, Math.floor(Math.min(...xs) - margin)),
    maxX: Math.min(width - 1, Math.ceil(Math.max(...xs) + margin)),
    minY: Math.max(0, Math.floor(Math.min(...ys) - margin)),
    maxY: Math.min(height - 1, Math.ceil(Math.max(...ys) + margin)),
  };
}

function distanceToSegment(px: number, py: number, a: BrushPoint, b: BrushPoint): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSquared = dx * dx + dy * dy;
  const along = lengthSquared === 0 ? 0 : ((px - a.x) * dx + (py - a.y) * dy) / lengthSquared;
  const t = Math.min(1, Math.max(0, along));
  return Math.hypot(px - (a.x + t * dx), py - (a.y + t * dy));
}

function fillPixels(
  mask: BinaryMask,
  bounds: PixelBounds,
  value: number,
  covers: (x: number, y: number) => boolean,
): void {
  for (let y = bounds.minY; y <= bounds.maxY; y += 1) {
    for (let x = bounds.minX; x <= bounds.maxX; x += 1) {
      // Centro del píxel, igual que el muestreo del canvas.
      if (covers(x + 0.5, y + 0.5)) {
        mask.data[y * mask.width + x] = value;
      }
    }
  }
}

function stampSegment(mask: BinaryMask, a: BrushPoint, b: BrushPoint, radius: number, value: number): void {
  const bounds = clippedBounds([a, b], radius, mask.width, mask.height);
  fillPixels(mask, bounds, value, (x, y) => distanceToSegment(x, y, a, b) <= radius);
}

function isInsidePolygon(x: number, y: number, polygon: BrushPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    const crosses = a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x;
    inside = crosses ? !inside : inside;
  }
  return inside;
}

function stampLasso(mask: BinaryMask, stroke: BrushStroke): void {
  if (stroke.points.length < MIN_LASSO_POINTS) {
    return;
  }
  const bounds = clippedBounds(stroke.points, 0, mask.width, mask.height);
  fillPixels(mask, bounds, 1, (x, y) => isInsidePolygon(x, y, stroke.points));
}

function stampBrush(mask: BinaryMask, stroke: BrushStroke): void {
  const value = strokeValue(stroke);
  const [first, ...rest] = stroke.points;
  let previous = first;
  for (const point of rest.length === 0 ? [first] : rest) {
    stampSegment(mask, previous, point, stroke.radius, value);
    previous = point;
  }
}

export function rasterizeStrokes(base: BinaryMask, strokes: readonly BrushStroke[]): BinaryMask {
  const mask = { ...base, data: base.data.slice() };
  for (const stroke of strokes) {
    if (stroke.points.length === 0) {
      continue;
    }
    if (stroke.mode === "lasso") {
      stampLasso(mask, stroke);
    } else {
      stampBrush(mask, stroke);
    }
  }
  return mask;
}
