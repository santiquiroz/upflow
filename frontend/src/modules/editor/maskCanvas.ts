// Lógica pura del canvas de máscara del Editor. El contrato del backend
// (generation_onnx._load_mask_image) es: PNG en escala de grises,
// BLANCO = editar, NEGRO = conservar. Todo lo que toca DOM/canvas real vive
// en EditorPanel; esto se testea sin jsdom-canvas.

export interface BrushPoint {
  x: number;
  y: number;
}

export interface BrushStroke {
  mode: "paint" | "erase";
  radius: number;
  points: BrushPoint[];
}

export const MIN_POINT_DISTANCE_PX = 2;

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
  return strokes.some((stroke) => stroke.mode === "paint" && stroke.points.length > 0);
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
  stroke(): void;
  arc(x: number, y: number, radius: number, start: number, end: number): void;
  fill(): void;
}

export function drawStrokes(ctx: StrokeContext, strokes: BrushStroke[], paintColor: string): void {
  for (const stroke of strokes) {
    if (stroke.points.length === 0) {
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
