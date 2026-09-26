import { clampPan, type ContainerSize, type Viewport } from "../modules/editor/viewport";

// Logica pura del comparador antes/despues: la cortina vive en porcentaje del
// ancho visible y el zoom reutiliza el modelo de viewport del Editor (zoom 1 =
// la foto ajustada al marco, pan en pixeles de pantalla).

export const CURTAIN_CENTER = 50;
export const CURTAIN_KEY_STEP = 2;
export const CURTAIN_PAGE_STEP = 10;
export const COMPARE_MIN_ZOOM = 1;
export const COMPARE_BASE_MAX_ZOOM = 8;
export const COMPARE_ZOOM_STEP = 1.5;

const CURTAIN_KEY_DELTAS: Record<string, number> = {
  ArrowLeft: -CURTAIN_KEY_STEP,
  ArrowDown: -CURTAIN_KEY_STEP,
  ArrowRight: CURTAIN_KEY_STEP,
  ArrowUp: CURTAIN_KEY_STEP,
  PageDown: -CURTAIN_PAGE_STEP,
  PageUp: CURTAIN_PAGE_STEP,
};

const CURTAIN_KEY_TARGETS: Record<string, number> = { Home: 0, End: 100 };

export function clampCurtain(percent: number): number {
  if (!Number.isFinite(percent)) {
    return CURTAIN_CENTER;
  }
  return Math.min(100, Math.max(0, percent));
}

export function curtainAtPointer(clientX: number, left: number, width: number, current = CURTAIN_CENTER): number {
  if (width <= 0) {
    return current;
  }
  return clampCurtain(((clientX - left) / width) * 100);
}

export function curtainAfterKey(current: number, key: string): number | null {
  if (key in CURTAIN_KEY_TARGETS) {
    return CURTAIN_KEY_TARGETS[key];
  }
  const delta = CURTAIN_KEY_DELTAS[key];
  return delta === undefined ? null : clampCurtain(current + delta);
}

export function actualSizeZoom(naturalWidth: number, fittedWidth: number): number | null {
  if (naturalWidth <= 0 || fittedWidth <= 0) {
    return null;
  }
  return Math.max(COMPARE_MIN_ZOOM, naturalWidth / fittedWidth);
}

export function maxZoomFor(actualSize: number | null): number {
  return Math.max(COMPARE_BASE_MAX_ZOOM, actualSize ?? 0);
}

export function pixelPercent(zoom: number, actualSize: number): number {
  return Math.round((zoom / actualSize) * 100);
}

export function zoomAroundCenter(
  viewport: Viewport,
  nextZoom: number,
  container: ContainerSize,
  maxZoom: number,
): Viewport {
  const zoom = Math.min(maxZoom, Math.max(COMPARE_MIN_ZOOM, nextZoom));
  const ratio = zoom / viewport.zoom;
  const centerX = container.width / 2;
  const centerY = container.height / 2;
  return clampPan(
    {
      zoom,
      panX: centerX - (centerX - viewport.panX) * ratio,
      panY: centerY - (centerY - viewport.panY) * ratio,
    },
    container,
  );
}
