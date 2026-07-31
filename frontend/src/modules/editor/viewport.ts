// Zoom y desplazamiento del lienzo del Editor. Lógica pura: el componente solo
// aplica el transform resultante.
//
// Modelo: a zoom 1 el contenido ocupa EXACTAMENTE el contenedor, y el transform
// es `translate(pan) scale(zoom)` con transform-origin 0 0. Por eso el pan legal
// vive en [contenedor*(1-zoom), 0] y a zoom 1 solo cabe 0 — el contenido nunca
// puede despegarse de los bordes y dejar huecos.
//
// Importante para el resto del panel: como el zoom se aplica por CSS transform,
// getBoundingClientRect() del canvas ya devuelve el rectángulo transformado, así
// que el mapeo click->píxel de imagen (toImagePoint) y el radio del pincel
// siguen siendo correctos sin tocarlos.

export interface Viewport {
  zoom: number;
  panX: number;
  panY: number;
}

export interface ContainerSize {
  width: number;
  height: number;
}

export interface FocusPoint {
  x: number;
  y: number;
}

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;
export const ZOOM_STEP = 1.5;

export const IDENTITY_VIEWPORT: Viewport = { zoom: MIN_ZOOM, panX: 0, panY: 0 };

export function clampZoom(zoom: number): number {
  if (!Number.isFinite(zoom)) {
    return MIN_ZOOM;
  }
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

export function clampPan(viewport: Viewport, container: ContainerSize): Viewport {
  if (container.width <= 0 || container.height <= 0) {
    // Contenedor sin medir (pestaña oculta, jsdom): no inventar límites.
    return viewport;
  }
  const minPanX = container.width * (1 - viewport.zoom);
  const minPanY = container.height * (1 - viewport.zoom);
  return {
    zoom: viewport.zoom,
    panX: Math.min(0, Math.max(minPanX, viewport.panX)),
    panY: Math.min(0, Math.max(minPanY, viewport.panY)),
  };
}

export function zoomAtPoint(
  viewport: Viewport,
  nextZoom: number,
  focus: FocusPoint,
  container: ContainerSize,
): Viewport {
  const zoom = clampZoom(nextZoom);
  const ratio = zoom / viewport.zoom;
  // El punto de contenido bajo el cursor tiene que quedarse bajo el cursor:
  // pan' = focus - (focus - pan) * zoom'/zoom.
  return clampPan(
    {
      zoom,
      panX: focus.x - (focus.x - viewport.panX) * ratio,
      panY: focus.y - (focus.y - viewport.panY) * ratio,
    },
    container,
  );
}

export function zoomByStep(viewport: Viewport, direction: number, container: ContainerSize): Viewport {
  const factor = direction > 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
  const center = { x: container.width / 2, y: container.height / 2 };
  return zoomAtPoint(viewport, viewport.zoom * factor, center, container);
}

export function panBy(viewport: Viewport, dx: number, dy: number, container: ContainerSize): Viewport {
  return clampPan({ zoom: viewport.zoom, panX: viewport.panX + dx, panY: viewport.panY + dy }, container);
}

export function transformStyle(viewport: Viewport): string {
  return `translate(${viewport.panX}px, ${viewport.panY}px) scale(${viewport.zoom})`;
}

export function isZoomed(viewport: Viewport): boolean {
  return viewport.zoom !== MIN_ZOOM || viewport.panX !== 0 || viewport.panY !== 0;
}

export function zoomPercent(viewport: Viewport): number {
  return Math.round(viewport.zoom * 100);
}
