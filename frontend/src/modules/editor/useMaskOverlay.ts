import { useRef, type RefObject } from "react";
import type { CommittedMasks, EdgeAdjust } from "./editorMaskCanvas";
import type { BrushStroke } from "./maskCanvas";
import { displaySize, overlayLayers, paintOverlay, type OverlayLayers } from "./maskOverlay";

interface MaskOverlayOptions {
  masks: CommittedMasks | null;
  photoWidth: number;
  photoHeight: number;
  edge: EdgeAdjust | null;
  redRef: RefObject<HTMLCanvasElement | null>;
  greenRef: RefObject<HTMLCanvasElement | null>;
  liveRef: RefObject<BrushStroke | null>;
}

interface CachedLayers {
  masks: CommittedMasks;
  layers: OverlayLayers;
}

function layersFor(cached: CachedLayers | null, masks: CommittedMasks, width: number, height: number): OverlayLayers {
  const reusable = cached?.masks === masks && cached.layers.width === width && cached.layers.height === height;
  return reusable ? cached.layers : overlayLayers(masks, width, height);
}

// Devuelve el redibujo del overlay: solo pinta, nunca reconstruye el historial.
export function useMaskOverlay({ masks, photoWidth, photoHeight, edge, redRef, greenRef, liveRef }: MaskOverlayOptions) {
  const cacheRef = useRef<CachedLayers | null>(null);
  return function redraw(): void {
    const red = redRef.current;
    const green = greenRef.current;
    if (!red || !green || !masks) return;
    const size = displaySize(red, photoWidth, photoHeight);
    const layers = layersFor(cacheRef.current, masks, size.width, size.height);
    cacheRef.current = { masks, layers };
    paintOverlay(red, green, layers, liveRef.current, photoWidth, edge);
  };
}
