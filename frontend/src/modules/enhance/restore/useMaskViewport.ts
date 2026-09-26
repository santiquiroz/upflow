import { useEffect, useRef, useState, type RefObject } from "react";
import {
  clampPan,
  IDENTITY_VIEWPORT,
  panBy,
  zoomAtPoint,
  zoomByStep,
  type ContainerSize,
  type Viewport,
} from "../../editor/viewport";

const WHEEL_ZOOM = 1.15;

export interface MaskViewport {
  viewport: Viewport;
  zoomStep: (direction: number) => void;
  reset: () => void;
  startPan: (clientX: number, clientY: number) => void;
  movePan: (clientX: number, clientY: number) => boolean;
  endPan: () => void;
}

// clientWidth/Height y no el rect: el rect incluye el borde y afloja los limites.
function innerSize(stage: HTMLElement | null): ContainerSize {
  return { width: stage?.clientWidth ?? 0, height: stage?.clientHeight ?? 0 };
}

function wheelZoom(stage: HTMLElement, event: WheelEvent, current: Viewport): Viewport {
  const rect = stage.getBoundingClientRect();
  const focus = { x: event.clientX - rect.left - stage.clientLeft, y: event.clientY - rect.top - stage.clientTop };
  const factor = event.deltaY < 0 ? WHEEL_ZOOM : 1 / WHEEL_ZOOM;
  return zoomAtPoint(current, current.zoom * factor, focus, innerSize(stage));
}

// Mismo modelo que el Editor (viewport.ts): transform CSS sobre foto y lienzo.
export function useMaskViewport(stageRef: RefObject<HTMLElement>): MaskViewport {
  const [viewport, setViewport] = useState<Viewport>(IDENTITY_VIEWPORT);
  const panOrigin = useRef<{ x: number; y: number } | null>(null);

  // React registra onWheel como pasivo: sin esto la pagina se desplaza al hacer zoom.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    function handleWheel(event: WheelEvent) {
      event.preventDefault();
      setViewport((current) => wheelZoom(stage as HTMLElement, event, current));
    }
    stage.addEventListener("wheel", handleWheel, { passive: false });
    return () => stage.removeEventListener("wheel", handleWheel);
  }, [stageRef]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => setViewport((current) => clampPan(current, innerSize(stage))));
    observer.observe(stage);
    return () => observer.disconnect();
  }, [stageRef]);

  return {
    viewport,
    zoomStep: (direction) => setViewport((current) => zoomByStep(current, direction, innerSize(stageRef.current))),
    reset: () => setViewport(IDENTITY_VIEWPORT),
    startPan: (clientX, clientY) => {
      panOrigin.current = { x: clientX, y: clientY };
    },
    movePan: (clientX, clientY) => {
      const origin = panOrigin.current;
      if (origin === null) return false;
      panOrigin.current = { x: clientX, y: clientY };
      const container = innerSize(stageRef.current);
      setViewport((current) => panBy(current, clientX - origin.x, clientY - origin.y, container));
      return true;
    },
    endPan: () => {
      panOrigin.current = null;
    },
  };
}
