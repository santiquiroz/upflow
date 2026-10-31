import { useRef, useState } from "react";
import { appendPoint, startStroke, type BrushPoint, type BrushStroke } from "./maskCanvas";
import type { RetouchKind, RetouchOffset } from "./retouchApply";

export type RetouchCommit = (kind: RetouchKind, stroke: BrushStroke, offset: RetouchOffset) => Promise<void>;

interface ActiveStroke {
  kind: RetouchKind;
  stroke: BrushStroke;
  offset: RetouchOffset;
}

const NO_OFFSET: RetouchOffset = { dx: 0, dy: 0 };

export function isRetouchTool(tool: string): tool is RetouchKind {
  return tool === "restore" || tool === "clone";
}

function sampleAt(point: BrushPoint, offset: RetouchOffset): BrushPoint {
  return { x: point.x + offset.dx, y: point.y + offset.dy };
}

// Un trazo se aplica al soltar: mientras una aplicación está en vuelo, los trazos nuevos se ignoran.
export function useRetouchTool(onCommit: RetouchCommit) {
  const [source, setSource] = useState<BrushPoint | null>(null);
  const [samplePoint, setSamplePoint] = useState<BrushPoint | null>(null);
  const [painting, setPainting] = useState(false);
  const [applying, setApplying] = useState(false);
  const activeRef = useRef<ActiveStroke | null>(null);
  const applyingRef = useRef(false);

  function start(kind: RetouchKind, point: BrushPoint, altKey: boolean, radius: number) {
    if (applyingRef.current) return;
    if (kind === "clone" && altKey) {
      setSource(point);
      return;
    }
    if (kind === "clone" && !source) return;
    const offset = kind === "clone" && source
      ? { dx: source.x - point.x, dy: source.y - point.y }
      : NO_OFFSET;
    activeRef.current = { kind, stroke: startStroke("paint", radius, point), offset };
    setSamplePoint(sampleAt(point, offset));
    setPainting(true);
  }

  function move(point: BrushPoint) {
    const active = activeRef.current;
    if (!active) return;
    active.stroke = appendPoint(active.stroke, point);
    setSamplePoint(sampleAt(point, active.offset));
  }

  async function end() {
    const active = activeRef.current;
    activeRef.current = null;
    setPainting(false);
    setSamplePoint(null);
    if (!active) return;
    applyingRef.current = true;
    setApplying(true);
    try {
      await onCommit(active.kind, active.stroke, active.offset);
    } finally {
      applyingRef.current = false;
      setApplying(false);
    }
  }

  function reset() {
    setSource(null);
  }

  return { source, samplePoint, painting, applying, start, move, end, reset };
}
