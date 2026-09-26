import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { Point } from "./geometryMath";
import { movedHandle, nudgedHandle, type Handles } from "./perspectiveMath";

interface PerspectiveOverlayProps {
  handles: Handles;
  onChange: (handles: Handles) => void;
}

const CORNER_LABEL_KEYS = [
  "restore.geometry.corner.topLeft",
  "restore.geometry.corner.topRight",
  "restore.geometry.corner.bottomRight",
  "restore.geometry.corner.bottomLeft",
] as const;
const SHADE = "rgb(0 0 0 / 0.45)";
const HANDLE =
  "absolute h-6 w-6 -translate-x-1/2 -translate-y-1/2 cursor-grab touch-none rounded-full border-2 border-accent bg-transparent shadow-[0_0_0_2px_rgb(0_0_0/0.55)] transition-transform duration-fast hover:scale-110 active:cursor-grabbing focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent after:absolute after:left-1/2 after:top-1/2 after:h-1 after:w-1 after:-translate-x-1/2 after:-translate-y-1/2 after:rounded-full after:bg-accent";

function percent(value: number): string {
  return `${value * 100}%`;
}

function polygonPath(handles: Handles): string {
  const [first, ...rest] = handles.map((handle) => `${handle.x * 100} ${handle.y * 100}`);
  return `M${first} ${rest.map((point) => `L${point}`).join(" ")} Z`;
}

function fractionIn(element: HTMLElement, event: PointerEvent<HTMLElement>): Point {
  const box = element.getBoundingClientRect();
  if (box.width <= 0 || box.height <= 0) {
    return { x: 0, y: 0 };
  }
  return { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height };
}

function QuadShade({ handles }: { handles: Handles }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 100 100" preserveAspectRatio="none" className="pointer-events-none absolute inset-0 h-full w-full">
      <path d={`M0 0 H100 V100 H0 Z ${polygonPath(handles)}`} fill={SHADE} fillRule="evenodd" />
      <path d={polygonPath(handles)} fill="none" stroke="var(--accent)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export function PerspectiveOverlay({ handles, onChange }: PerspectiveOverlayProps) {
  const { t } = useTranslation();
  const overlayRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<number | null>(null);

  function startDrag(event: PointerEvent<HTMLButtonElement>, index: number) {
    overlayRef.current?.setPointerCapture?.(event.pointerId);
    setDragging(index);
  }

  function drag(event: PointerEvent<HTMLDivElement>) {
    if (dragging === null) return;
    onChange(movedHandle(handles, dragging, fractionIn(event.currentTarget, event)));
  }

  function nudge(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const next = nudgedHandle(handles, index, event.key, event.shiftKey);
    if (next === null) return;
    event.preventDefault();
    onChange(next);
  }

  return (
    <div
      ref={overlayRef}
      data-testid="restore-perspective-overlay"
      onPointerMove={drag}
      onPointerUp={() => setDragging(null)}
      onPointerCancel={() => setDragging(null)}
      className="absolute inset-0 touch-none"
    >
      <QuadShade handles={handles} />
      {handles.map((handle, index) => (
        <button
          key={CORNER_LABEL_KEYS[index]}
          type="button"
          aria-label={t(CORNER_LABEL_KEYS[index])}
          onPointerDown={(event) => startDrag(event, index)}
          onKeyDown={(event) => nudge(event, index)}
          className={HANDLE}
          style={{ left: percent(handle.x), top: percent(handle.y) }}
        />
      ))}
    </div>
  );
}
