import { Crop, RotateCw, Ruler, Undo2 } from "lucide-react";
import { useState, type PointerEvent } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RestoreGeometry } from "../../../lib/restoreApiTypes";
import {
  MAX_STRAIGHTEN_DEG,
  cropFromSelection,
  isNeutralGeometry,
  previewRotationDeg,
  rotateClockwise,
  selectionFromPoints,
  withAngle,
  withoutCrop,
  type FractionRect,
  type Point,
  type Size,
} from "./geometryMath";

type GeometryMode = "view" | "crop" | "straighten";

interface GeometryToolsProps {
  previewUrl: string;
  alt: string;
  geometry: RestoreGeometry;
  workingSize: Size;
  busy: boolean;
  onApply: (geometry: RestoreGeometry) => void;
}

const NEUTRAL_GEOMETRY: RestoreGeometry = { rotate90: 0, crop: null, angle: 0 };
const GRID_STYLE = {
  backgroundImage:
    "linear-gradient(to right, rgb(255 255 255 / 0.35) 1px, transparent 1px), linear-gradient(to bottom, rgb(255 255 255 / 0.35) 1px, transparent 1px)",
  backgroundSize: "12.5% 12.5%",
};
const TOOL_BUTTON =
  "inline-flex items-center gap-1.5 rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text transition-[border-color,opacity] duration-fast hover:border-accent disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent aria-pressed:border-accent";
const PRIMARY_BUTTON =
  "rounded-sm bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function percent(value: number): string {
  return `${value * 100}%`;
}

function pointOf(event: PointerEvent<HTMLElement>): Point {
  return { x: event.clientX, y: event.clientY };
}

function CropOverlay({
  selection,
  onSelect,
}: {
  selection: FractionRect | null;
  onSelect: (selection: FractionRect) => void;
}) {
  const [start, setStart] = useState<Point | null>(null);

  function select(event: PointerEvent<HTMLDivElement>, from: Point) {
    onSelect(selectionFromPoints(from, pointOf(event), event.currentTarget.getBoundingClientRect()));
  }

  function handleDown(event: PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setStart(pointOf(event));
    select(event, pointOf(event));
  }

  return (
    <div
      data-testid="restore-crop-overlay"
      onPointerDown={handleDown}
      onPointerMove={(event) => start && select(event, start)}
      onPointerUp={() => setStart(null)}
      className="absolute inset-0 cursor-crosshair touch-none"
    >
      {selection && (
        <div
          data-testid="restore-crop-selection"
          className="absolute border-2 border-accent shadow-[0_0_0_9999px_rgb(0_0_0/0.45)]"
          style={{
            left: percent(selection.x),
            top: percent(selection.y),
            width: percent(selection.width),
            height: percent(selection.height),
          }}
        />
      )}
    </div>
  );
}

function PreviewStage({
  previewUrl,
  alt,
  mode,
  rotationDeg,
  selection,
  onSelect,
}: {
  previewUrl: string;
  alt: string;
  mode: GeometryMode;
  rotationDeg: number;
  selection: FractionRect | null;
  onSelect: (selection: FractionRect) => void;
}) {
  return (
    <div className="relative mx-auto w-fit max-w-full overflow-hidden rounded border border-border bg-surface">
      <img
        src={previewUrl}
        alt={alt}
        className="block max-h-[480px] max-w-full select-none"
        draggable={false}
        style={rotationDeg === 0 ? undefined : { transform: `rotate(${rotationDeg}deg)` }}
      />
      {mode === "straighten" && (
        <div aria-hidden="true" data-testid="restore-straighten-grid" className="pointer-events-none absolute inset-0" style={GRID_STYLE} />
      )}
      {mode === "crop" && <CropOverlay selection={selection} onSelect={onSelect} />}
    </div>
  );
}

function StraightenControls({
  angle,
  busy,
  onChange,
  onApply,
  onCancel,
}: {
  angle: number;
  busy: boolean;
  onChange: (angle: number) => void;
  onApply: () => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-2 text-sm text-text">
        {t("restore.geometry.straighten")}
        <input
          type="range"
          min={-MAX_STRAIGHTEN_DEG}
          max={MAX_STRAIGHTEN_DEG}
          step={0.1}
          value={angle}
          onChange={(event) => onChange(Number(event.target.value))}
          className="w-48"
        />
      </label>
      <span className="font-mono-tabular text-sm text-text-dim">{t("restore.geometry.angle", { angle: angle.toFixed(1) })}</span>
      <button type="button" className={PRIMARY_BUTTON} disabled={busy} onClick={onApply}>
        {t("restore.geometry.applyStraighten")}
      </button>
      <button type="button" className={TOOL_BUTTON} onClick={onCancel}>
        {t("restore.geometry.cancel")}
      </button>
    </div>
  );
}

function CropControls({
  canApply,
  busy,
  onApply,
  onCancel,
}: {
  canApply: boolean;
  busy: boolean;
  onApply: () => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-3">
      <p className="text-sm text-text-dim">{t("restore.geometry.cropHint")}</p>
      <button type="button" className={PRIMARY_BUTTON} disabled={busy || !canApply} onClick={onApply}>
        {t("restore.geometry.applyCrop")}
      </button>
      <button type="button" className={TOOL_BUTTON} onClick={onCancel}>
        {t("restore.geometry.cancel")}
      </button>
    </div>
  );
}

export function GeometryTools({ previewUrl, alt, geometry, workingSize, busy, onApply }: GeometryToolsProps) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<GeometryMode>("view");
  const [draftAngle, setDraftAngle] = useState(geometry.angle);
  const [selection, setSelection] = useState<FractionRect | null>(null);
  const croppedGeometry = selection === null ? null : cropFromSelection(geometry, selection, workingSize);
  const rotationDeg = mode === "straighten" ? previewRotationDeg(geometry.angle, draftAngle) : 0;

  function enter(next: GeometryMode) {
    setMode(next);
    setSelection(null);
    setDraftAngle(geometry.angle);
  }

  function apply(next: RestoreGeometry | null) {
    if (next === null) return;
    enter("view");
    onApply(next);
  }

  return (
    <section aria-labelledby="restore-geometry-title" className="flex flex-col gap-3">
      <div>
        <h2 id="restore-geometry-title" className="text-sm font-medium text-text">
          {t("restore.geometry.title")}
        </h2>
        <p className="text-xs text-text-dim">{t("restore.geometry.hint")}</p>
      </div>
      <PreviewStage
        previewUrl={previewUrl}
        alt={alt}
        mode={mode}
        rotationDeg={rotationDeg}
        selection={selection}
        onSelect={setSelection}
      />
      <div className="flex flex-wrap gap-2">
        <button type="button" className={TOOL_BUTTON} disabled={busy} onClick={() => apply(rotateClockwise(geometry))}>
          <RotateCw aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("restore.geometry.rotate")}
        </button>
        <button type="button" className={TOOL_BUTTON} disabled={busy} aria-pressed={mode === "crop"} onClick={() => enter(mode === "crop" ? "view" : "crop")}>
          <Crop aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("restore.geometry.crop")}
        </button>
        <button type="button" className={TOOL_BUTTON} disabled={busy} aria-pressed={mode === "straighten"} onClick={() => enter(mode === "straighten" ? "view" : "straighten")}>
          <Ruler aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("restore.geometry.straighten")}
        </button>
        {geometry.crop !== null && (
          <button type="button" className={TOOL_BUTTON} disabled={busy} onClick={() => apply(withoutCrop(geometry))}>
            {t("restore.geometry.clearCrop")}
          </button>
        )}
        {!isNeutralGeometry(geometry) && (
          <button type="button" className={TOOL_BUTTON} disabled={busy} onClick={() => apply(NEUTRAL_GEOMETRY)}>
            <Undo2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            {t("restore.geometry.reset")}
          </button>
        )}
      </div>
      {mode === "straighten" && (
        <StraightenControls
          angle={draftAngle}
          busy={busy}
          onChange={(angle) => setDraftAngle(withAngle(geometry, angle).angle)}
          onApply={() => apply(withAngle(geometry, draftAngle))}
          onCancel={() => enter("view")}
        />
      )}
      {mode === "crop" && (
        <CropControls canApply={croppedGeometry !== null} busy={busy} onApply={() => apply(croppedGeometry)} onCancel={() => enter("view")} />
      )}
    </section>
  );
}
