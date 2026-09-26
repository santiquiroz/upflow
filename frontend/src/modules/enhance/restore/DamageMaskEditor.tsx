import { Hand, Maximize, Paintbrush, Eraser, RotateCcw, Trash2, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import {
  appendPoint,
  drawStrokes,
  isCommittableStroke,
  startStroke,
  toImagePoint,
  type BinaryMask,
  type BrushStroke,
} from "../../editor/maskCanvas";
import { isZoomed, transformStyle, zoomPercent } from "../../editor/viewport";
import { MASK_CSS_COLOR, overlayPixels, type MaskSettings, type MaskSize, type ProbabilityMap } from "./damageMask";
import { SliderField } from "./RestoreStepControls";
import { stepUi, type SliderControl } from "./restoreSteps";
import type { DamageMaskState } from "./useDamageMask";
import { useMaskViewport, type MaskViewport } from "./useMaskViewport";

export type MaskTool = "add" | "erase" | "move";

const TOOLS: readonly { id: MaskTool; labelKey: string; icon: typeof Hand }[] = [
  { id: "add", labelKey: "restore.mask.tool.add", icon: Paintbrush },
  { id: "erase", labelKey: "restore.mask.tool.erase", icon: Eraser },
  { id: "move", labelKey: "editor.tool.pan", icon: Hand },
];
const MIN_BRUSH_PX = 4;
const MAX_BRUSH_PX = 96;
const DEFAULT_BRUSH_PX = 20;
const MIDDLE_BUTTON = 1;
const PERCENT = 100;

export interface DamageMaskEditorProps {
  previewUrl: string;
  alt: string;
  size: MaskSize;
  damage: DamageMaskState;
  settings: MaskSettings;
  leaveLargeHoles: boolean;
  leaveFaces: boolean;
  damageOverFaces: boolean;
  onOptionChange: (option: string, value: unknown) => void;
  onConfirm: () => void;
}

function repairSliders(): { sensitivity: SliderControl; grow: SliderControl } {
  const ui = stepUi("repair");
  const grow = ui?.advanced.find((control) => control.option === "grow_px");
  if (!ui || grow?.kind !== "slider") {
    throw new Error("The repair step has no sensitivity or grow controls");
  }
  return { sensitivity: ui.intensity, grow };
}

function paintOverlay(canvas: HTMLCanvasElement, map: ProbabilityMap | null, mask: BinaryMask | null): void {
  const context = canvas.getContext("2d");
  if (context === null) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (mask === null) return;
  const image = context.createImageData(mask.width, mask.height);
  image.data.set(overlayPixels(map, mask));
  context.putImageData(image, 0, 0);
}

// Solo el ultimo tramo: repintar el trazo entero acumularia la transparencia.
function drawLiveSegment(canvas: HTMLCanvasElement, stroke: BrushStroke): void {
  const context = canvas.getContext("2d");
  if (context === null) return;
  drawStrokes(context, [{ ...stroke, points: stroke.points.slice(-2) }], MASK_CSS_COLOR);
}

interface IconButtonProps {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: ReactNode;
}

function IconButton({ label, disabled, onClick, children }: IconButtonProps) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="rounded p-2 text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      {children}
    </button>
  );
}

function ToolPicker({ tool, onChange }: { tool: MaskTool; onChange: (tool: MaskTool) => void }) {
  const { t } = useTranslation();
  return (
    <div
      role="radiogroup"
      aria-label={t("restore.mask.tools")}
      className="flex gap-1 rounded border border-border bg-surface-2 p-1"
    >
      {TOOLS.map(({ id, labelKey, icon: Icon }) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={tool === id}
          onClick={() => onChange(id)}
          className={`inline-flex items-center gap-1.5 rounded px-3 py-1 text-xs transition-colors ${
            tool === id ? "bg-accent font-medium text-surface" : "text-text-dim hover:text-text"
          }`}
        >
          <Icon aria-hidden="true" className="h-3.5 w-3.5" />
          {t(labelKey)}
        </button>
      ))}
    </div>
  );
}

interface ToolbarProps {
  tool: MaskTool;
  brushPx: number;
  view: MaskViewport;
  damage: DamageMaskState;
  onToolChange: (tool: MaskTool) => void;
  onBrushChange: (px: number) => void;
}

function MaskToolbar({ tool, brushPx, view, damage, onToolChange, onBrushChange }: ToolbarProps) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-3">
      <ToolPicker tool={tool} onChange={onToolChange} />
      <label className="flex items-center gap-2 text-xs text-text-dim">
        {t("restore.mask.brushSize")}
        <input
          type="range"
          min={MIN_BRUSH_PX}
          max={MAX_BRUSH_PX}
          value={brushPx}
          onChange={(event) => onBrushChange(Number(event.target.value))}
          className="h-1.5 w-28 cursor-pointer accent-accent"
        />
      </label>
      <div className="ml-auto flex items-center gap-1">
        <IconButton label={t("editor.zoom.out")} onClick={() => view.zoomStep(-1)}>
          <ZoomOut aria-hidden="true" className="h-4 w-4" />
        </IconButton>
        <span className="min-w-[3.5rem] text-center font-mono-tabular text-xs text-text-dim">
          {zoomPercent(view.viewport)}%
        </span>
        <IconButton label={t("editor.zoom.in")} onClick={() => view.zoomStep(1)}>
          <ZoomIn aria-hidden="true" className="h-4 w-4" />
        </IconButton>
        <IconButton label={t("editor.zoom.reset")} disabled={!isZoomed(view.viewport)} onClick={view.reset}>
          <Maximize aria-hidden="true" className="h-4 w-4" />
        </IconButton>
        <span aria-hidden="true" className="mx-1 h-5 w-px bg-border" />
        <IconButton label={t("editor.undo")} disabled={!damage.canUndo} onClick={damage.undo}>
          <RotateCcw aria-hidden="true" className="h-4 w-4" />
        </IconButton>
        <IconButton label={t("restore.mask.clear")} disabled={!damage.edited} onClick={damage.clear}>
          <Trash2 aria-hidden="true" className="h-4 w-4" />
        </IconButton>
      </div>
    </div>
  );
}

interface StageProps {
  previewUrl: string;
  alt: string;
  size: MaskSize;
  tool: MaskTool;
  brushPx: number;
  showMask: boolean;
  damage: DamageMaskState;
  view: MaskViewport;
  stageRef: React.RefObject<HTMLDivElement>;
}

function cursorClass(tool: MaskTool): string {
  return tool === "move" ? "cursor-grab active:cursor-grabbing" : "cursor-crosshair";
}

function MaskStage({ previewUrl, alt, size, tool, brushPx, showMask, damage, view, stageRef }: StageProps) {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const liveRef = useRef<BrushStroke | null>(null);
  const canPaint = damage.mapStatus !== "loading";

  useEffect(() => {
    if (canvasRef.current) paintOverlay(canvasRef.current, damage.map, damage.mask);
  }, [damage.map, damage.mask]);

  function imagePoint(event: ReactPointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return toImagePoint(event.clientX, event.clientY, rect, size.width, size.height);
  }

  // El pincel se elige en px de pantalla; el trazo se guarda en px de la foto.
  function brushRadius(canvas: HTMLCanvasElement): number {
    const shown = canvas.getBoundingClientRect().width;
    return (brushPx / 2) * (shown > 0 ? size.width / shown : 1);
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    event.currentTarget.setPointerCapture?.(event.pointerId);
    if (tool === "move" || event.button === MIDDLE_BUTTON) {
      event.preventDefault();
      view.startPan(event.clientX, event.clientY);
      return;
    }
    if (!canPaint) return;
    const mode = tool === "add" ? "paint" : "erase";
    liveRef.current = startStroke(mode, brushRadius(event.currentTarget), imagePoint(event));
    drawLiveSegment(event.currentTarget, liveRef.current);
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (view.movePan(event.clientX, event.clientY) || liveRef.current === null) return;
    const next = appendPoint(liveRef.current, imagePoint(event));
    if (next === liveRef.current) return;
    liveRef.current = next;
    drawLiveSegment(event.currentTarget, next);
  }

  function handlePointerUp() {
    view.endPan();
    const stroke = liveRef.current;
    liveRef.current = null;
    if (stroke !== null && isCommittableStroke(stroke)) damage.addStroke(stroke);
  }

  return (
    <div
      ref={stageRef}
      data-testid="damage-mask-stage"
      className="relative w-fit max-w-full touch-none self-center overflow-hidden rounded border border-border"
    >
      <div style={{ transform: transformStyle(view.viewport), transformOrigin: "0 0" }}>
        <img src={previewUrl} alt={alt} draggable={false} className="block max-h-[70vh] max-w-full select-none" />
        <canvas
          ref={canvasRef}
          width={size.width}
          height={size.height}
          aria-label={t("restore.mask.canvas")}
          className={`absolute inset-0 h-full w-full ${cursorClass(tool)} ${showMask ? "" : "opacity-0"}`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />
      </div>
    </div>
  );
}

function MapStatusLine({ status }: { status: DamageMaskState["mapStatus"] }) {
  const { t } = useTranslation();
  if (status === "loading") {
    return <p role="status" className="text-xs text-text-dim">{t("restore.mask.loading")}</p>;
  }
  if (status === "failed") {
    return <p role="alert" className="text-xs text-warn">{t("restore.mask.loadFailed")}</p>;
  }
  return null;
}

interface OptionCheckboxProps {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}

function OptionCheckbox({ label, checked, onChange }: OptionCheckboxProps) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-text-dim">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-3.5 w-3.5 shrink-0 cursor-pointer accent-accent"
      />
      {label}
    </label>
  );
}

type WarningsProps = Pick<
  DamageMaskEditorProps,
  "damage" | "leaveLargeHoles" | "leaveFaces" | "damageOverFaces" | "onOptionChange"
>;

function MaskWarnings({ damage, leaveLargeHoles, leaveFaces, damageOverFaces, onOptionChange }: WarningsProps) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-2">
      <p className="rounded border border-warn bg-surface-2 px-3 py-2 text-xs text-text">
        {t("restore.mask.checkWriting")}
      </p>
      {damage.largeHoles > 0 && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs text-warn">{t("restore.warning.largeHoles")}</p>
          <OptionCheckbox
            label={t("restore.action.leaveLargeHoles")}
            checked={leaveLargeHoles}
            onChange={(value) => onOptionChange("leave_large_holes", value)}
          />
        </div>
      )}
      {damageOverFaces && (
        <div className="flex flex-col gap-1.5">
          <p className="text-xs text-warn">{t("restore.warning.damageOverFaces")}</p>
          <OptionCheckbox
            label={t("restore.action.leaveFaces")}
            checked={leaveFaces}
            onChange={(value) => onOptionChange("leave_faces", value)}
          />
        </div>
      )}
    </div>
  );
}

export function coverageText(coverage: number): number {
  return Math.round(coverage * PERCENT * 10) / 10;
}

export function DamageMaskEditor(props: DamageMaskEditorProps) {
  const { previewUrl, alt, size, damage, settings, onOptionChange, onConfirm } = props;
  const { t } = useTranslation();
  const stageRef = useRef<HTMLDivElement>(null);
  const view = useMaskViewport(stageRef);
  const [tool, setTool] = useState<MaskTool>("add");
  const [brushPx, setBrushPx] = useState(DEFAULT_BRUSH_PX);
  const [showMask, setShowMask] = useState(true);
  const sliders = repairSliders();
  const hasMap = damage.map !== null;

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-3">
      <MaskToolbar
        tool={tool}
        brushPx={brushPx}
        view={view}
        damage={damage}
        onToolChange={setTool}
        onBrushChange={setBrushPx}
      />
      <MaskStage
        previewUrl={previewUrl}
        alt={alt}
        size={size}
        tool={tool}
        brushPx={brushPx}
        showMask={showMask}
        damage={damage}
        view={view}
        stageRef={stageRef}
      />
      <p className="text-center text-xs text-text-faint">{t("editor.zoomHint")}</p>
      <MapStatusLine status={damage.mapStatus} />
      <OptionCheckbox label={t("restore.mask.show")} checked={showMask} onChange={setShowMask} />
      <div className="grid gap-3 sm:grid-cols-2">
        <SliderField
          control={sliders.sensitivity}
          value={settings.sensitivity}
          disabled={!hasMap}
          onChange={(value) => onOptionChange("sensitivity", value)}
        />
        <SliderField
          control={sliders.grow}
          value={settings.growPx}
          disabled={!hasMap}
          onChange={(value) => onOptionChange("grow_px", value)}
        />
      </div>
      <p className="font-mono-tabular text-xs text-text">
        {t("restore.mask.coverage", { pct: coverageText(damage.coverage) })}
      </p>
      <MaskWarnings {...props} />
      <button
        type="button"
        onClick={onConfirm}
        className="w-fit rounded bg-accent px-4 py-2 text-sm font-medium text-bg hover:bg-accent-hover active:bg-accent-press focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {t("restore.mask.confirm")}
      </button>
    </div>
  );
}
