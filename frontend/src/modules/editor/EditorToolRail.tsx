import {
  BoxSelect,
  Brush,
  Eraser,
  Hand,
  Lasso,
  Maximize,
  MousePointerClick,
  RotateCcw,
  Shield,
  Trash2,
  Users,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { KeyboardEvent } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";

export type Tool =
  | "people"
  | "tap"
  | "protect"
  | "brush"
  | "eraser"
  | "lasso"
  | "rect"
  | "restore"
  | "clone"
  | "heal"
  | "pan";

interface EditorToolRailProps {
  tool: Tool;
  onToolChange: (tool: Tool) => void;
  disabledTools: Partial<Record<Tool, string>>;
  toolTitles?: Partial<Record<Tool, string>>;
  brushSize: number;
  onBrushSizeChange: (size: number) => void;
  zoomLabel: string;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
  canResetZoom: boolean;
  onUndo: () => void;
  canUndo: boolean;
  onClear: () => void;
  canClear: boolean;
  onClearProtection: () => void;
  canClearProtection: boolean;
  onNewImage: () => void;
  busy: boolean;
}

export const BRUSH_SIZE_MIN = 8;
export const BRUSH_SIZE_MAX = 128;

function moveRadioFocus(
  event: KeyboardEvent<HTMLDivElement>,
  items: readonly { value: Tool }[],
  disabledTools: Partial<Record<Tool, string>>,
  onToolChange: (tool: Tool) => void,
) {
  const direction = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
  const target = event.target;
  if (!direction || !(target instanceof HTMLButtonElement)) return;

  const enabledItems = items.filter((item) => !disabledTools[item.value]);
  const currentIndex = enabledItems.findIndex((item) => item.value === target.dataset.tool);
  if (currentIndex < 0 || enabledItems.length === 0) return;

  event.preventDefault();
  const nextIndex = (currentIndex + direction + enabledItems.length) % enabledItems.length;
  const nextItem = enabledItems[nextIndex];
  const nextButton = event.currentTarget.querySelector<HTMLButtonElement>(`[data-tool="${nextItem.value}"]`);
  nextButton?.focus();
  onToolChange(nextItem.value);
}

export function EditorToolRail({
  tool,
  onToolChange,
  disabledTools,
  toolTitles = {},
  brushSize,
  onBrushSizeChange,
  zoomLabel,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  canResetZoom,
  onUndo,
  canUndo,
  onClear,
  canClear,
  onClearProtection,
  canClearProtection,
  onNewImage,
  busy,
}: EditorToolRailProps) {
  const { t } = useTranslation();
  const tools = [
    { value: "people", icon: <Users aria-hidden="true" className="h-4 w-4 shrink-0" />, label: t("editor.tool.people") },
    { value: "tap", icon: <MousePointerClick aria-hidden="true" className="h-4 w-4 shrink-0" />, label: t("editor.tool.tap") },
    { value: "protect", icon: <Shield aria-hidden="true" className="h-4 w-4 shrink-0" />, label: t("editor.tool.protect") },
    { value: "brush", icon: <Brush aria-hidden="true" className="h-4 w-4 shrink-0" />, label: t("editor.tool.brush") },
    { value: "eraser", icon: <Eraser aria-hidden="true" className="h-4 w-4 shrink-0" />, label: t("editor.tool.eraser") },
    { value: "lasso", icon: <Lasso aria-hidden="true" className="h-4 w-4 shrink-0" />, label: t("editor.tool.lasso") },
    { value: "rect", icon: <BoxSelect aria-hidden="true" className="h-4 w-4 shrink-0" />, label: t("editor.tool.rect") },
  ] as const;
  const tabStopTool = tools.some((item) => item.value === tool && !disabledTools[item.value])
    ? tool
    : tools.find((item) => !disabledTools[item.value])?.value;
  const panTool = {
    value: "pan" as const,
    icon: <Hand aria-hidden="true" className="h-4 w-4 shrink-0" />,
    label: t("editor.tool.pan"),
  };
  const renderedTools = [...tools, panTool] as const;

  function renderToolButton(item: (typeof renderedTools)[number], radio = false) {
    const disabledReason = disabledTools[item.value];
    return (
      <button
        key={item.value}
        type="button"
        data-tool={item.value}
        role={radio ? "radio" : undefined}
        aria-checked={radio ? tool === item.value : undefined}
        aria-pressed={!radio ? tool === item.value : undefined}
        tabIndex={radio ? (tabStopTool === item.value ? 0 : -1) : undefined}
        aria-label={item.label}
        title={disabledReason ?? toolTitles[item.value] ?? item.label}
        disabled={Boolean(disabledReason)}
        onClick={() => onToolChange(item.value)}
        style={tool === item.value ? { boxShadow: "inset 3px 0 0 var(--accent)" } : undefined}
        className={`flex min-h-9 ${item.value === "pan" ? "w-auto" : "w-full"} items-center gap-2 rounded px-2 py-1.5 text-left text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
          tool === item.value
            ? "bg-surface-2 font-medium text-text"
            : "text-text-dim hover:bg-surface hover:text-text"
        }`}
      >
        {item.icon}
        <span>{item.label}</span>
      </button>
    );
  }

  return (
    <nav className="flex min-h-0 w-full flex-none flex-col gap-2 overflow-y-auto lg:w-[148px]">
      <div>
        <p className="mb-1 px-2 text-[11px] font-medium text-text-faint">{t("editor.tools.select")}</p>
        <div
          className="flex flex-col gap-0.5"
          role="radiogroup"
          aria-label={t("editor.tools.label")}
          onKeyDown={(event) => moveRadioFocus(event, tools, disabledTools, onToolChange)}
        >
          {tools.map((item) => renderToolButton(item, true))}
        </div>
      </div>

      <label className="flex flex-col gap-1 px-2 py-1 text-xs text-text-dim">
        {t("editor.brushSize")}
        <input
          type="range"
          min={BRUSH_SIZE_MIN}
          max={BRUSH_SIZE_MAX}
          value={brushSize}
          onChange={(event) => onBrushSizeChange(Number(event.target.value))}
        />
      </label>

      <div className="mt-1 flex flex-wrap gap-1 border-t border-border pt-2">
        {renderToolButton(panTool)}
        <button
          type="button"
          aria-label={t("editor.zoom.out")}
          title={t("editor.zoom.out")}
          onClick={onZoomOut}
          className="min-h-9 min-w-9 rounded border border-border p-2 text-text-dim hover:bg-surface-2 hover:text-text"
        >
          <ZoomOut aria-hidden="true" className="h-4 w-4" />
        </button>
        <span className="flex min-h-9 min-w-[3.5rem] items-center justify-center text-center font-mono-tabular text-xs text-text-dim">
          {zoomLabel}
        </span>
        <button
          type="button"
          aria-label={t("editor.zoom.in")}
          title={t("editor.zoom.in")}
          onClick={onZoomIn}
          className="min-h-9 min-w-9 rounded border border-border p-2 text-text-dim hover:bg-surface-2 hover:text-text"
        >
          <ZoomIn aria-hidden="true" className="h-4 w-4" />
        </button>
        <button
          type="button"
          aria-label={t("editor.zoom.reset")}
          title={t("editor.zoom.reset")}
          disabled={!canResetZoom}
          onClick={onZoomReset}
          className="min-h-9 rounded border border-border px-2 text-xs text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
        >
          <Maximize aria-hidden="true" className="mr-1 inline h-3.5 w-3.5" />
          {t("editor.zoom.reset")}
        </button>
        <button
          type="button"
          aria-label={t("editor.undo")}
          title={t("editor.undo")}
          disabled={!canUndo}
          onClick={onUndo}
          className="min-h-9 rounded border border-border px-2 text-xs text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
        >
          <RotateCcw aria-hidden="true" className="mr-1 inline h-3.5 w-3.5" />
          {t("editor.undo")}
        </button>
        <button
          type="button"
          aria-label={t("editor.clear")}
          title={t("editor.clear")}
          disabled={!canClear}
          onClick={onClear}
          className="min-h-9 rounded border border-border px-2 text-xs text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
        >
          <Trash2 aria-hidden="true" className="mr-1 inline h-3.5 w-3.5" />
          {t("editor.clear")}
        </button>
        <button
          type="button"
          aria-label={t("editor.protect.clear")}
          title={t("editor.protect.clear")}
          disabled={!canClearProtection}
          onClick={onClearProtection}
          className="min-h-9 rounded border border-border px-2 text-xs text-text-dim hover:bg-surface-2 disabled:opacity-40"
        >
          {t("editor.protect.clear")}
        </button>
        <button
          type="button"
          aria-label={t("editor.newImage")}
          title={t("editor.newImage")}
          disabled={busy}
          onClick={onNewImage}
          className="min-h-9 rounded border border-border px-2 text-xs text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
        >
          <X aria-hidden="true" className="mr-1 inline h-3.5 w-3.5" />
          {t("editor.newImage")}
        </button>
      </div>
    </nav>
  );
}
