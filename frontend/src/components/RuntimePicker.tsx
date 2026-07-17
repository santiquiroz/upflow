import { Cpu, Sparkles, Zap, type LucideIcon } from "lucide-react";
import type { UpscaleBackend } from "../lib/apiTypes";

interface RuntimeOption {
  value: UpscaleBackend;
  label: string;
  subtitle: string;
  Icon: LucideIcon;
}

// Mirrors the backend UPSCALE_BACKEND contract (app/api/routes.py): the create
// route accepts exactly "auto" | "ncnn" | "onnx" and rejects anything else with
// a 400, so this list is the single source of truth for the selectable values.
export const RUNTIME_OPTIONS: readonly RuntimeOption[] = [
  {
    value: "auto",
    label: "Auto",
    subtitle: "Best backend for your device",
    Icon: Sparkles,
  },
  {
    value: "ncnn",
    label: "NCNN Vulkan",
    subtitle: "Portable fallback — runs on any Vulkan GPU",
    Icon: Cpu,
  },
  {
    value: "onnx",
    label: "ONNX DirectML",
    subtitle: "~2× faster on modern GPUs for video",
    Icon: Zap,
  },
];

export function formatRuntimeSummary(backend: UpscaleBackend): string {
  return RUNTIME_OPTIONS.find((option) => option.value === backend)?.label ?? backend;
}

function runtimeOptionClassName(isSelected: boolean): string {
  const base =
    "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast focus-within:outline focus-within:outline-2 focus-within:outline-accent";
  if (isSelected) {
    return `${base} border-accent bg-surface-2`;
  }
  return `${base} border-border bg-surface hover:border-text-faint`;
}

function RuntimeOptionRow({
  option,
  isSelected,
  onChange,
}: {
  option: RuntimeOption;
  isSelected: boolean;
  onChange: (value: UpscaleBackend) => void;
}) {
  const { Icon } = option;
  return (
    <label className={runtimeOptionClassName(isSelected)}>
      <span className="flex items-center gap-2">
        <input
          type="radio"
          name="runtime"
          value={option.value}
          checked={isSelected}
          onChange={() => onChange(option.value)}
          className="h-3.5 w-3.5 accent-accent"
        />
        <Icon aria-hidden="true" className="h-4 w-4 text-text-faint" strokeWidth={1.75} />
        <span className="text-sm text-text">{option.label}</span>
      </span>
      <span className="pl-[26px] text-xs text-text-faint">{option.subtitle}</span>
    </label>
  );
}

interface RuntimePickerProps {
  value: UpscaleBackend;
  onChange: (value: UpscaleBackend) => void;
}

export function RuntimePicker({ value, onChange }: RuntimePickerProps) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Runtime</legend>
      {RUNTIME_OPTIONS.map((option) => (
        <RuntimeOptionRow
          key={option.value}
          option={option}
          isSelected={option.value === value}
          onChange={onChange}
        />
      ))}
    </fieldset>
  );
}
