import { Cpu, Zap, type LucideIcon } from "lucide-react";
import type { VideoEncoder } from "../lib/apiTypes";

interface EncoderOption {
  value: VideoEncoder;
  label: string;
  subtitle: string;
  Icon: LucideIcon;
}

// Mirrors the backend video_encoder contract (app/services/video_encoders.py):
// the create route accepts exactly "software" | "auto" and rejects anything else
// with a 400, so this list is the single source of truth for selectable values.
export const ENCODER_OPTIONS: readonly EncoderOption[] = [
  {
    value: "software",
    label: "Software (x264/x265)",
    subtitle: "Always compatible — best quality per bit",
    Icon: Cpu,
  },
  {
    value: "auto",
    label: "Auto (GPU)",
    subtitle: "Uses the GPU encoder — far faster in 4K (NVENC/AMF/QSV)",
    Icon: Zap,
  },
];

export function formatEncoderSummary(encoder: VideoEncoder): string {
  return ENCODER_OPTIONS.find((option) => option.value === encoder)?.label ?? encoder;
}

function encoderOptionClassName(isSelected: boolean): string {
  const base =
    "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast focus-within:outline focus-within:outline-2 focus-within:outline-accent";
  if (isSelected) {
    return `${base} border-accent bg-surface-2`;
  }
  return `${base} border-border bg-surface hover:border-text-faint`;
}

function EncoderOptionRow({
  option,
  isSelected,
  onChange,
}: {
  option: EncoderOption;
  isSelected: boolean;
  onChange: (value: VideoEncoder) => void;
}) {
  const { Icon } = option;
  return (
    <label className={encoderOptionClassName(isSelected)}>
      <span className="flex items-center gap-2">
        <input
          type="radio"
          name="encoder"
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

interface EncoderPickerProps {
  value: VideoEncoder;
  onChange: (value: VideoEncoder) => void;
}

export function EncoderPicker({ value, onChange }: EncoderPickerProps) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Encoder</legend>
      {ENCODER_OPTIONS.map((option) => (
        <EncoderOptionRow
          key={option.value}
          option={option}
          isSelected={option.value === value}
          onChange={onChange}
        />
      ))}
    </fieldset>
  );
}
