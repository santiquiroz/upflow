interface AudioEnhanceControlsProps {
  value: string | null;
  onChange: (mode: string | null) => void;
  keepAudio: boolean;
}

interface AudioEnhanceOption {
  value: string | null;
  label: string;
}

const AUDIO_ENHANCE_OPTIONS: readonly AudioEnhanceOption[] = [
  { value: null, label: "Off" },
  { value: "rnnoise", label: "RNNoise" },
  { value: "deepfilter", label: "DeepFilterNet" },
];

function segmentButtonClassName(isActive: boolean, isDisabled: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isDisabled) {
    return `${base} cursor-not-allowed border-border bg-surface text-text-faint opacity-50`;
  }
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

export function AudioEnhanceControls({ value, onChange, keepAudio }: AudioEnhanceControlsProps) {
  const disabled = !keepAudio;

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        Audio enhance
      </legend>
      <div role="group" aria-label="Audio enhance" className="flex flex-wrap gap-2">
        {AUDIO_ENHANCE_OPTIONS.map((option) => (
          <button
            key={option.label}
            type="button"
            aria-pressed={value === option.value}
            disabled={disabled}
            className={segmentButtonClassName(value === option.value, disabled)}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
      {disabled && <p className="text-xs text-text-faint">Requires "Keep original audio" to be enabled.</p>}
    </fieldset>
  );
}
