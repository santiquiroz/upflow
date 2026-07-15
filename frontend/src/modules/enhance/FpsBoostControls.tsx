export interface FpsBoostValue {
  fpsMultiplier: number;
  targetFps: string | null;
}

interface FpsBoostControlsProps {
  value: FpsBoostValue;
  onChange: (value: FpsBoostValue) => void;
}

interface TargetFpsOption {
  value: string;
  label: string;
}

const MULTIPLIER_OPTIONS: readonly number[] = [2, 3, 4];

const TARGET_FPS_OPTIONS: readonly TargetFpsOption[] = [
  { value: "60000/1001", label: "59.94 fps" },
  { value: "60/1", label: "60 fps" },
];

function isMultiplierActive(value: FpsBoostValue): boolean {
  return value.fpsMultiplier > 1;
}

function isTargetActive(value: FpsBoostValue): boolean {
  return value.targetFps !== null;
}

function isOff(value: FpsBoostValue): boolean {
  return !isMultiplierActive(value) && !isTargetActive(value);
}

function segmentButtonClassName(isActive: boolean, isDisabled: boolean): string {
  const base =
    "font-mono-tabular rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isDisabled) {
    return `${base} cursor-not-allowed border-border bg-surface text-text-faint opacity-50`;
  }
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

export function FpsBoostControls({ value, onChange }: FpsBoostControlsProps) {
  const multiplierDisabled = isTargetActive(value);
  const targetDisabled = isMultiplierActive(value);

  function selectOff(): void {
    onChange({ fpsMultiplier: 1, targetFps: null });
  }

  function selectMultiplier(multiplier: number): void {
    onChange({ fpsMultiplier: multiplier, targetFps: null });
  }

  function selectTarget(targetFps: string): void {
    onChange({ fpsMultiplier: 1, targetFps });
  }

  return (
    <fieldset className="flex flex-col gap-3">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">FPS boost</legend>
      <div role="group" aria-label="FPS multiplier" className="flex flex-wrap gap-2">
        <button
          type="button"
          aria-pressed={isOff(value)}
          className={segmentButtonClassName(isOff(value), false)}
          onClick={selectOff}
        >
          Off
        </button>
        {MULTIPLIER_OPTIONS.map((multiplier) => (
          <button
            key={multiplier}
            type="button"
            aria-pressed={value.fpsMultiplier === multiplier}
            disabled={multiplierDisabled}
            className={segmentButtonClassName(value.fpsMultiplier === multiplier, multiplierDisabled)}
            onClick={() => selectMultiplier(multiplier)}
          >
            {multiplier}×
          </button>
        ))}
      </div>
      <div role="group" aria-label="Target FPS" className="flex flex-wrap gap-2">
        {TARGET_FPS_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={value.targetFps === option.value}
            disabled={targetDisabled}
            className={segmentButtonClassName(value.targetFps === option.value, targetDisabled)}
            onClick={() => selectTarget(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}
