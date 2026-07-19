const INTERP_ENGINE_LABELS: Record<string, string> = {
  rife: "RIFE",
  gmfss: "GMFSS (max quality, very slow)",
};

export function interpEngineLabel(engine: string): string {
  return INTERP_ENGINE_LABELS[engine] ?? engine;
}

interface InterpEngineControlsProps {
  engines: string[];
  value: string;
  onChange: (value: string) => void;
}

function segmentButtonClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

export function InterpEngineControls({ engines, value, onChange }: InterpEngineControlsProps) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        Interpolation engine
      </legend>
      <div className="flex flex-wrap gap-2">
        {engines.map((engine) => (
          <button
            key={engine}
            type="button"
            aria-pressed={value === engine}
            className={segmentButtonClassName(value === engine)}
            onClick={() => onChange(engine)}
          >
            {interpEngineLabel(engine)}
          </button>
        ))}
      </div>
      {value === "gmfss" && (
        <p role="status" className="text-xs text-warn">
          GMFSS trades speed for quality: expect roughly 10x or more processing time versus RIFE. Best
          suited to short clips where maximum quality matters more than turnaround.
        </p>
      )}
    </fieldset>
  );
}
