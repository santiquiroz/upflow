import { useTranslation } from "../../i18n/LocaleProvider";
// "RIFE" es el nombre del motor y no se traduce; el de GMFSS lleva una
// aclaracion que si. El mapa vive a nivel de modulo, asi que guarda claves.
const INTERP_ENGINE_LABEL_KEYS: Record<string, string> = {
  gmfss: "enhance.interp.gmfss",
};

export function interpEngineLabel(engine: string, t: (key: string) => string): string {
  if (engine === "rife") {
    return "RIFE";
  }
  const key = INTERP_ENGINE_LABEL_KEYS[engine];
  return key ? t(key) : engine;
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
  const { t } = useTranslation();
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
            {interpEngineLabel(engine, t)}
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
