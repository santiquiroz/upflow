import { useId } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";

// Realzar la banda de presencia mas de ~6 dB deja de sonar a voz enfocada y
// empieza a sonar a telefono. El limite es del oficio, no del filtro.
export const PRESENCE_MIN_DB = 0;
export const PRESENCE_MAX_DB = 6;
export const PRESENCE_DEFAULT_DB = 3;
const PRESENCE_STEP_DB = 0.5;

interface PresenceControlProps {
  value: number;
  onChange: (db: number) => void;
}

export function PresenceControl({ value, onChange }: PresenceControlProps) {
  const { t } = useTranslation();
  const sliderId = useId();

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={sliderId} className="text-xs font-medium text-text-dim">
          {t("voice.presence.label")}
        </label>
        <span className="font-mono-tabular text-[10px] text-text-faint">
          +{value.toFixed(1)} dB
        </span>
      </div>
      <input
        id={sliderId}
        type="range"
        min={PRESENCE_MIN_DB}
        max={PRESENCE_MAX_DB}
        step={PRESENCE_STEP_DB}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer accent-accent"
      />
      <p className="text-[11px] leading-relaxed text-text-faint">
        {t("voice.presence.hint")}
      </p>
    </div>
  );
}
