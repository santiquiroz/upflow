import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvPreset } from "../../../services/cctv";
import { translateOr } from "./cctvText";

function presetButtonClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  return isActive ? `${base} border-accent bg-accent text-bg` : `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

export function CctvPresetPicker({
  presets,
  value,
  suggested,
  onChange,
}: {
  presets: CctvPreset[];
  value: string | null;
  suggested: string | null;
  onChange: (presetId: string) => void;
}) {
  const { t } = useTranslation();
  const selected = presets.find((preset) => preset.id === value) ?? null;
  return (
    <div className="flex flex-col gap-2">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.preset.legend")}</span>
      <div role="group" aria-label={t("cctv.preset.legend")} className="flex flex-wrap gap-2">
        {presets.map((preset) => (
          <button
            key={preset.id}
            type="button"
            aria-pressed={preset.id === value}
            onClick={() => onChange(preset.id)}
            className={presetButtonClassName(preset.id === value)}
          >
            {translateOr(t, preset.labelKey, preset.label)}
            {preset.id === suggested && <span className="ml-1.5 text-[11px] opacity-80">· {t("cctv.preset.suggested")}</span>}
          </button>
        ))}
      </div>
      {selected && <p className="text-xs text-text-dim">{translateOr(t, selected.descriptionKey, selected.description)}</p>}
    </div>
  );
}
