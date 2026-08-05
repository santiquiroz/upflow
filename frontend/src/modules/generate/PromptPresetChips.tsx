import { useQuery } from "@tanstack/react-query";
import { apiGet } from "../../lib/api";
import { useTranslation } from "../../i18n/LocaleProvider";

export interface PromptPreset {
  id: string;
  mode: string;
  labelKey: string;
  prompt: string;
  negativePrompt: string;
}

// El NOMBRE del preset viene como clave y se traduce; el PROMPT viene literal y
// se usa tal cual. Traducir el prompt cambiaria lo que el modelo genera.
export function PromptPresetChips({
  mode,
  onApply,
}: {
  mode: string;
  onApply: (preset: PromptPreset) => void;
}) {
  const { t } = useTranslation();
  const presetsQuery = useQuery({
    queryKey: ["promptPresets"],
    queryFn: () => apiGet<{ presets: PromptPreset[] }>("/generation/prompt-presets"),
  });

  const presets = (presetsQuery.data?.presets ?? []).filter((preset) => preset.mode === mode);
  if (presets.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-text-dim">{t("generate.preset.title")}</span>
      <div className="flex flex-wrap gap-2">
        {presets.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => onApply(preset)}
            className="rounded-sm border border-border bg-surface px-3 py-1.5 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-accent hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            {t(preset.labelKey)}
          </button>
        ))}
      </div>
      <span className="text-xs text-text-faint">{t("generate.preset.hint")}</span>
    </div>
  );
}
