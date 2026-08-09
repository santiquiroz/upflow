import { PackDownload } from "../../components/PackDownload";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { SeparationModel } from "../../lib/apiTypes";

function modelButtonClassName(isActive: boolean): string {
  const base =
    "inline-flex items-center gap-2 rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

/**
 * Modo karaoke: separar la mezcla en instrumental + voz.
 *
 * El picker sale del catálogo del backend (instalados y descargables): la idea
 * no es casarse con un modelo, cada tarjeta faltante trae su botón de descarga
 * por variante del pack "karaoke".
 */
export function KaraokeSection({
  enabled,
  onToggle,
  models,
  selectedModel,
  onSelectModel,
}: {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  models: SeparationModel[];
  selectedModel: string | null;
  onSelectModel: (modelId: string) => void;
}) {
  const { t } = useTranslation();
  const installed = models.filter((model) => model.installed);
  const missing = models.filter((model) => !model.installed);

  return (
    <div className="flex flex-col gap-3">
      <label className="flex items-center gap-2 text-sm text-text">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(event) => onToggle(event.target.checked)}
          className="h-4 w-4 accent-accent"
        />
        {t("audio.karaoke.toggle")}
      </label>

      {enabled && installed.length === 0 && (
        <p role="status" className="text-xs text-warn">
          {t("audio.karaoke.noModels")}
        </p>
      )}

      {installed.length > 0 && (
        <fieldset className="flex flex-col gap-2">
          <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
            {t("audio.karaoke.model")}
          </legend>
          <div className="flex flex-wrap gap-2">
            {installed.map((model) => (
              <button
                key={model.id}
                type="button"
                aria-pressed={selectedModel === model.id}
                className={modelButtonClassName(selectedModel === model.id)}
                onClick={() => onSelectModel(model.id)}
              >
                {model.name}
              </button>
            ))}
          </div>
        </fieldset>
      )}

      {missing.map((model) => (
        <PackDownload
          key={model.id}
          pack="karaoke"
          variant={model.id}
          reason={t("audio.karaoke.modelMissing", { name: model.name })}
        />
      ))}

      <p className="text-xs text-text-faint">{t("audio.karaoke.credit")}</p>
    </div>
  );
}
