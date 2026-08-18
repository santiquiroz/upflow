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

function orderedCategories(models: SeparationModel[]): string[] {
  // El orden lo fija el catálogo del backend; acá solo se de-duplica.
  const seen: string[] = [];
  for (const model of models) {
    if (!seen.includes(model.category)) {
      seen.push(model.category);
    }
  }
  return seen;
}

/**
 * La advertencia de un modelo, visible SIEMPRE — esté instalado o no, esté
 * elegido o no. Es el punto: enterarse de que un modelo cuesta ~9x cuando el
 * trabajo ya arrancó no es haber elegido.
 */
function ModelWarning({ model }: { model: SeparationModel }) {
  const { t } = useTranslation();
  if (!model.warningKey) {
    return null;
  }
  return (
    <p role="note" className="text-xs text-warn">
      <span className="font-semibold">{model.name}:</span> {t(model.warningKey)}
    </p>
  );
}

/**
 * Un grupo del picker (categoría "karaoke" o "cleanup"): botones para los
 * modelos instalados, la descripción del modelo seleccionado, y una tarjeta
 * de descarga por cada modelo que falta.
 */
function ModelCategoryGroup({
  category,
  models,
  selectedModel,
  onSelectModel,
}: {
  category: string;
  models: SeparationModel[];
  selectedModel: string | null;
  onSelectModel: (modelId: string) => void;
}) {
  const { t } = useTranslation();
  const installed = models.filter((model) => model.installed);
  const missing = models.filter((model) => !model.installed);
  const selected = installed.find((model) => model.id === selectedModel);

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t(`audio.karaoke.category.${category}`)}
      </legend>
      {/* Los mismos modelos viven en la sección Limpieza, encadenables. Acá
          corren de a UNO y devuelven los dos stems; sin decirlo, las dos
          secciones se leen como la misma cosa duplicada. */}
      {category === "cleanup" && (
        <p className="text-xs text-text-faint">{t("audio.karaoke.cleanupGroupHint")}</p>
      )}
      {installed.length > 0 && (
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
              {/* border y color heredados (border-current): el chip tiene que
                  leerse tanto sobre el botón activo (fondo acento) como sobre
                  el inactivo. */}
              {model.badgeKey && (
                <span className="rounded-sm border border-current px-1 text-[0.65rem] uppercase tracking-wide">
                  {t(model.badgeKey)}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
      {selected && (
        <p role="status" className="text-xs text-text-dim">
          {t(selected.descriptionKey)}
        </p>
      )}
      {models.map((model) => (
        <ModelWarning key={`warning-${model.id}`} model={model} />
      ))}
      {missing.map((model) => (
        <PackDownload
          key={model.id}
          pack="karaoke"
          variant={model.id}
          reason={`${t("audio.karaoke.modelMissing", { name: model.name })} ${t(model.descriptionKey)}`}
        />
      ))}
    </fieldset>
  );
}

/**
 * Modo separación: karaoke (voz/instrumental) y limpieza (quitar reverb).
 *
 * El picker sale del catálogo del backend (instalados y descargables),
 * agrupado por categoría: la idea no es casarse con un modelo, cada tarjeta
 * faltante trae su botón de descarga por variante del pack "karaoke".
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

      {orderedCategories(models).map((category) => (
        <ModelCategoryGroup
          key={category}
          category={category}
          models={models.filter((model) => model.category === category)}
          selectedModel={selectedModel}
          onSelectModel={onSelectModel}
        />
      ))}

      <p className="text-xs text-text-faint">{t("audio.karaoke.credit")}</p>
    </div>
  );
}
