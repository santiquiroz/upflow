import { useTranslation } from "../../i18n/LocaleProvider";
import type { SeparationModel } from "../../lib/apiTypes";

/** Cuántos entran en la combinación contando el principal. */
const MAX_MODELOS = 3;

/** Los que se pueden combinar con el elegido: mismas pistas, ya instalados.
 *
 * Mismas pistas no es un detalle: promediar un instrumental con un bajo no da
 * un instrumental mejor, da una suma que además nadie puede etiquetar.
 */
export function compatibleModels(
  models: SeparationModel[],
  selectedModel: string | null,
): SeparationModel[] {
  const elegido = models.find((model) => model.id === selectedModel);
  if (!elegido) {
    return [];
  }
  const pistas = elegido.stems.map((stem) => stem.id).join("|");
  return models.filter(
    (model) =>
      model.installed &&
      model.id !== elegido.id &&
      model.stems.map((stem) => stem.id).join("|") === pistas,
  );
}

/** Combinar varios separadores en uno ("máxima calidad").
 *
 * Cada arquitectura falla distinto: lo que comparten es la señal, que se suma
 * en fase, y sus artefactos no están correlacionados y se cancelan en parte.
 * Lo que NO hace es inventar separación que ninguno logró — por eso el texto
 * dice qué cuesta (una pasada más por modelo) en vez de prometer magia.
 */
export function EnsembleSection({
  models,
  selectedModel,
  chosen,
  onChange,
}: {
  models: SeparationModel[];
  selectedModel: string | null;
  chosen: string[];
  onChange: (models: string[]) => void;
}) {
  const { t } = useTranslation();
  const compatibles = compatibleModels(models, selectedModel);

  if (compatibles.length === 0) {
    return null;
  }

  function alternar(modelId: string): void {
    if (chosen.includes(modelId)) {
      onChange(chosen.filter((id) => id !== modelId));
      return;
    }
    // Lleno = se ignora, no se reemplaza en silencio: cambiar una elección que
    // el usuario no tocó es peor que no hacer nada visible. Mismo criterio que
    // el comparador de modelos.
    if (chosen.length + 1 >= MAX_MODELOS) {
      return;
    }
    onChange([...chosen, modelId]);
  }

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface p-3">
      <p className="text-sm text-text">{t("audio.ensemble.title")}</p>
      <p className="text-xs text-text-dim">{t("audio.ensemble.why")}</p>

      <div className="flex flex-wrap gap-2">
        {compatibles.map((modelo) => (
          <label
            key={modelo.id}
            className="flex items-center gap-1.5 rounded border border-border bg-surface-2 px-2 py-1 text-xs text-text"
          >
            <input
              type="checkbox"
              checked={chosen.includes(modelo.id)}
              onChange={() => alternar(modelo.id)}
              className="h-3.5 w-3.5 accent-accent"
            />
            {modelo.name}
          </label>
        ))}
      </div>

      {chosen.length > 0 && (
        <p role="status" className="text-xs text-warn">
          {/* El costo, dicho antes y no después: son N pasadas completas sobre
              el tema, no un ajuste que sale gratis. */}
          {t("audio.ensemble.cost", { count: chosen.length + 1 })}
        </p>
      )}
    </div>
  );
}
