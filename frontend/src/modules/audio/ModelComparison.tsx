import { useMutation } from "@tanstack/react-query";
import { Scale } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { SeparationModel } from "../../lib/apiTypes";
import { jobQueueStore } from "../../lib/jobQueueStore";
import { compareSeparationModels } from "../../services/audio";

/** Cuántos se pueden comparar de una. Espejo de MAX_MODELS_PER_COMPARISON. */
const MAX_MODELOS = 3;

function formatearMinutos(segundos: number): string {
  const minutos = Math.floor(segundos / 60);
  const resto = Math.floor(segundos % 60);
  return `${minutos}:${String(resto).padStart(2, "0")}`;
}

function alternar(elegidos: string[], modelId: string): string[] {
  if (elegidos.includes(modelId)) {
    return elegidos.filter((id) => id !== modelId);
  }
  if (elegidos.length >= MAX_MODELOS) {
    return elegidos;
  }
  return [...elegidos, modelId];
}

/** Probar 2-3 separadores en un fragmento antes de pagar el tema entero.
 *
 * Qué separador anda mejor depende del material: los rankings publicados
 * promedian sobre un dataset que no es el del usuario. Correrlos sobre SU
 * archivo es la única respuesta que sirve, y sobre 30 s cuesta ~8 veces menos
 * que sobre una canción.
 *
 * No se muestra con menos de dos modelos instalados: comparar contra nada.
 */
export function ModelComparison({
  file,
  models,
}: {
  file: File | null;
  models: SeparationModel[];
}) {
  const { t } = useTranslation();
  const instalados = models.filter((model) => model.installed);
  const [elegidos, setElegidos] = useState<string[]>([]);

  const comparar = useMutation({
    mutationFn: () =>
      compareSeparationModels({ file: file as File, models: elegidos }),
    onSuccess: (comparacion) => {
      comparacion.entries.forEach((entrada) => {
        const modelo = models.find((model) => model.id === entrada.modelId);
        jobQueueStore.addTrackedJob({
          id: entrada.jobId,
          kind: "audio",
          // El nombre lleva el modelo: dos entradas iguales en la cola no se
          // pueden comparar sin abrir el detalle de cada una.
          fileName: `${file?.name ?? ""} [${modelo?.name ?? entrada.modelId}]`,
          createdAt: Date.now(),
        });
      });
    },
  });

  if (instalados.length < 2) {
    return null;
  }

  const puedeComparar = file !== null && elegidos.length >= 2;

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface p-3">
      <p className="text-sm text-text">{t("audio.compare.title")}</p>
      <p className="text-xs text-text-dim">{t("audio.compare.why")}</p>

      <div className="flex flex-wrap gap-2">
        {instalados.map((modelo) => (
          <label
            key={modelo.id}
            className="flex items-center gap-1.5 rounded border border-border bg-surface-2 px-2 py-1 text-xs text-text"
          >
            <input
              type="checkbox"
              checked={elegidos.includes(modelo.id)}
              onChange={() => setElegidos(alternar(elegidos, modelo.id))}
              className="h-3.5 w-3.5 accent-accent"
            />
            {modelo.name}
          </label>
        ))}
      </div>

      <button
        type="button"
        onClick={() => comparar.mutate()}
        disabled={!puedeComparar || comparar.isPending}
        className="self-start flex items-center gap-1.5 rounded border border-accent bg-surface-2 px-3 py-1.5 text-sm text-text disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Scale className="h-4 w-4" /> {t("audio.compare.submit")}
      </button>

      {comparar.data && (
        <p role="status" className="text-xs text-text-dim">
          {/* Qué parte del tema se está oyendo: el arranque lo elige el
              servidor (el medio), y sin decirlo el resultado es un misterio. */}
          {t("audio.compare.started", {
            count: comparar.data.entries.length,
            at: formatearMinutos(comparar.data.offsetSeconds),
            seconds: comparar.data.excerptSeconds,
          })}
        </p>
      )}
      {comparar.isError && (
        <p role="alert" className="text-xs text-danger">
          {(comparar.error as Error).message}
        </p>
      )}
    </div>
  );
}
