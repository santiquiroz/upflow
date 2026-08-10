import { useId } from "react";
import { PackDownload } from "../../components/PackDownload";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { CleanupStep } from "../../lib/apiTypes";

interface CleanupStepCardProps {
  step: CleanupStep;
  // Posición en la cadena, 1-indexada. Se muestra porque el orden NO es
  // arbitrario: quitar ruido primero, eco después, reverb al final (ver
  // cleanup_chain.py). Una lista plana invitaría a creer que da lo mismo.
  position: number;
  isLast: boolean;
  enabled: boolean;
  /** La cadena entera está apagada: el paso se ve pero no se elige. */
  locked: boolean;
  /** Ids que este paso apagaría al encenderse, ya traducidos a nombres. */
  conflictNames: string[];
  onToggle: (enabled: boolean) => void;
}

/**
 * Qué hace ESTE paso, no a qué familia pertenece: `deecho_dereverb` es de la
 * familia de-echo pero resuelve las dos, y titularlo "quitar eco" lo dejaría
 * indistinguible del de-echo simple justo arriba. La clave sale de `covers`,
 * que es el dato que ya describe el alcance real de la pasada.
 */
function taskKeyFor(step: CleanupStep): string {
  return `audio.cleanup.task.${step.covers.join("_")}`;
}

export function CleanupStepCard({
  step,
  position,
  isLast,
  enabled,
  locked,
  conflictNames,
  onToggle,
}: CleanupStepCardProps) {
  const { t } = useTranslation();
  const checkboxId = useId();
  const descriptionId = useId();

  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center gap-1 pt-2.5">
        <span
          aria-hidden="true"
          className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border font-mono-tabular text-[10px] ${
            enabled ? "border-accent bg-accent text-bg" : "border-border text-text-faint"
          }`}
        >
          {position}
        </span>
        {!isLast && <span aria-hidden="true" className="w-px flex-1 bg-border" />}
      </div>

      <div
        className={`flex flex-1 flex-col gap-2 rounded border px-3 py-2.5 transition-[background-color,border-color,opacity] duration-fast motion-reduce:transition-none ${
          enabled ? "border-border bg-surface-2" : "border-border/60 bg-surface opacity-70"
        }`}
      >
        <div className="flex gap-3">
          <input
            id={checkboxId}
            type="checkbox"
            checked={enabled}
            // Un modelo sin descargar no se puede elegir: la tarjeta de descarga
            // va debajo, en vez de dejar armar una selección que daría 400.
            disabled={locked || !step.installed}
            aria-describedby={descriptionId}
            onChange={(event) => onToggle(event.target.checked)}
            className="mt-1 h-3.5 w-3.5 shrink-0 accent-accent enabled:cursor-pointer disabled:cursor-not-allowed"
          />
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            {/* El nombre del modelo va DENTRO del label: los dos De-Echo son
                el mismo trabajo en dos intensidades, así que sin el nombre las
                dos filas tendrían el mismo nombre accesible y no habría cómo
                distinguirlas con un lector de pantalla. */}
            <label
              htmlFor={checkboxId}
              className={`flex flex-wrap items-center justify-between gap-2 text-sm ${
                locked || !step.installed ? "" : "cursor-pointer"
              } ${enabled ? "font-medium text-text" : "text-text-dim"}`}
            >
              {t(taskKeyFor(step))}
              <span className="shrink-0 rounded-sm border border-border px-1.5 py-0.5 text-[10px] font-normal uppercase tracking-wide text-text-faint">
                {step.name}
              </span>
            </label>
            <p id={descriptionId} className="text-xs leading-relaxed text-text-dim">
              {t(step.descriptionKey)}
            </p>
            {/* Decirlo ANTES de tildar, no después de que el otro se apague
                solo: la exclusión sale del catálogo y el usuario merece saber
                qué va a perder al elegir. */}
            {conflictNames.length > 0 && (
              <p className="text-xs text-text-faint">
                {t("audio.cleanup.replaces", { models: conflictNames.join(", ") })}
              </p>
            )}
          </div>
        </div>
        {!step.installed && (
          <PackDownload
            pack="karaoke"
            variant={step.id}
            reason={t("audio.karaoke.modelMissing", { name: step.name })}
          />
        )}
      </div>
    </div>
  );
}
