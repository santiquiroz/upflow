import { useTranslation } from "../../i18n/LocaleProvider";
import type { CleanupStep } from "../../lib/apiTypes";
import { CleanupStepCard } from "./CleanupStepCard";
import type { CleanupSelection } from "./useCleanupSelection";

// Tres claves en vez de una con {{count}}: "1 pasada" y "3 pasadas" no comparten
// forma y translate() no pluraliza. Mismo criterio que voiceSummaryKey.
export function cleanupSummaryKey(count: number): string {
  if (count === 0) return "audio.cleanup.summary.none";
  if (count === 1) return "audio.cleanup.summary.one";
  return "audio.cleanup.summary.many";
}

interface CleanupChainPanelProps {
  /** No vacío: la sección entera no se renderiza sin catálogo (ver AudioPanel). */
  steps: CleanupStep[];
  selection: CleanupSelection;
}

function namesOf(steps: CleanupStep[], ids: string[]): string[] {
  return steps.filter((step) => ids.includes(step.id)).map((step) => step.name);
}

export function CleanupChainPanel({ steps: catalog, selection }: CleanupChainPanelProps) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-3">
      {/* Opt-in explícito, igual que la cadena de voz: los pasos se ven aunque
          esté apagada, para que se descubra que existen. */}
      <label className="flex cursor-pointer items-start gap-2.5">
        <input
          type="checkbox"
          checked={selection.active}
          onChange={(event) => selection.setActive(event.target.checked)}
          className="mt-0.5 h-3.5 w-3.5 shrink-0 cursor-pointer accent-accent"
        />
        <span className="flex flex-col gap-1">
          <span className="text-sm font-medium text-text">{t("audio.cleanup.activate")}</span>
          <span className="text-xs leading-relaxed text-text-faint">
            {t("audio.cleanup.chainHint")}
          </span>
        </span>
      </label>
      <div
        className={`flex flex-col transition-opacity duration-fast motion-reduce:transition-none ${
          selection.active ? "" : "opacity-50"
        }`}
      >
        {catalog.map((step, index) => (
          <CleanupStepCard
            key={step.id}
            step={step}
            position={index + 1}
            isLast={index === catalog.length - 1}
            enabled={selection.isEnabled(step.id)}
            locked={!selection.active}
            conflictNames={namesOf(catalog, selection.conflictsOf(step.id))}
            onToggle={(enabled) => selection.toggleStep(step.id, enabled)}
          />
        ))}
      </div>
      {/* Avisar, no bloquear: encadenar tres máscaras puede ser exactamente lo
          que alguien quiere, pero no debería enterarse al escuchar el resultado. */}
      {selection.isOverprocessing && (
        <p role="status" className="text-xs text-warn">
          {t("audio.cleanup.overprocessed", { count: selection.enabledIds.length })}
        </p>
      )}
      <p className="text-xs text-text-faint">{t("audio.cleanup.singleOutputNote")}</p>
    </div>
  );
}
