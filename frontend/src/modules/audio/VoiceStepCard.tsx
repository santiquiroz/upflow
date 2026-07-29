import { Cpu, Sparkles } from "lucide-react";
import { useId, type ReactNode } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { VoiceStep } from "../../lib/apiTypes";

interface VoiceStepCardProps {
  step: VoiceStep;
  // Posicion en la cadena, 1-indexada. Se muestra porque el orden NO es
  // arbitrario: tiene causalidad documentada (ver build_filter_chain) y una lista
  // plana de checkboxes invitaria a creer que da lo mismo en que orden se aplican.
  position: number;
  isLast: boolean;
  enabled: boolean;
  /** La mejora de voz entera esta apagada: el paso se muestra pero no se elige. */
  locked: boolean;
  onToggle: (enabled: boolean) => void;
  children?: ReactNode;
}

// El badge dice que EJECUTA el paso, no ofrece elegir: hoy los 6 pasos son
// filtros de ffmpeg y el backend todavia no publica variantes de modelo, asi que
// un selector DSP/IA seria un control sin nada detras.
function StrategyBadge({ kind }: { kind: string }) {
  const { t } = useTranslation();
  const isModel = kind === "model";
  const Icon = isModel ? Sparkles : Cpu;

  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1 rounded-sm border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
        isModel ? "border-accent text-accent" : "border-border text-text-faint"
      }`}
    >
      <Icon aria-hidden="true" className="h-3 w-3" strokeWidth={2} />
      {t(isModel ? "voice.strategy.model" : "voice.strategy.dsp")}
    </span>
  );
}

export function VoiceStepCard({
  step,
  position,
  isLast,
  enabled,
  locked,
  onToggle,
  children,
}: VoiceStepCardProps) {
  const { t } = useTranslation();
  const checkboxId = useId();
  const descriptionId = useId();

  return (
    <div className="flex gap-3">
      {/* Columna del espinazo: numero de orden mas la linea que conecta los
          pasos. Un paso apagado se queda EN SU LUGAR, atenuado, en vez de
          desaparecer: asi el orden sigue legible y se ve donde encajaria. */}
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
            disabled={locked}
            aria-describedby={descriptionId}
            onChange={(event) => onToggle(event.target.checked)}
            className="mt-1 h-3.5 w-3.5 shrink-0 accent-accent enabled:cursor-pointer disabled:cursor-not-allowed"
          />
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <label
                htmlFor={checkboxId}
                className={`text-sm ${locked ? "" : "cursor-pointer"} ${
                  enabled ? "font-medium text-text" : "text-text-dim"
                }`}
              >
                {t(step.labelKey)}
              </label>
              <StrategyBadge kind={step.kind} />
            </div>
            {/* La explicacion va SIEMPRE visible, no en un tooltip: usar hover
                como unico mecanismo para informacion que hace falta para decidir
                es un fallo de accesibilidad conocido, y ademas este panel lo usan
                tanto productores de audio como gente que nunca vio un de-esser. */}
            <p id={descriptionId} className="text-xs leading-relaxed text-text-dim">
              {t(step.descriptionKey)}
            </p>
          </div>
        </div>
        {/* La configuracion anidada queda FUERA del label del checkbox: dentro,
            un click en un radio del destino tambien toggearia el paso. */}
        {enabled && children && <div className="pl-6">{children}</div>}
      </div>
    </div>
  );
}
