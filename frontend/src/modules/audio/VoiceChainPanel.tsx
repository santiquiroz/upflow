import { useTranslation } from "../../i18n/LocaleProvider";
import type { VoiceCatalog } from "../../lib/apiTypes";
import { DeliveryTargetPicker } from "./DeliveryTargetPicker";
import { PresenceControl } from "./PresenceControl";
import { VoiceStepCard } from "./VoiceStepCard";
import {
  LOUDNESS_STEP_ID,
  PRESENCE_STEP_ID,
  type VoiceSelection,
} from "./useVoiceSelection";

// Tres claves en vez de una con {{count}}: "1 paso" y "3 pasos" no comparten
// forma, y translate() no hace pluralizacion. Explicito y el test de paridad de
// catalogos lo cubre.
export function voiceSummaryKey(count: number): string {
  if (count === 0) return "voice.summary.none";
  if (count === 1) return "voice.summary.one";
  return "voice.summary.many";
}

interface VoiceChainPanelProps {
  catalog: VoiceCatalog | undefined;
  isLoading: boolean;
  isError: boolean;
  selection: VoiceSelection;
}

export function VoiceChainPanel({
  catalog,
  isLoading,
  isError,
  selection,
}: VoiceChainPanelProps) {
  const { t } = useTranslation();

  if (isLoading) {
    return <p className="text-sm text-text-dim">{t("voice.loading")}</p>;
  }
  if (isError || !catalog) {
    return (
      <p role="status" className="text-sm text-danger">
        {t("voice.loadFailed")}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Opt-in explicito. Los pasos se ven igual estando apagado, para que se
          descubra que existen, pero no se pueden elegir: asi nadie termina con
          procesado de voz en un job donde solo pidio quitar ruido. */}
      <label className="flex cursor-pointer items-start gap-2.5">
        <input
          type="checkbox"
          checked={selection.active}
          onChange={(event) => selection.setActive(event.target.checked)}
          className="mt-0.5 h-3.5 w-3.5 shrink-0 cursor-pointer accent-accent"
        />
        <span className="flex flex-col gap-1">
          <span className="text-sm font-medium text-text">{t("voice.activate")}</span>
          <span className="text-xs leading-relaxed text-text-faint">
            {t("voice.chainHint")}
          </span>
        </span>
      </label>
      <div
        className={`flex flex-col transition-opacity duration-fast motion-reduce:transition-none ${
          selection.active ? "" : "opacity-50"
        }`}
      >
        {catalog.steps.map((step, index) => (
          <VoiceStepCard
            key={step.id}
            step={step}
            position={index + 1}
            isLast={index === catalog.steps.length - 1}
            enabled={selection.isEnabled(step.id)}
            locked={!selection.active}
            onToggle={(enabled) => selection.toggleStep(step.id, enabled)}
          >
            {step.id === PRESENCE_STEP_ID && (
              <PresenceControl
                value={selection.presenceDb}
                onChange={selection.setPresenceDb}
              />
            )}
            {step.id === LOUDNESS_STEP_ID && (
              <DeliveryTargetPicker
                deliveries={catalog.deliveries}
                value={selection.delivery}
                onChange={selection.setDelivery}
              />
            )}
          </VoiceStepCard>
        ))}
      </div>
      {selection.needsDelivery && (
        <p role="status" className="text-xs text-warn">
          {t("voice.deliveryRequired")}
        </p>
      )}
    </div>
  );
}
