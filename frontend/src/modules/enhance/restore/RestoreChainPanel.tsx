import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RestoreCapabilities } from "../../../lib/restoreApiTypes";
import { RestorePresetPicker } from "./RestorePresetPicker";
import { RestoreStepCard } from "./RestoreStepCard";
import type { RestoreSelection } from "./useRestoreSelection";

const PERCENT = 100;

interface RestoreChainPanelProps {
  capabilities: RestoreCapabilities;
  suggestedPresets: string[];
  selection: RestoreSelection;
}

export function RestoreChainPanel({ capabilities, suggestedPresets, selection }: RestoreChainPanelProps) {
  const { t } = useTranslation();
  const { steps } = capabilities;

  return (
    <div className="flex flex-col gap-4">
      <RestorePresetPicker
        presets={capabilities.presets}
        selectedId={selection.presetId}
        suggestedIds={suggestedPresets}
        customized={selection.customized}
        onSelect={selection.choosePreset}
      />
      <div className="flex flex-col">
        <p className="mb-2 text-xs leading-relaxed text-text-faint">{t("restore.chain.orderHint")}</p>
        {steps.map((step, index) => (
          <RestoreStepCard
            key={step.id}
            step={step}
            position={index + 1}
            isLast={index === steps.length - 1}
            enabled={selection.isEnabled(step.id)}
            options={selection.optionsOf(step.id)}
            onToggle={(enabled) => selection.toggleStep(step.id, enabled)}
            onOptionChange={(option, value) => selection.setOption(step.id, option, value)}
          />
        ))}
      </div>
      {/* Avisar, no bloquear: el backend acepta la combinacion. */}
      {selection.isOverprocessing && (
        <p role="status" className="text-xs text-warn">
          {t("restore.warning.overprocessed", { limit: Math.round(capabilities.halftoneDenoiseLimit * PERCENT) })}
        </p>
      )}
    </div>
  );
}
