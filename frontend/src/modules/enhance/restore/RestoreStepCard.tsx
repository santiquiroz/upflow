import { useId, useState } from "react";
import { PackDownload } from "../../../components/PackDownload";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RestoreStepCapability, RestoreStepOptions } from "../../../lib/restoreApiTypes";
import { StepControlField } from "./RestoreStepControls";
import { stepUi, type StepUi } from "./restoreSteps";

interface RestoreStepCardProps {
  step: RestoreStepCapability;
  // 1-indexada: el orden es el de ejecucion del backend y no se puede cambiar.
  position: number;
  isLast: boolean;
  enabled: boolean;
  options: RestoreStepOptions;
  onToggle: (enabled: boolean) => void;
  onOptionChange: (option: string, value: unknown) => void;
}

function PositionMarker({ position, enabled, isLast }: { position: number; enabled: boolean; isLast: boolean }) {
  return (
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
  );
}

interface StepSettingsProps {
  ui: StepUi;
  options: RestoreStepOptions;
  onOptionChange: (option: string, value: unknown) => void;
}

function StepSettings({ ui, options, onOptionChange }: StepSettingsProps) {
  const { t } = useTranslation();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const advancedId = useId();
  const { intensity, advanced } = ui;

  return (
    <div className="flex flex-col gap-2 pl-6">
      <StepControlField
        control={intensity}
        value={options[intensity.option]}
        onChange={(value) => onOptionChange(intensity.option, value)}
      />
      {advanced.length > 0 && (
        <button
          type="button"
          aria-expanded={showAdvanced}
          aria-controls={advancedId}
          onClick={() => setShowAdvanced((open) => !open)}
          className="w-fit text-xs text-text-dim underline-offset-2 hover:text-text hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {t("restore.control.advanced")}
        </button>
      )}
      {showAdvanced && (
        <div id={advancedId} className="flex flex-col gap-2 border-l border-border pl-3">
          {advanced.map((control) => (
            <StepControlField
              key={control.option}
              control={control}
              value={options[control.option]}
              onChange={(value) => onOptionChange(control.option, value)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function RestoreStepCard({
  step,
  position,
  isLast,
  enabled,
  options,
  onToggle,
  onOptionChange,
}: RestoreStepCardProps) {
  const { t } = useTranslation();
  const checkboxId = useId();
  const descriptionId = useId();
  const ui = stepUi(step.id);

  return (
    <div className="flex gap-3">
      <PositionMarker position={position} enabled={enabled} isLast={isLast} />
      <div
        className={`mb-2 flex flex-1 flex-col gap-2 rounded border px-3 py-2.5 transition-[background-color,border-color,opacity] duration-fast motion-reduce:transition-none ${
          enabled ? "border-border bg-surface-2" : "border-border/60 bg-surface opacity-80"
        }`}
      >
        <div className="flex gap-3">
          <input
            id={checkboxId}
            type="checkbox"
            checked={enabled}
            disabled={!step.installed}
            aria-describedby={descriptionId}
            onChange={(event) => onToggle(event.target.checked)}
            className="mt-1 h-3.5 w-3.5 shrink-0 accent-accent enabled:cursor-pointer disabled:cursor-not-allowed"
          />
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <label
              htmlFor={checkboxId}
              className={`text-sm ${step.installed ? "cursor-pointer" : ""} ${enabled ? "font-medium text-text" : "text-text-dim"}`}
            >
              {t(step.labelKey)}
            </label>
            <p id={descriptionId} className="text-xs leading-relaxed text-text-dim">
              {t(step.descriptionKey)}
            </p>
            {/* Antes de tildar: quien enciende un paso que inventa tiene que saberlo al elegirlo. */}
            {step.warningKey && <p className="text-xs leading-relaxed text-warn">{t(step.warningKey)}</p>}
          </div>
        </div>
        {enabled && ui && <StepSettings ui={ui} options={options} onOptionChange={onOptionChange} />}
        {!step.installed && step.pack && (
          <PackDownload pack={step.pack} reason={t("restore.step.packMissing", { pack: step.pack })} />
        )}
      </div>
    </div>
  );
}
