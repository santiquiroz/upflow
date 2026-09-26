import { useId } from "react";
import type { TranslationParams } from "../../../i18n";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { ChoiceControl, SliderControl, StepControl, ToggleControl } from "./restoreSteps";

interface FieldProps<Control> {
  control: Control;
  value: unknown;
  onChange: (value: unknown) => void;
}

const PERCENT = 100;

type Translate = (key: string, params?: TranslationParams) => string;

function sliderText(control: SliderControl, value: number, t: Translate): string {
  if (control.format === "pixels") {
    return t("restore.control.pixels", { value: value > 0 ? `+${value}` : value });
  }
  return t("restore.control.percent", { value: Math.round(value * PERCENT) });
}

interface SliderFieldProps extends FieldProps<SliderControl> {
  disabled?: boolean;
}

export function SliderField({ control, value, onChange, disabled = false }: SliderFieldProps) {
  const { t } = useTranslation();
  const sliderId = useId();
  const numeric = Number(value);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={sliderId} className="text-xs font-medium text-text-dim">
          {t(control.labelKey)}
        </label>
        <span className="font-mono-tabular text-[10px] text-text-faint">{sliderText(control, numeric, t)}</span>
      </div>
      <input
        id={sliderId}
        type="range"
        min={control.min}
        max={control.max}
        step={control.step}
        value={numeric}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full accent-accent enabled:cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
      />
    </div>
  );
}

function ToggleField({ control, value, onChange }: FieldProps<ToggleControl>) {
  const { t } = useTranslation();

  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-text-dim">
      <input
        type="checkbox"
        checked={value === true}
        onChange={(event) => onChange(event.target.checked)}
        className="h-3.5 w-3.5 shrink-0 cursor-pointer accent-accent"
      />
      {t(control.labelKey)}
    </label>
  );
}

function ChoiceField({ control, value, onChange }: FieldProps<ChoiceControl>) {
  const { t } = useTranslation();
  const selectId = useId();

  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={selectId} className="text-xs font-medium text-text-dim">
        {t(control.labelKey)}
      </label>
      <select
        id={selectId}
        value={String(value)}
        onChange={(event) => onChange(event.target.value)}
        className="w-fit rounded border border-border bg-surface px-2 py-1 text-xs text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {control.choices.map((choice) => (
          <option key={choice} value={choice}>
            {t(`${control.labelKey}.${choice}`)}
          </option>
        ))}
      </select>
    </div>
  );
}

export function StepControlField({ control, value, onChange }: FieldProps<StepControl>) {
  if (control.kind === "slider") {
    return <SliderField control={control} value={value} onChange={onChange} />;
  }
  if (control.kind === "toggle") {
    return <ToggleField control={control} value={value} onChange={onChange} />;
  }
  return <ChoiceField control={control} value={value} onChange={onChange} />;
}
