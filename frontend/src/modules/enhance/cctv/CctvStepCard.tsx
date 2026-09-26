import { useTranslation } from "../../../i18n/LocaleProvider";
import type {
  CctvEnumParamSchema,
  CctvFilterSchema,
  CctvNumberParamSchema,
  CctvParamSchema,
  CctvParamValue,
  CctvParams,
  CctvStepSchema,
} from "../../../services/cctv";
import { findFilter, matchingPreset, missingParams, paramValue, stepWarningKey, type StepChoice } from "./cctvSteps";
import { translateOr, type Translate } from "./cctvText";

const CONTROL_CLASS =
  "rounded-sm border border-border bg-surface px-2 py-1 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const FIELD_CLASS = `font-mono-tabular ${CONTROL_CLASS}`;
const CUSTOM_PRESET = "";

interface StepCardProps {
  step: CctvStepSchema;
  choice: StepChoice | null;
  onToggle: (enabled: boolean) => void;
  onFilterChange: (filter: string) => void;
  onParamChange: (name: string, value: CctvParamValue | null) => void;
  onParamsReplace: (params: CctvParams) => void;
}

function unavailableText(t: Translate, step: CctvStepSchema): string {
  const names = step.filters.map((filter) => filter.name).join(", ");
  return t("cctv.filterUnavailable", { filter: names });
}

function parseNumber(param: CctvNumberParamSchema, raw: string): number | null {
  if (raw.trim() === "") {
    return null;
  }
  const value = Number(raw);
  return param.type === "int" ? Math.trunc(value) : value;
}

function enumValue(param: CctvEnumParamSchema, raw: string): CctvParamValue {
  return param.choices.find((choice) => String(choice) === raw) ?? raw;
}

function ParamField({
  stepId,
  param,
  choice,
  onChange,
}: {
  stepId: string;
  param: CctvParamSchema;
  choice: StepChoice;
  onChange: (name: string, value: CctvParamValue | null) => void;
}) {
  const id = `cctv-${stepId}-${param.name}`;
  const value = paramValue(param, choice);
  return (
    <label htmlFor={id} className="flex flex-col gap-1">
      <span className="font-mono-tabular text-xs text-text-dim">{param.name}</span>
      {param.type === "enum" ? (
        <select id={id} value={String(value)} onChange={(event) => onChange(param.name, enumValue(param, event.target.value))} className={FIELD_CLASS}>
          {param.choices.map((option) => (
            <option key={String(option)} value={String(option)}>
              {option}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type="number"
          min={param.min}
          max={param.max}
          step={param.type === "int" ? 1 : "any"}
          value={value ?? ""}
          onChange={(event) => onChange(param.name, parseNumber(param, event.target.value))}
          className={`${FIELD_CLASS} w-28`}
        />
      )}
    </label>
  );
}

function FilterSelect({
  step,
  choice,
  onChange,
}: {
  step: CctvStepSchema;
  choice: StepChoice;
  onChange: (filter: string) => void;
}) {
  const { t } = useTranslation();
  if (step.filters.length < 2) {
    return null;
  }
  const id = `cctv-${step.id}-filter`;
  return (
    <label htmlFor={id} className="flex flex-col gap-1">
      <span className="text-xs text-text-dim">{t("cctv.step.filter")}</span>
      <select id={id} value={choice.filter} onChange={(event) => onChange(event.target.value)} className={FIELD_CLASS}>
        {step.filters.map((filter) => (
          <option key={filter.name} value={filter.name} disabled={!filter.available}>
            {filter.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function FilterDescription({ filter }: { filter: CctvFilterSchema }) {
  const { t } = useTranslation();
  return (
    <p className="text-xs text-text-dim">
      {translateOr(t, filter.descriptionKey, filter.description)}{" "}
      {filter.docUrl && (
        <a href={filter.docUrl} target="_blank" rel="noreferrer" className="text-accent underline-offset-2 hover:underline">
          {t("cctv.step.docs")}
        </a>
      )}
    </p>
  );
}

function PresetSelect({
  stepId,
  filter,
  choice,
  onApply,
}: {
  stepId: string;
  filter: CctvFilterSchema;
  choice: StepChoice;
  onApply: (params: CctvParams) => void;
}) {
  const { t } = useTranslation();
  const presets = filter.presets ?? [];
  if (presets.length === 0) {
    return null;
  }
  const id = `cctv-${stepId}-preset`;
  const hintId = `${id}-hint`;
  const apply = (name: string) => {
    const preset = presets.find((candidate) => candidate.name === name);
    if (preset) onApply(preset.params);
  };
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-xs text-text-dim">
        {t("cctv.step.preset")}
      </label>
      <select
        id={id}
        value={matchingPreset(filter, choice)?.name ?? CUSTOM_PRESET}
        aria-describedby={hintId}
        onChange={(event) => apply(event.target.value)}
        className={`${CONTROL_CLASS} max-w-full`}
      >
        <option value={CUSTOM_PRESET} disabled>
          {t("cctv.step.presetCustom")}
        </option>
        {presets.map((preset) => (
          <option key={preset.name} value={preset.name}>
            {translateOr(t, preset.labelKey, preset.label)}
          </option>
        ))}
      </select>
      <p id={hintId} className="text-xs text-text-faint">
        {t("cctv.step.presetHint")}
      </p>
    </div>
  );
}

function StepNotes({ step, choice }: { step: CctvStepSchema; choice: StepChoice }) {
  const { t } = useTranslation();
  const warningKey = stepWarningKey(step.id, choice);
  const missing = missingParams(step, choice);
  return (
    <>
      {warningKey && <p className="text-xs text-warn">{t(warningKey)}</p>}
      {missing.length > 0 && <p className="text-xs text-warn">{t("cctv.step.missingParams", { params: missing.join(", ") })}</p>}
    </>
  );
}

function StepSettings({
  step,
  choice,
  onFilterChange,
  onParamChange,
  onParamsReplace,
}: Omit<StepCardProps, "onToggle"> & { choice: StepChoice }) {
  const { t } = useTranslation();
  const filter = findFilter(step, choice.filter);
  return (
    <>
      {filter && <FilterDescription filter={filter} />}
      <StepNotes step={step} choice={choice} />
      {filter && <PresetSelect stepId={step.id} filter={filter} choice={choice} onApply={onParamsReplace} />}
      <details className="text-sm">
        <summary className="cursor-pointer text-xs text-text-dim">{t("cctv.step.advanced")}</summary>
        <div className="mt-2 flex flex-wrap gap-3">
          <FilterSelect step={step} choice={choice} onChange={onFilterChange} />
          {(filter?.params ?? []).map((param) => (
            <ParamField key={param.name} stepId={step.id} param={param} choice={choice} onChange={onParamChange} />
          ))}
        </div>
      </details>
    </>
  );
}

export function CctvStepCard({ step, choice, onToggle, onFilterChange, onParamChange, onParamsReplace }: StepCardProps) {
  const { t } = useTranslation();
  const checkboxId = `cctv-step-${step.id}`;
  return (
    <li className="flex flex-col gap-1.5 rounded border border-border bg-surface p-3">
      <label htmlFor={checkboxId} className="flex items-center gap-2 text-sm text-text">
        <input
          id={checkboxId}
          type="checkbox"
          checked={choice !== null}
          disabled={!step.available}
          onChange={(event) => onToggle(event.target.checked)}
          className="h-3.5 w-3.5 accent-accent"
        />
        {translateOr(t, step.labelKey, step.label)}
      </label>
      {!step.available && <p className="text-xs text-warn">{unavailableText(t, step)}</p>}
      {choice && (
        <StepSettings
          step={step}
          choice={choice}
          onFilterChange={onFilterChange}
          onParamChange={onParamChange}
          onParamsReplace={onParamsReplace}
        />
      )}
    </li>
  );
}
