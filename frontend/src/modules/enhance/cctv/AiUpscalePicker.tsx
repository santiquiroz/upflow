import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvAiUpscaleModel } from "../../../services/cctv";
import { aiUpscaleFor, generativeTagKey, type AiUpscaleChoice } from "./cctvAiUpscale";
import { CctvOptionGroup } from "./CctvOptionGroup";

const SELECT_ID = "cctv-ai-upscale-model";
const NONE = "";
const FIELD_CLASS =
  "w-full max-w-md rounded-sm border border-border bg-surface px-2 py-1.5 text-sm text-text disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

interface AiUpscalePickerProps {
  models: readonly CctvAiUpscaleModel[];
  value: AiUpscaleChoice | null;
  scaleHint: number | null;
  onChange: (value: AiUpscaleChoice | null) => void;
}

function ScaleChoice({
  model,
  value,
  onChange,
}: {
  model: CctvAiUpscaleModel;
  value: AiUpscaleChoice;
  onChange: (value: AiUpscaleChoice) => void;
}) {
  const { t } = useTranslation();
  const options = model.scales.map((scale) => ({ value: scale, label: t("cctv.scaleOption", { scale }) }));
  return (
    <CctvOptionGroup
      legend={t("cctv.ai.upscale.scale")}
      options={options}
      value={value.scale}
      onChange={(scale) => onChange({ ...value, scale })}
    />
  );
}

export function AiUpscalePicker({ models, value, scaleHint, onChange }: AiUpscalePickerProps) {
  const { t } = useTranslation();
  const selected = models.find((model) => model.id === value?.modelId) ?? null;
  const preferredScale = value?.scale ?? scaleHint;
  return (
    <div className="flex flex-col gap-2">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.ai.upscale.legend")}</span>
      <label htmlFor={SELECT_ID} className="text-xs text-text-dim">
        {t("cctv.ai.upscale.model")}
      </label>
      <select
        id={SELECT_ID}
        value={value?.modelId ?? NONE}
        disabled={models.length === 0}
        onChange={(event) => onChange(aiUpscaleFor(models, event.target.value || null, preferredScale))}
        className={FIELD_CLASS}
      >
        <option value={NONE}>{t("cctv.ai.upscale.none")}</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {t("cctv.ai.upscale.option", { label: model.label, tag: t(generativeTagKey(model.generative)) })}
          </option>
        ))}
      </select>
      {models.length === 0 && <p className="text-xs text-text-faint">{t("cctv.ai.upscale.noModels")}</p>}
      {selected && value && <ScaleChoice model={selected} value={value} onChange={onChange} />}
      {selected && <p className="text-xs text-warn">{t("cctv.ai.upscale.hint")}</p>}
    </div>
  );
}
