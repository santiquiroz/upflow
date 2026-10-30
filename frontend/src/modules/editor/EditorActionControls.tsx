import { AccordionSection } from "../../components/AccordionSection";
import { DevicePicker } from "../../components/DevicePicker";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { DeviceInfoResponse, GenerationModelSummary, InitImageResponse } from "../../lib/apiTypes";
import { canReplace } from "./eraseJobParams";
import { InsertObjectPanel } from "./InsertObjectPanel";
import { InpaintModelSelect, type EditMode } from "./InpaintModelSelect";

export function EditorActionControls({
  baseInfo,
  models,
  mode,
  setMode,
  prompt,
  setPrompt,
  modelId,
  setModelId,
  device,
  setDevice,
  erasePromptText,
  setErasePromptText,
  eraseNegative,
  setEraseNegative,
  replaceNegative,
  setReplaceNegative,
  steps,
  setSteps,
  guidance,
  setGuidance,
  seedText,
  setSeedText,
  strength,
  setStrength,
  isEraseOnlyModel,
  autoErase,
  canSubmit,
  onGenerate,
  onBaseSelected,
}: {
  baseInfo: InitImageResponse;
  models: GenerationModelSummary[];
  mode: EditMode;
  setMode: (mode: EditMode) => void;
  prompt: string;
  setPrompt: (prompt: string) => void;
  modelId: string | null;
  setModelId: (modelId: string | null) => void;
  device: string | null;
  setDevice: (device: string) => void;
  erasePromptText: string;
  setErasePromptText: (prompt: string) => void;
  eraseNegative: string;
  setEraseNegative: (prompt: string) => void;
  replaceNegative: string;
  setReplaceNegative: (prompt: string) => void;
  steps: number;
  setSteps: (steps: number) => void;
  guidance: number;
  setGuidance: (guidance: number) => void;
  seedText: string;
  setSeedText: (seed: string) => void;
  strength: number;
  setStrength: (strength: number) => void;
  isEraseOnlyModel: boolean;
  autoErase: boolean;
  canSubmit: boolean;
  onGenerate: () => void;
  onBaseSelected: (file: File) => void;
}) {
  const { t } = useTranslation();
  const replaceAvailable = canReplace(models);
  return (
    <>
      <div className="flex gap-1 rounded border border-border bg-bg p-1" role="radiogroup" aria-label={t("editor.mode.label")}>
        {(["erase", "replace"] as const).map((value) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={mode === value}
            disabled={value === "replace" && !replaceAvailable}
            onClick={() => setMode(value)}
            className={`min-h-9 flex-1 rounded px-3 py-1.5 text-sm transition-colors ${
              mode === value ? "bg-surface-2 font-medium text-text" : "text-text-dim hover:text-text"
            } disabled:cursor-not-allowed disabled:opacity-50`}
          >
            {t(`editor.mode.${value}`)}
            {value === "replace" && !replaceAvailable && (
              <span aria-hidden="true" className="ml-1 text-xs">
                {t("editor.replace.needsSdxl")}
              </span>
            )}
          </button>
        ))}
      </div>

      {mode === "replace" && (
        <label className="mt-3 flex flex-col gap-1 text-sm text-text">
          {t("editor.prompt.label")}
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={t("editor.prompt.placeholder")}
            rows={2}
            className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
          />
        </label>
      )}

      <div className="mt-3 flex flex-col gap-3">
        <InpaintModelSelect
          models={models}
          value={modelId}
          onChange={setModelId}
          label={t("editor.model.label")}
          mode={mode}
          hint={autoErase ? t("editor.erase.autoHint") : undefined}
        />
        <DevicePicker
          value={device}
          onChange={(selected: DeviceInfoResponse) => setDevice(selected.id)}
          requiresGpu={false}
          allowAuto={false}
        />
      </div>

      {!isEraseOnlyModel && (
        <AccordionSection
          title={t("editor.advanced.title")}
          summary={t("editor.advanced.summary", { steps: String(steps), guidance: String(guidance) })}
          tooltip={t("editor.advanced.tooltip")}
        >
          <div className="flex flex-col gap-3">
            {mode === "erase" && (
              <label className="flex flex-col gap-1 text-sm text-text">
                {t("editor.advanced.fillPrompt")}
                <textarea
                  value={erasePromptText}
                  onChange={(event) => setErasePromptText(event.target.value)}
                  rows={2}
                  className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
                />
              </label>
            )}
            <label className="flex flex-col gap-1 text-sm text-text">
              {t("editor.advanced.negativePrompt")}
              <textarea
                value={mode === "erase" ? eraseNegative : replaceNegative}
                onChange={(event) =>
                  mode === "erase"
                    ? setEraseNegative(event.target.value)
                    : setReplaceNegative(event.target.value)
                }
                rows={2}
                className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
              />
            </label>
            <div className="grid grid-cols-3 gap-3 max-[700px]:grid-cols-1">
              <label className="flex flex-col gap-1 text-sm text-text">
                {t("editor.advanced.steps")}
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={steps}
                  onChange={(event) => setSteps(Math.min(100, Math.max(1, Number(event.target.value) || 1)))}
                  className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm text-text">
                {t("editor.advanced.guidance")}
                <input
                  type="number"
                  min={0}
                  max={30}
                  step={0.5}
                  value={guidance}
                  onChange={(event) =>
                    setGuidance(Math.min(30, Math.max(0, Number(event.target.value) || 0)))
                  }
                  className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
                />
              </label>
              <label className="flex flex-col gap-1 text-sm text-text">
                {t("editor.advanced.seed")}
                <input
                  type="text"
                  inputMode="numeric"
                  value={seedText}
                  onChange={(event) => setSeedText(event.target.value.replace(/[^0-9]/g, ""))}
                  placeholder={t("editor.advanced.seedPlaceholder")}
                  className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
                />
              </label>
            </div>
            <label className="flex flex-col gap-1 text-sm text-text">
              <span>
                {t("editor.advanced.strength")}{" "}
                <span className="font-mono-tabular text-text-dim">{strength.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={0.05}
                max={1}
                step={0.05}
                value={strength}
                onChange={(event) => setStrength(Number(event.target.value))}
              />
              <span className="text-xs text-text-faint">{t("editor.advanced.strengthHint")}</span>
            </label>
          </div>
        </AccordionSection>
      )}
      {isEraseOnlyModel && <p className="mt-3 text-xs text-text-faint">{t("editor.eraser.hint")}</p>}

      <div className="mt-3">
        <InsertObjectPanel targetInfo={baseInfo} />
      </div>

      <button
        type="button"
        disabled={!canSubmit}
        onClick={onGenerate}
        className="mt-3 min-h-10 w-full rounded bg-accent px-5 py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-40"
      >
        {mode === "erase" ? t("editor.submit.erase") : t("editor.submit.replace")}
      </button>
      <label
        htmlFor="editor-base-image-change"
        className="mt-3 block min-h-9 cursor-pointer py-1.5 text-sm text-text-dim underline hover:text-text"
      >
        {t("editor.changeImage")}
      </label>
      <input
        id="editor-base-image-change"
        type="file"
        accept="image/*"
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onBaseSelected(file);
          event.target.value = "";
        }}
      />
    </>
  );
}
