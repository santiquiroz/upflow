import { Sparkles, UploadCloud } from "lucide-react";
import { useId, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { Link } from "react-router-dom";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { ModelPicker } from "../../components/ModelPicker";
import { useGenerationCapabilities, useGenerationJob, type GenerationJobPhase } from "../../hooks/useGenerationJob";
import { useTranslation } from "../../i18n/LocaleProvider";
import type {
  DeviceInfoResponse,
  GenerationModelSummary,
  InitImageResponse,
  ModelResponse,
} from "../../lib/apiTypes";
import {
  uploadGenerationInitImage,
  type CreateGenerationJobParams,
} from "../../services/generation";

const SIZE_OPTIONS = [256, 384, 512, 640, 768, 896, 1024];
const UPSCALE_SCALE_OPTIONS = [2, 3, 4];
const DEFAULT_STRENGTH = 0.6;
const STRENGTH_MIN = 0.05;
const STRENGTH_MAX = 1;
const STRENGTH_STEP = 0.05;

type GenerationMode = "text-to-image" | "image-to-image";

export const CPU_ONLY_WARNING =
  "No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos por imagen. ¿Continuar igual?";

function isJobBusy(phase: GenerationJobPhase): boolean {
  return phase === "uploading" || phase === "queued" || phase === "running";
}

function parseSeed(raw: string): number | null {
  if (raw.trim() === "") {
    return null;
  }
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) ? null : parsed;
}

// Mirrors ImagePanel's picker-to-params translation: a builtin model is
// selected by name (ncnn engine looks it up by name), an ONNX model by id.
function resolveUpscaleModelName(model: ModelResponse | null): string | null {
  return model?.kind === "builtin-ncnn" ? model.name : null;
}

function resolveUpscaleModelId(model: ModelResponse | null): string | null {
  return model?.kind === "onnx" ? model.id : null;
}

function UnavailableBanner({ reason }: { reason: string | null }) {
  return (
    <div role="alert" className="rounded border border-border bg-surface p-4 text-sm text-text-dim">
      {reason ?? "Generation is not available on this machine."}
    </div>
  );
}

function NoModelsHint() {
  return (
    <p className="text-sm text-text-dim">
      No generation models installed yet. Install one from the{" "}
      <Link to="/models" className="text-accent underline">
        Models
      </Link>{" "}
      page.
    </p>
  );
}

function ModelSelect({
  models,
  value,
  onChange,
}: {
  models: GenerationModelSummary[];
  value: string | null;
  onChange: (modelId: string | null) => void;
}) {
  if (models.length === 0) {
    return <NoModelsHint />;
  }
  return (
    <select
      id="generate-model"
      value={value ?? ""}
      onChange={(event) => onChange(event.target.value || null)}
      className="rounded border border-border bg-surface p-2 text-sm text-text"
    >
      <option value="">Select a model…</option>
      {models.map((model) => (
        // Una conversion en curso se VE (deshabilitada, con el aviso de demora):
        // sin esto, instalar un modelo desde el installer parecia no traer nada
        // durante los ~40 minutos que tarda la conversion.
        <option key={model.id} value={model.id} disabled={model.status !== "installed"}>
          {model.status === "converting" ? `${model.name} (convirtiendo… ~30-45 min)` : model.name}
        </option>
      ))}
    </select>
  );
}

function CpuConfirmBanner({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) {
  return (
    <div role="alert" className="flex flex-col gap-2 rounded border border-warn bg-surface-2 p-3 text-sm text-text">
      <p>{CPU_ONLY_WARNING}</p>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={onConfirm}
          className="inline-flex w-fit items-center rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg"
        >
          Continuar igual
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="inline-flex w-fit items-center rounded border border-border px-3 py-1.5 text-sm text-text-dim"
        >
          Cancelar
        </button>
      </div>
    </div>
  );
}

function SizeSelect({ id, value, onChange }: { id: string; value: number; onChange: (value: number) => void }) {
  return (
    <select
      id={id}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      className="rounded border border-border bg-surface p-2 text-sm text-text"
    >
      {SIZE_OPTIONS.map((size) => (
        <option key={size} value={size}>
          {size}
        </option>
      ))}
    </select>
  );
}

function InitImageDropzone({
  image,
  pendingFileName,
  isUploading,
  errorMessage,
  onFileSelected,
}: {
  image: InitImageResponse | null;
  pendingFileName: string | null;
  isUploading: boolean;
  errorMessage: string | null;
  onFileSelected: (file: File) => void;
}) {
  const { t } = useTranslation();

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files[0];
    if (dropped) {
      onFileSelected(dropped);
    }
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) {
      onFileSelected(selected);
    }
    event.target.value = "";
  }

  return (
    <div className="flex flex-col gap-2">
      <label
        htmlFor="generation-init-image-input"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
        className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-8 text-center transition-[border-color] duration-fast hover:border-accent"
      >
        <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
        {isUploading ? (
          <span role="status" className="text-sm text-text">
            {t("generation.initImage.uploading", { filename: pendingFileName ?? "" })}
          </span>
        ) : image ? (
          <>
            <span className="text-sm text-text">{image.originalFilename}</span>
            <span className="font-mono-tabular text-xs text-text-dim">
              {t("generation.initImage.dimensions", {
                width: image.width,
                height: image.height,
              })}
            </span>
            <span className="text-xs text-accent">{t("generation.initImage.replace")}</span>
          </>
        ) : (
          <span className="text-sm text-text">{t("generation.initImage.drop")}</span>
        )}
        <span className="text-xs text-text-faint">{t("generation.initImage.formats")}</span>
        <input
          id="generation-init-image-input"
          type="file"
          accept="image/*"
          aria-label={t("generation.initImage.inputLabel")}
          className="sr-only"
          onChange={handleInputChange}
        />
      </label>
      {errorMessage && (
        <p role="alert" className="text-xs text-danger">
          {errorMessage}
        </p>
      )}
    </div>
  );
}

function StrengthControl({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const { t } = useTranslation();
  const sliderId = useId();

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={sliderId} className="text-xs font-medium text-text-dim">
          {t("generation.strength.label")}
        </label>
        <span className="font-mono-tabular text-[10px] text-text-faint">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        id={sliderId}
        type="range"
        min={STRENGTH_MIN}
        max={STRENGTH_MAX}
        step={STRENGTH_STEP}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer accent-accent"
      />
      <p className="text-[11px] leading-relaxed text-text-faint">
        {t("generation.strength.hint")}
      </p>
    </div>
  );
}

export function GeneratePanel() {
  const { t } = useTranslation();
  const [mode, setMode] = useState<GenerationMode>("text-to-image");
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [modelId, setModelId] = useState<string | null>(null);
  const [steps, setSteps] = useState(25);
  const [guidance, setGuidance] = useState(7.5);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [seed, setSeed] = useState("");
  const [device, setDevice] = useState<string | null>(null);
  const [autoUpscale, setAutoUpscale] = useState(false);
  const [upscaleModel, setUpscaleModel] = useState<ModelResponse | null>(null);
  const [upscaleScale, setUpscaleScale] = useState(2);
  const [cpuConfirmPending, setCpuConfirmPending] = useState(false);
  const [initImage, setInitImage] = useState<InitImageResponse | null>(null);
  const [initImageFileName, setInitImageFileName] = useState<string | null>(null);
  const [isInitImageUploading, setIsInitImageUploading] = useState(false);
  const [initImageError, setInitImageError] = useState<string | null>(null);
  const [strength, setStrength] = useState(DEFAULT_STRENGTH);
  const initImageUploadSequence = useRef(0);

  const capabilitiesQuery = useGenerationCapabilities();
  const { phase, job, errorMessage, submit, cancel } = useGenerationJob();

  const capabilities = capabilitiesQuery.data;

  if (capabilities && !capabilities.available) {
    return <UnavailableBanner reason={capabilities.reason} />;
  }

  const models = capabilities?.models ?? [];
  const needsCpuConfirm = capabilities?.cpuOnly === true && (device === null || device === "cpu");

  function buildParams(): CreateGenerationJobParams {
    const params: CreateGenerationJobParams = {
      prompt,
      negativePrompt: negativePrompt.trim() === "" ? null : negativePrompt,
      modelId: modelId ?? "",
      steps,
      guidance,
      width,
      height,
      seed: parseSeed(seed),
      device,
      autoUpscale,
      upscaleModelName: autoUpscale ? resolveUpscaleModelName(upscaleModel) : null,
      upscaleModelId: autoUpscale ? resolveUpscaleModelId(upscaleModel) : null,
      upscaleScale: autoUpscale ? upscaleScale : null,
    };
    if (mode === "image-to-image" && initImage) {
      params.initImageToken = initImage.initImageToken;
      params.strength = strength;
    }
    return params;
  }

  function handleGenerate() {
    if (needsCpuConfirm && !cpuConfirmPending) {
      setCpuConfirmPending(true);
      return;
    }
    setCpuConfirmPending(false);
    submit(buildParams());
  }

  function handleCancelCpuConfirm() {
    setCpuConfirmPending(false);
  }

  function handleAutoUpscaleChange(event: ChangeEvent<HTMLInputElement>) {
    setAutoUpscale(event.target.checked);
  }

  function handleModeChange(nextMode: GenerationMode) {
    setMode(nextMode);
    setCpuConfirmPending(false);
  }

  async function handleInitImageSelected(file: File) {
    const sequence = initImageUploadSequence.current + 1;
    initImageUploadSequence.current = sequence;
    setInitImage(null);
    setInitImageFileName(file.name);
    setInitImageError(null);
    setIsInitImageUploading(true);

    try {
      const uploaded = await uploadGenerationInitImage(file);
      if (sequence === initImageUploadSequence.current) {
        setInitImage(uploaded);
      }
    } catch (error) {
      if (sequence === initImageUploadSequence.current) {
        const detail = error instanceof Error ? error.message : t("generation.initImage.unknownError");
        setInitImageError(t("generation.initImage.uploadFailed", { error: detail }));
      }
    } finally {
      if (sequence === initImageUploadSequence.current) {
        setIsInitImageUploading(false);
      }
    }
  }

  // A device change invalidates any already-armed CPU-only confirmation --
  // otherwise switching from cpu to a GPU device after arming would leave a
  // stale warning on screen for a device that no longer needs confirming.
  function handleDeviceChange(selected: DeviceInfoResponse) {
    setDevice(selected.id);
    setCpuConfirmPending(false);
  }

  const initImageRequired = mode === "image-to-image" && initImage === null;
  const canSubmit =
    prompt.trim() !== "" &&
    modelId !== null &&
    !isJobBusy(phase) &&
    !initImageRequired;

  return (
    <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
      <div className="flex flex-col gap-6">
        <fieldset className="flex flex-col gap-2">
          <legend className="text-xs font-medium text-text-dim">
            {t("generation.mode.label")}
          </legend>
          <div className="grid grid-cols-2 gap-2">
            <label className="flex cursor-pointer items-center gap-2 rounded border border-border bg-surface p-3 text-sm text-text">
              <input
                type="radio"
                name="generation-mode"
                value="text-to-image"
                checked={mode === "text-to-image"}
                onChange={() => handleModeChange("text-to-image")}
                className="h-3.5 w-3.5 accent-accent"
              />
              {t("generation.mode.textToImage")}
            </label>
            <label className="flex cursor-pointer items-center gap-2 rounded border border-border bg-surface p-3 text-sm text-text">
              <input
                type="radio"
                name="generation-mode"
                value="image-to-image"
                checked={mode === "image-to-image"}
                onChange={() => handleModeChange("image-to-image")}
                className="h-3.5 w-3.5 accent-accent"
              />
              {t("generation.mode.imageToImage")}
            </label>
          </div>
        </fieldset>
        {mode === "image-to-image" && (
          <div className="flex flex-col gap-4 rounded border border-border bg-surface-2 p-4">
            <InitImageDropzone
              image={initImage}
              pendingFileName={initImageFileName}
              isUploading={isInitImageUploading}
              errorMessage={initImageError}
              onFileSelected={(file) => void handleInitImageSelected(file)}
            />
            <StrengthControl value={strength} onChange={setStrength} />
          </div>
        )}
        <div className="flex flex-col gap-2">
          <label htmlFor="generate-prompt" className="text-xs font-medium text-text-dim">
            Prompt
          </label>
          <textarea
            id="generate-prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={3}
            className="rounded border border-border bg-surface p-2 text-sm text-text"
          />
        </div>
        <div className="flex flex-col gap-2">
          <label htmlFor="generate-negative-prompt" className="text-xs font-medium text-text-dim">
            Negative prompt
          </label>
          <textarea
            id="generate-negative-prompt"
            value={negativePrompt}
            onChange={(event) => setNegativePrompt(event.target.value)}
            rows={2}
            className="rounded border border-border bg-surface p-2 text-sm text-text"
          />
        </div>
        <div className="flex flex-col gap-2">
          <label htmlFor="generate-model" className="text-xs font-medium text-text-dim">
            Model
          </label>
          <ModelSelect models={models} value={modelId} onChange={setModelId} />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="generate-steps" className="text-xs font-medium text-text-dim">
              Steps
            </label>
            <input
              id="generate-steps"
              type="number"
              value={steps}
              onChange={(event) => setSteps(Number(event.target.value))}
              className="rounded border border-border bg-surface p-2 text-sm text-text"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="generate-guidance" className="text-xs font-medium text-text-dim">
              Guidance
            </label>
            <input
              id="generate-guidance"
              type="number"
              step="0.1"
              value={guidance}
              onChange={(event) => setGuidance(Number(event.target.value))}
              className="rounded border border-border bg-surface p-2 text-sm text-text"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="generate-width" className="text-xs font-medium text-text-dim">
              Width
            </label>
            <SizeSelect id="generate-width" value={width} onChange={setWidth} />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="generate-height" className="text-xs font-medium text-text-dim">
              Height
            </label>
            <SizeSelect id="generate-height" value={height} onChange={setHeight} />
          </div>
          <div className="flex flex-col gap-2">
            <label htmlFor="generate-seed" className="text-xs font-medium text-text-dim">
              Seed
            </label>
            <input
              id="generate-seed"
              type="text"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
              placeholder="Random"
              className="rounded border border-border bg-surface p-2 text-sm text-text"
            />
          </div>
        </div>
        <DevicePicker value={device} onChange={handleDeviceChange} requiresGpu={false} allowAuto={false} />
        <div className="flex flex-col gap-3 rounded border border-border bg-surface p-3">
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="checkbox"
              checked={autoUpscale}
              onChange={handleAutoUpscaleChange}
              className="h-3.5 w-3.5 accent-accent"
            />
            Escalar automáticamente al terminar
          </label>
          {autoUpscale && (
            <div className="flex flex-col gap-3">
              <ModelPicker value={upscaleModel?.id ?? null} onChange={setUpscaleModel} />
              <div className="flex flex-col gap-2">
                <label htmlFor="generate-upscale-scale" className="text-xs font-medium text-text-dim">
                  Scale
                </label>
                <select
                  id="generate-upscale-scale"
                  value={upscaleScale}
                  onChange={(event) => setUpscaleScale(Number(event.target.value))}
                  className="rounded border border-border bg-surface p-2 text-sm text-text"
                >
                  {UPSCALE_SCALE_OPTIONS.map((scale) => (
                    <option key={scale} value={scale}>
                      {scale}x
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>
        <div className="flex flex-col gap-2">
          {cpuConfirmPending && needsCpuConfirm && (
            <CpuConfirmBanner onConfirm={handleGenerate} onCancel={handleCancelCpuConfirm} />
          )}
          {initImageRequired && (
            <p role="status" className="text-xs text-warn">
              {t("generation.initImage.required")}
            </p>
          )}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={!canSubmit}
            className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            <Sparkles aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            Generate
          </button>
        </div>
      </div>
      <JobCard phase={phase} job={job} fileName={prompt.slice(0, 60) || undefined} errorMessage={errorMessage} onCancel={cancel} />
    </div>
  );
}
