import { Sparkles } from "lucide-react";
import { useState, type ChangeEvent } from "react";
import { Link } from "react-router-dom";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { ModelPicker } from "../../components/ModelPicker";
import { useGenerationCapabilities, useGenerationJob, type GenerationJobPhase } from "../../hooks/useGenerationJob";
import type { CreateGenerationJobParams } from "../../services/generation";
import type { GenerationModelSummary, ModelResponse } from "../../lib/apiTypes";

const SIZE_OPTIONS = [256, 384, 512, 640, 768, 896, 1024];
const UPSCALE_SCALE_OPTIONS = [2, 3, 4];

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
        <option key={model.id} value={model.id}>
          {model.name}
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

export function GeneratePanel() {
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

  const capabilitiesQuery = useGenerationCapabilities();
  const { phase, job, errorMessage, submit, cancel } = useGenerationJob();

  const capabilities = capabilitiesQuery.data;

  if (capabilities && !capabilities.available) {
    return <UnavailableBanner reason={capabilities.reason} />;
  }

  const models = capabilities?.models ?? [];
  const needsCpuConfirm = capabilities?.cpuOnly === true && (device === null || device === "cpu");

  function buildParams(): CreateGenerationJobParams {
    return {
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

  const canSubmit = prompt.trim() !== "" && modelId !== null && !isJobBusy(phase);

  return (
    <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
      <div className="flex flex-col gap-6">
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
        <DevicePicker value={device} onChange={(selected) => setDevice(selected.id)} requiresGpu={false} />
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
          {cpuConfirmPending && <CpuConfirmBanner onConfirm={handleGenerate} onCancel={handleCancelCpuConfirm} />}
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
