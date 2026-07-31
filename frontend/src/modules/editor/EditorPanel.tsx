import { useQuery } from "@tanstack/react-query";
import { Brush, Eraser, ImageUp, MousePointerClick, RotateCcw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { useGenerationCapabilities, useGenerationJob } from "../../hooks/useGenerationJob";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { DeviceInfoResponse, GenerationModelSummary, InitImageResponse } from "../../lib/apiTypes";
import { fetchEditorCapabilities, segmentEditorObject } from "../../services/editor";
import { uploadGenerationInitImage, type CreateGenerationJobParams } from "../../services/generation";
import {
  appendPoint,
  drawStrokes,
  fitGenerationSize,
  startStroke,
  toImagePoint,
  type BrushStroke,
} from "./maskCanvas";

// Prompts fijos del modo "Eliminar": el modelo rellena con entorno coherente.
// En inglés porque los checkpoints SD responden mejor a prompts en inglés.
const ERASE_PROMPT = "background, empty scene, seamless continuation of the surroundings, high quality";
const ERASE_NEGATIVE_PROMPT = "object, person, animal, text, watermark, logo";
const DEFAULT_STEPS = 30;
const DEFAULT_GUIDANCE = 7.0;

type Tool = "brush" | "eraser" | "tap";
type EditMode = "erase" | "replace";

interface HistoryEntry {
  kind: "stroke" | "segment";
  stroke?: BrushStroke;
  mask?: HTMLCanvasElement;
}

function pngBlobToAlphaCanvas(blob: Blob, width: number, height: number): Promise<HTMLCanvasElement> {
  // La máscara llega como PNG blanco-sobre-negro; para superponerla al canvas
  // hay que volverla blanco-sobre-transparente (alpha = luminancia).
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("canvas 2d context unavailable"));
        return;
      }
      ctx.drawImage(image, 0, 0, width, height);
      const data = ctx.getImageData(0, 0, width, height);
      for (let i = 0; i < data.data.length; i += 4) {
        data.data[i + 3] = data.data[i];
        data.data[i] = 255;
        data.data[i + 1] = 255;
        data.data[i + 2] = 255;
      }
      ctx.putImageData(data, 0, 0);
      resolve(canvas);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("could not decode mask"));
    };
    image.src = url;
  });
}

function replayHistory(canvas: HTMLCanvasElement, history: HistoryEntry[], live: BrushStroke | null): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const entry of history) {
    if (entry.kind === "segment" && entry.mask) {
      ctx.globalCompositeOperation = "source-over";
      ctx.drawImage(entry.mask, 0, 0);
    } else if (entry.stroke) {
      drawStrokes(ctx, [entry.stroke], "#ffffff");
    }
  }
  if (live) {
    drawStrokes(ctx, [live], "#ffffff");
  }
}

function maskHasContent(history: HistoryEntry[]): boolean {
  return history.some(
    (entry) => entry.kind === "segment" || (entry.stroke?.mode === "paint" && entry.stroke.points.length > 0),
  );
}

function exportMaskBlob(maskCanvas: HTMLCanvasElement): Promise<Blob> {
  // Contrato del backend: escala de grises, blanco=editar, NEGRO=conservar —
  // el canvas de trabajo es blanco-sobre-transparente, acá se aplasta a negro.
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = maskCanvas.width;
  exportCanvas.height = maskCanvas.height;
  const ctx = exportCanvas.getContext("2d");
  if (!ctx) {
    return Promise.reject(new Error("canvas 2d context unavailable"));
  }
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
  ctx.drawImage(maskCanvas, 0, 0);
  return new Promise((resolve, reject) => {
    exportCanvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("could not export the mask"));
      }
    }, "image/png");
  });
}

function InpaintModelSelect({
  models,
  value,
  onChange,
  label,
}: {
  models: GenerationModelSummary[];
  value: string | null;
  onChange: (id: string | null) => void;
  label: string;
}) {
  const usable = models.filter((model) => model.status === "installed" && model.supportsInpaint !== false);
  return (
    <label className="flex flex-col gap-1 text-sm text-text">
      {label}
      <select
        className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value || null)}
      >
        <option value="" disabled>
          —
        </option>
        {usable.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
    </label>
  );
}

export function EditorPanel() {
  const { t } = useTranslation();
  const capabilitiesQuery = useGenerationCapabilities();
  const editorCapabilities = useQuery({ queryKey: ["editorCapabilities"], queryFn: fetchEditorCapabilities });
  const { phase, job, errorMessage, submit, cancel, reset } = useGenerationJob();

  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [baseInfo, setBaseInfo] = useState<InitImageResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [tool, setTool] = useState<Tool>("brush");
  const [brushSize, setBrushSize] = useState(32);
  const [mode, setMode] = useState<EditMode>("erase");
  const [prompt, setPrompt] = useState("");
  const [modelId, setModelId] = useState<string | null>(null);
  const [device, setDevice] = useState<string | null>(null);
  // Vista avanzada: los defaults de Quitar quedan visibles y editables en vez
  // de vivir escondidos en constantes.
  const [erasePromptText, setErasePromptText] = useState(ERASE_PROMPT);
  const [eraseNegative, setEraseNegative] = useState(ERASE_NEGATIVE_PROMPT);
  const [replaceNegative, setReplaceNegative] = useState("");
  const [steps, setSteps] = useState(DEFAULT_STEPS);
  const [guidance, setGuidance] = useState(DEFAULT_GUIDANCE);
  const [seedText, setSeedText] = useState("");
  const [strength, setStrength] = useState(1);
  const [segmenting, setSegmenting] = useState(false);
  const [segmentError, setSegmentError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const liveStrokeRef = useRef<BrushStroke | null>(null);
  const uploadSequence = useRef(0);

  const models = capabilitiesQuery.data?.models ?? [];
  const tapAvailable = editorCapabilities.data?.tapSelect ?? false;
  const busy = phase === "uploading" || phase === "queued" || phase === "running";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      replayHistory(canvas, history, liveStrokeRef.current);
    }
  }, [history, baseInfo]);

  useEffect(() => () => {
    if (baseUrl) {
      URL.revokeObjectURL(baseUrl);
    }
  }, [baseUrl]);

  const handleBaseSelected = useCallback(
    async (file: File) => {
      const sequence = ++uploadSequence.current;
      setUploadError(null);
      setHistory([]);
      setSegmentError(null);
      reset();
      try {
        const response = await uploadGenerationInitImage(file);
        if (sequence !== uploadSequence.current) {
          return;
        }
        setBaseInfo(response);
        setBaseUrl((previous) => {
          if (previous) {
            URL.revokeObjectURL(previous);
          }
          return typeof URL.createObjectURL === "function" ? URL.createObjectURL(file) : null;
        });
      } catch (error) {
        if (sequence === uploadSequence.current) {
          setUploadError(error instanceof Error ? error.message : String(error));
        }
      }
    },
    [reset],
  );

  function imagePointFromEvent(event: React.PointerEvent | React.MouseEvent): { x: number; y: number } | null {
    const canvas = canvasRef.current;
    if (!canvas || !baseInfo) {
      return null;
    }
    const rect = canvas.getBoundingClientRect();
    return toImagePoint(event.clientX, event.clientY, rect, baseInfo.width, baseInfo.height);
  }

  function scaledBrushRadius(): number {
    const canvas = canvasRef.current;
    if (!canvas || !baseInfo) {
      return brushSize;
    }
    const rect = canvas.getBoundingClientRect();
    const displayScale = rect.width > 0 ? baseInfo.width / rect.width : 1;
    return (brushSize / 2) * displayScale;
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (tool === "tap" || !baseInfo || busy) {
      return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
      return;
    }
    event.currentTarget.setPointerCapture?.(event.pointerId);
    liveStrokeRef.current = startStroke(tool === "brush" ? "paint" : "erase", scaledBrushRadius(), point);
    const canvas = canvasRef.current;
    if (canvas) {
      replayHistory(canvas, history, liveStrokeRef.current);
    }
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!liveStrokeRef.current) {
      return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
      return;
    }
    liveStrokeRef.current = appendPoint(liveStrokeRef.current, point);
    const canvas = canvasRef.current;
    if (canvas) {
      replayHistory(canvas, history, liveStrokeRef.current);
    }
  }

  function handlePointerUp() {
    const stroke = liveStrokeRef.current;
    if (!stroke) {
      return;
    }
    liveStrokeRef.current = null;
    setHistory((previous) => [...previous, { kind: "stroke", stroke }]);
  }

  async function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (tool !== "tap" || !baseInfo || segmenting || busy) {
      return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
      return;
    }
    setSegmenting(true);
    setSegmentError(null);
    try {
      const blob = await segmentEditorObject(baseInfo.initImageToken, point.x, point.y);
      const mask = await pngBlobToAlphaCanvas(blob, baseInfo.width, baseInfo.height);
      setHistory((previous) => [...previous, { kind: "segment", mask }]);
    } catch (error) {
      setSegmentError(error instanceof Error ? error.message : String(error));
    } finally {
      setSegmenting(false);
    }
  }

  async function handleGenerate() {
    const canvas = canvasRef.current;
    if (!canvas || !baseInfo || !modelId) {
      return;
    }
    setUploadError(null);
    try {
      const maskBlob = await exportMaskBlob(canvas);
      const maskFile = new File([maskBlob], "mask.png", { type: "image/png" });
      const maskResponse = await uploadGenerationInitImage(maskFile);
      const size = fitGenerationSize(baseInfo.width, baseInfo.height);
      const seed = seedText.trim() === "" ? null : Number.parseInt(seedText, 10);
      const negative = mode === "erase" ? eraseNegative : replaceNegative;
      const params: CreateGenerationJobParams = {
        prompt: mode === "erase" ? erasePromptText : prompt,
        negativePrompt: negative.trim() === "" ? null : negative,
        modelId,
        steps,
        guidance,
        width: size.width,
        height: size.height,
        seed: seed !== null && Number.isFinite(seed) ? seed : null,
        device,
        autoUpscale: false,
        upscaleModelName: null,
        upscaleScale: null,
        upscaleModelId: null,
        initImageToken: baseInfo.initImageToken,
        maskImageToken: maskResponse.initImageToken,
        strength,
      };
      submit(params);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleUseResultAsBase() {
    if (!job?.downloadUrl) {
      return;
    }
    try {
      const response = await fetch(job.downloadUrl);
      const blob = await response.blob();
      await handleBaseSelected(new File([blob], "edited.png", { type: "image/png" }));
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : String(error));
    }
  }

  const canSubmit =
    Boolean(baseInfo) &&
    Boolean(modelId) &&
    maskHasContent(history) &&
    (mode === "erase" ? erasePromptText.trim().length > 0 : prompt.trim().length > 0) &&
    !busy &&
    !segmenting;

  const toolButton = (value: Tool, icon: React.ReactNode, label: string, disabled = false, title?: string) => (
    <button
      type="button"
      role="radio"
      aria-checked={tool === value}
      aria-label={label}
      title={title ?? label}
      disabled={disabled}
      onClick={() => setTool(value)}
      className={`flex items-center gap-1.5 rounded px-3 py-1.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        tool === value ? "bg-accent font-medium text-surface" : "text-text-dim hover:bg-surface hover:text-text"
      }`}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <div className="flex flex-col gap-4">
      {!baseInfo && (
        <div className="flex flex-col items-center gap-3 rounded border border-dashed border-border bg-surface p-10 text-center">
          <ImageUp aria-hidden="true" className="h-8 w-8 text-text-dim" strokeWidth={1.5} />
          <p className="text-sm text-text-dim">{t("editor.dropzone.hint")}</p>
          <label
            htmlFor="editor-base-image-input"
            className="cursor-pointer rounded bg-accent px-4 py-2 text-sm font-medium text-surface"
          >
            {t("editor.dropzone.label")}
          </label>
          <input
            id="editor-base-image-input"
            type="file"
            accept="image/*"
            className="sr-only"
            aria-label={t("editor.dropzone.inputLabel")}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void handleBaseSelected(file);
              }
              event.target.value = "";
            }}
          />
        </div>
      )}

      {baseInfo && (
        <>
          <div
            className="flex flex-wrap items-center gap-2 rounded border border-border bg-surface p-2"
            role="radiogroup"
            aria-label={t("editor.tools.label")}
          >
            {toolButton("brush", <Brush aria-hidden="true" className="h-4 w-4" />, t("editor.tool.brush"))}
            {toolButton("eraser", <Eraser aria-hidden="true" className="h-4 w-4" />, t("editor.tool.eraser"))}
            {toolButton(
              "tap",
              <MousePointerClick aria-hidden="true" className="h-4 w-4" />,
              t("editor.tool.tap"),
              !tapAvailable,
              tapAvailable ? undefined : t("editor.tool.tapUnavailable"),
            )}
            <label className="ml-2 flex items-center gap-2 text-xs text-text-dim">
              {t("editor.brushSize")}
              <input
                type="range"
                min={8}
                max={128}
                value={brushSize}
                onChange={(event) => setBrushSize(Number(event.target.value))}
              />
            </label>
            <div className="ml-auto flex items-center gap-1">
              <button
                type="button"
                aria-label={t("editor.undo")}
                title={t("editor.undo")}
                disabled={history.length === 0}
                onClick={() => setHistory((previous) => previous.slice(0, -1))}
                className="rounded p-2 text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
              >
                <RotateCcw aria-hidden="true" className="h-4 w-4" />
              </button>
              <button
                type="button"
                aria-label={t("editor.clear")}
                title={t("editor.clear")}
                disabled={history.length === 0}
                onClick={() => setHistory([])}
                className="rounded p-2 text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
              >
                <Trash2 aria-hidden="true" className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="relative w-fit max-w-full self-center overflow-hidden rounded border border-border">
            {baseUrl && <img src={baseUrl} alt={baseInfo.originalFilename} className="block max-w-full" />}
            <canvas
              ref={canvasRef}
              width={baseInfo.width}
              height={baseInfo.height}
              data-testid="editor-mask-canvas"
              className={`absolute inset-0 h-full w-full opacity-60 ${
                tool === "tap" ? "cursor-pointer" : "cursor-crosshair"
              } ${baseUrl ? "" : "relative bg-surface-2"}`}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
              onClick={(event) => void handleCanvasClick(event)}
            />
            {segmenting && (
              <p className="absolute bottom-2 left-2 rounded bg-surface px-2 py-1 text-xs text-text-dim">
                {t("editor.segmenting")}
              </p>
            )}
          </div>
          <p className="text-center text-xs text-text-faint">{t("editor.maskHint")}</p>
          {segmentError && <p className="text-center text-xs text-danger">{segmentError}</p>}

          <div
            className="flex gap-1 self-center rounded border border-border bg-surface-2 p-1"
            role="radiogroup"
            aria-label={t("editor.mode.label")}
          >
            {(["erase", "replace"] as const).map((value) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={mode === value}
                onClick={() => setMode(value)}
                className={`rounded px-4 py-1.5 text-sm transition-colors ${
                  mode === value ? "bg-accent font-medium text-surface" : "text-text-dim hover:text-text"
                }`}
              >
                {t(`editor.mode.${value}`)}
              </button>
            ))}
          </div>

          {mode === "replace" && (
            <label className="flex flex-col gap-1 text-sm text-text">
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

          <div className="grid grid-cols-2 gap-3 max-[700px]:grid-cols-1">
            <InpaintModelSelect models={models} value={modelId} onChange={setModelId} label={t("editor.model.label")} />
            <DevicePicker
              value={device}
              onChange={(selected: DeviceInfoResponse) => setDevice(selected.id)}
              requiresGpu={false}
              allowAuto={false}
            />
          </div>

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
                    onChange={(event) =>
                      setSteps(Math.min(100, Math.max(1, Number(event.target.value) || 1)))
                    }
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

          {uploadError && <p className="text-sm text-danger">{uploadError}</p>}

          <div className="flex items-center gap-3">
            <button
              type="button"
              disabled={!canSubmit}
              onClick={() => void handleGenerate()}
              className="rounded bg-accent px-5 py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-40"
            >
              {mode === "erase" ? t("editor.submit.erase") : t("editor.submit.replace")}
            </button>
            <label
              htmlFor="editor-base-image-change"
              className="cursor-pointer text-sm text-text-dim underline hover:text-text"
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
                if (file) {
                  void handleBaseSelected(file);
                }
                event.target.value = "";
              }}
            />
          </div>

          <JobCard
            phase={phase}
            job={job}
            fileName={baseInfo.originalFilename}
            errorMessage={errorMessage}
            onCancel={cancel}
          />
          {job?.status === "completed" && job.downloadUrl && (
            <button
              type="button"
              onClick={() => void handleUseResultAsBase()}
              className="self-start rounded border border-border px-4 py-2 text-sm text-text hover:bg-surface-2"
            >
              {t("editor.useAsBase")}
            </button>
          )}
        </>
      )}
    </div>
  );
}
