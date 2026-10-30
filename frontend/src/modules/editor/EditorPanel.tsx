import { useQuery } from "@tanstack/react-query";
import { ImageUp } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { useGenerationCapabilities, useGenerationJob } from "../../hooks/useGenerationJob";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { DeviceInfoResponse, GenerationModelSummary, InitImageResponse } from "../../lib/apiTypes";
import { fetchEditorCapabilities, segmentEditorObject } from "../../services/editor";
import { uploadGenerationInitImage, type CreateGenerationJobParams } from "../../services/generation";
import { buildEditorJobParams, shouldUseAutoErase } from "./eraseJobParams";
import { EditorSidePanel } from "./EditorSidePanel";
import { BRUSH_SIZE_MAX, BRUSH_SIZE_MIN, EditorToolRail, type Tool } from "./EditorToolRail";
import {
  canvasHasPixels, createMaskCanvas, exportMaskBlob, exportProtectMaskBlob, hasProtection, maskHasContent,
  pngBlobToAlphaCanvas, renderMaskOverlays, replayMasks, type HistoryEntry,
} from "./editorMaskCanvas";
import { InsertObjectPanel } from "./InsertObjectPanel";
import { useEditorHandoff } from "./useEditorHandoff";
import {
  appendPoint,
  fitGenerationSize,
  isCommittableStroke,
  rectanglePoints,
  startStroke,
  toImagePoint,
  type BrushStroke,
} from "./maskCanvas";
import {
  IDENTITY_VIEWPORT,
  clampPan,
  isZoomed,
  panBy,
  transformStyle,
  zoomAtPoint,
  zoomByStep,
  zoomPercent,
  type Viewport,
} from "./viewport";
import { useEditorShortcuts } from "./useEditorShortcuts";

// Prompts fijos del modo "Eliminar": el modelo rellena con entorno coherente.
// En inglés porque los checkpoints SD responden mejor a prompts en inglés.
const ERASE_PROMPT = "background, empty scene, seamless continuation of the surroundings, high quality";
const ERASE_NEGATIVE_PROMPT = "object, person, animal, text, watermark, logo";
// Negative anatómico por defecto del modo Reemplazar: la causa más frecuente
// de cuerpos rotos en inpaint es no vetar explícitamente las deformaciones.
const REPLACE_NEGATIVE_PROMPT =
  "deformed, distorted anatomy, extra limbs, extra fingers, missing fingers, disconnected limbs, blurry, watermark, text";
const DEFAULT_STEPS = 30;
const DEFAULT_GUIDANCE = 7.0;
const BRUSH_SIZE_STEP = 4;
type EditMode = "erase" | "replace";

function InpaintModelSelect({
  models,
  value,
  onChange,
  label,
  mode,
  hint,
}: {
  models: GenerationModelSummary[];
  value: string | null;
  onChange: (id: string | null) => void;
  label: string;
  mode: EditMode;
  hint?: string;
}) {
  // Los motores de solo-borrado rellenan continuando el entorno: no saben poner
  // algo concreto, así que en Reemplazar no se ofrecen.
  const usable = models.filter(
    (model) =>
      model.status === "installed" &&
      model.supportsInpaint !== false &&
      // Turbo: 512px nativo y sin negative prompt — no sirve para inpaint.
      model.speed !== "turbo" &&
      (mode === "erase" || !model.eraseOnly),
  );
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
      {hint && <span className="text-xs text-text-faint">{hint}</span>}
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
  const [baseUploading, setBaseUploading] = useState(false);
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
  const [replaceNegative, setReplaceNegative] = useState(REPLACE_NEGATIVE_PROMPT);
  const [steps, setSteps] = useState(DEFAULT_STEPS);
  const [guidance, setGuidance] = useState(DEFAULT_GUIDANCE);
  const [seedText, setSeedText] = useState("");
  const [strength, setStrength] = useState(1);
  const [segmenting, setSegmenting] = useState(false);
  const [preparingJob, setPreparingJob] = useState(false);
  const [segmentError, setSegmentError] = useState<string | null>(null);

  const [viewport, setViewport] = useState<Viewport>(IDENTITY_VIEWPORT);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const protectionOverlayRef = useRef<HTMLCanvasElement | null>(null);
  const editMaskRef = useRef<HTMLCanvasElement | null>(null);
  const protectMaskRef = useRef<HTMLCanvasElement | null>(null);
  const segmentSequence = useRef(0);
  const segmentPendingRef = useRef(false);
  const preparingJobRef = useRef(false);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const liveStrokeRef = useRef<BrushStroke | null>(null);
  const panOriginRef = useRef<{ x: number; y: number } | null>(null);
  const rectOriginRef = useRef<{ x: number; y: number } | null>(null);
  const uploadSequence = useRef(0);

  function stageSize(): { width: number; height: number } {
    // clientWidth/Height y no getBoundingClientRect: el rect incluye el borde de
    // 1px del recuadro y el contenido vive en la caja interior, así que medir el
    // borde afloja los límites de arrastre 2*(zoom-1) px y deja una franja vacía.
    const stage = stageRef.current;
    return { width: stage?.clientWidth ?? 0, height: stage?.clientHeight ?? 0 };
  }

  const models = capabilitiesQuery.data?.models ?? [];
  const selectedModel = models.find((model) => model.id === modelId) ?? null;
  // Un motor de solo-borrado no sabe poner algo concreto: si el usuario pasa a
  // Reemplazar hay que soltarlo, o el job saldria con un modelo que no aplica.
  const isEraseOnlyModel = selectedModel?.eraseOnly === true;
  const autoErase = shouldUseAutoErase(mode, modelId, models);
  const tapAvailable = editorCapabilities.data?.tapSelect ?? false;
  const busy = baseUploading || preparingJob || phase === "uploading" || phase === "queued" || phase === "running";
  const handleUndo = useCallback(() => {
    setHistory((previous) => previous.slice(0, -1));
  }, []);
  const handleBrushSmaller = useCallback(() => {
    setBrushSize((current) => Math.max(BRUSH_SIZE_MIN, current - BRUSH_SIZE_STEP));
  }, []);
  const handleBrushBigger = useCallback(() => {
    setBrushSize((current) => Math.min(BRUSH_SIZE_MAX, current + BRUSH_SIZE_STEP));
  }, []);

  useEditorShortcuts({
    onUndo: handleUndo,
    onBrushSmaller: handleBrushSmaller,
    onBrushBigger: handleBrushBigger,
    enabled: Boolean(baseInfo),
  });

  function redrawMasks() {
    const canvas = canvasRef.current;
    const protectionOverlay = protectionOverlayRef.current;
    if (!canvas || !protectionOverlay || !baseInfo) return;
    const { width, height } = baseInfo;
    if (editMaskRef.current?.width !== width || editMaskRef.current?.height !== height) {
      editMaskRef.current = createMaskCanvas(width, height);
      protectMaskRef.current = createMaskCanvas(width, height);
    }
    const edit = editMaskRef.current!;
    const protect = protectMaskRef.current!;
    replayMasks(edit, protect, history, liveStrokeRef.current);
    renderMaskOverlays(canvas, protectionOverlay, edit, protect);
  }

  useEffect(() => {
    redrawMasks();
  }, [history, baseInfo]);

  useEffect(() => {
    if (mode === "replace" && isEraseOnlyModel) {
      setModelId(null);
    }
  }, [mode, isEraseOnlyModel]);

  useEffect(() => () => {
    if (baseUrl) {
      URL.revokeObjectURL(baseUrl);
    }
  }, [baseUrl]);

  // La rueda se escucha a mano con passive:false — React registra onWheel como
  // pasivo y preventDefault ahí no frena el scroll de la página.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }
    function handleWheel(event: WheelEvent) {
      event.preventDefault();
      const rect = stage!.getBoundingClientRect();
      // El origen del transform es la caja interior: descontar el borde o el
      // punto anclado se corre un píxel por paso y el error se acumula.
      const focus = {
        x: event.clientX - rect.left - stage!.clientLeft,
        y: event.clientY - rect.top - stage!.clientTop,
      };
      const container = { width: stage!.clientWidth, height: stage!.clientHeight };
      setViewport((current) =>
        zoomAtPoint(current, current.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), focus, container),
      );
    }
    stage.addEventListener("wheel", handleWheel, { passive: false });
    return () => stage.removeEventListener("wheel", handleWheel);
  }, [baseInfo]);

  // Al achicar la ventana el lienzo encoge y un desplazamiento que era legal
  // deja la foto entera fuera de la vista: el usuario ve un recuadro vacío sin
  // ninguna pista. Se vuelve a acotar cada vez que cambia el tamaño.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      setViewport((current) =>
        clampPan(current, { width: stage.clientWidth, height: stage.clientHeight }),
      );
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, [baseInfo]);

  const handleBaseSelected = useCallback(
    async (file: File) => {
      const sequence = ++uploadSequence.current;
      setBaseUploading(true);
      segmentSequence.current += 1;
      segmentPendingRef.current = false;
      setSegmenting(false);
      liveStrokeRef.current = null;
      rectOriginRef.current = null;
      setUploadError(null);
      setHistory([]);
      setSegmentError(null);
      setViewport(IDENTITY_VIEWPORT);
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
      } finally {
        if (sequence === uploadSequence.current) setBaseUploading(false);
      }
    },
    [reset],
  );

  useEditorHandoff(handleBaseSelected, setUploadError);

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
    if (!baseInfo) {
      return;
    }
    // El botón del medio arrastra la vista con cualquier herramienta activa.
    if (tool === "pan" || event.button === 1) {
      event.preventDefault();
      event.currentTarget.setPointerCapture?.(event.pointerId);
      panOriginRef.current = { x: event.clientX, y: event.clientY };
      return;
    }
    if (tool === "tap" || tool === "protect" || busy) {
      return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
      return;
    }
    event.currentTarget.setPointerCapture?.(event.pointerId);
    if (tool === "rect") {
      // El rectángulo es un lazo de 4 puntos que se recalcula al arrastrar:
      // reusa el dibujo/relleno del lazo sin un modo de trazo nuevo.
      rectOriginRef.current = point;
      liveStrokeRef.current = { mode: "lasso", radius: 0, points: [point] };
    } else if (tool === "lasso") {
      liveStrokeRef.current = startStroke("lasso", 0, point);
    } else {
      liveStrokeRef.current = startStroke(tool === "brush" ? "paint" : "erase", scaledBrushRadius(), point);
    }
    redrawMasks();
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const panOrigin = panOriginRef.current;
    if (panOrigin) {
      const dx = event.clientX - panOrigin.x;
      const dy = event.clientY - panOrigin.y;
      panOriginRef.current = { x: event.clientX, y: event.clientY };
      const container = stageSize();
      setViewport((current) => panBy(current, dx, dy, container));
      return;
    }
    if (!liveStrokeRef.current) {
      return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
      return;
    }
    const rectOrigin = rectOriginRef.current;
    if (rectOrigin) {
      liveStrokeRef.current = { ...liveStrokeRef.current, points: rectanglePoints(rectOrigin, point) };
    } else {
      liveStrokeRef.current = appendPoint(liveStrokeRef.current, point);
    }
    redrawMasks();
  }

  function handlePointerUp() {
    panOriginRef.current = null;
    rectOriginRef.current = null;
    const stroke = liveStrokeRef.current;
    if (!stroke) {
      return;
    }
    liveStrokeRef.current = null;
    if (!isCommittableStroke(stroke)) {
      // Un lazo de 1-2 puntos no forma polígono: se descarta y se borra su preview.
      redrawMasks();
      return;
    }
    setHistory((previous) => [...previous, { kind: "stroke", stroke }]);
  }

  function handleZoomStep(direction: number) {
    const container = stageSize();
    setViewport((current) => zoomByStep(current, direction, container));
  }

  function handleCloseImage() {
    // Cerrar la imagen actual vuelve al dropzone: antes la única forma de
    // empezar con otra foto era recargar la página o subir encima.
    uploadSequence.current += 1;
    setBaseUploading(false);
    segmentSequence.current += 1;
    segmentPendingRef.current = false;
    setSegmenting(false);
    setBaseInfo(null);
    setBaseUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return null;
    });
    setHistory([]);
    setSegmentError(null);
    setUploadError(null);
    setViewport(IDENTITY_VIEWPORT);
    liveStrokeRef.current = null;
    reset();
  }

  async function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if ((tool !== "tap" && tool !== "protect") || !tapAvailable || !baseInfo || segmentPendingRef.current || busy) {
      return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
      return;
    }
    const sequence = ++segmentSequence.current;
    const kind = tool === "protect" ? "protect" : "segment";
    segmentPendingRef.current = true;
    setSegmenting(true);
    setSegmentError(null);
    try {
      const blob = await segmentEditorObject(baseInfo.initImageToken, point.x, point.y);
      const mask = await pngBlobToAlphaCanvas(blob, baseInfo.width, baseInfo.height);
      if (sequence !== segmentSequence.current) return;
      setHistory((previous) => [...previous, { kind, mask, point }]);
    } catch (error) {
      if (sequence === segmentSequence.current) {
        setSegmentError(error instanceof Error ? error.message : String(error));
      }
    } finally {
      if (sequence === segmentSequence.current) {
        segmentPendingRef.current = false;
        setSegmenting(false);
      }
    }
  }

  async function handleGenerate() {
    const canvas = editMaskRef.current;
    if (!canvas || !baseInfo || !modelId || busy || preparingJobRef.current || segmentPendingRef.current) {
      return;
    }
    const sequence = uploadSequence.current;
    preparingJobRef.current = true;
    setPreparingJob(true);
    setUploadError(null);
    try {
      const protection = protectMaskRef.current;
      const protectBlobPromise = protection && canvasHasPixels(protection)
        ? exportProtectMaskBlob(protection)
        : Promise.resolve(null);
      const [maskBlob, protectBlob] = await Promise.all([exportMaskBlob(canvas), protectBlobPromise]);
      const maskFile = new File([maskBlob], "mask.png", { type: "image/png" });
      const maskResponse = await uploadGenerationInitImage(maskFile);
      if (sequence !== uploadSequence.current) return;
      let protectMaskToken: string | undefined;
      if (protectBlob) {
        const protectFile = new File([protectBlob], "protect-mask.png", { type: "image/png" });
        protectMaskToken = (await uploadGenerationInitImage(protectFile)).initImageToken;
      }
      if (sequence !== uploadSequence.current) return;
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
        ...(protectMaskToken ? { protectMaskToken } : {}),
      };
      submit(buildEditorJobParams(params, mode, models));
    } catch (error) {
      if (sequence === uploadSequence.current) {
        setUploadError(error instanceof Error ? error.message : String(error));
      }
    } finally {
      preparingJobRef.current = false;
      setPreparingJob(false);
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
    (isEraseOnlyModel || (mode === "erase" ? erasePromptText.trim().length > 0 : prompt.trim().length > 0)) &&
    !busy &&
    !segmenting;
  const panelView = job?.status === "completed" ? "review" : "action";

  return (
    <div className={baseInfo
      ? "grid min-h-0 grid-cols-[148px_minmax(0,1fr)_340px] items-start gap-3"
      : "flex flex-col gap-4"}
    >
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
          <EditorToolRail
            tool={tool}
            onToolChange={setTool}
            disabledTools={{
              tap: tapAvailable ? undefined : t("editor.tool.tapUnavailable"),
              protect: tapAvailable ? undefined : t("editor.tool.tapUnavailable"),
            }}
            toolTitles={{ protect: t("editor.protect.hint") }}
            brushSize={brushSize}
            onBrushSizeChange={setBrushSize}
            zoomLabel={`${zoomPercent(viewport)}%`}
            onZoomIn={() => handleZoomStep(1)}
            onZoomOut={() => handleZoomStep(-1)}
            onZoomReset={() => setViewport(IDENTITY_VIEWPORT)}
            canResetZoom={isZoomed(viewport)}
            onUndo={handleUndo}
            canUndo={history.length > 0}
            onClear={() => setHistory((previous) =>
              previous.filter((entry) => entry.kind === "protect" || entry.kind === "clearProtection")
            )}
            canClear={maskHasContent(history)}
            onClearProtection={() =>
              setHistory((previous) => [...previous, { kind: "clearProtection" }])
            }
            canClearProtection={hasProtection(history)}
            onNewImage={handleCloseImage}
            busy={busy}
          />

          <main className="flex min-w-0 flex-col items-center gap-3 overflow-auto">
            <div
              ref={stageRef}
              data-testid="editor-stage"
              className="relative w-fit max-w-full touch-none self-center overflow-hidden rounded border border-border"
            >
              {/* El zoom es un transform CSS sobre imagen+canvas juntos: el
                  rectángulo que devuelve el canvas ya viene transformado, así que
                  el mapeo click->píxel y el radio del pincel siguen valiendo. */}
              <div style={{ transform: transformStyle(viewport), transformOrigin: "0 0" }}>
                {baseUrl && <img src={baseUrl} alt={baseInfo.originalFilename} className="block max-w-full" />}
                <canvas
                  ref={canvasRef}
                  width={baseInfo.width}
                  height={baseInfo.height}
                  data-testid="editor-mask-canvas"
                  className={`absolute inset-0 h-full w-full opacity-60 ${
                    tool === "pan"
                      ? "cursor-grab active:cursor-grabbing"
                      : tool === "tap" || tool === "protect"
                        ? "cursor-pointer"
                        : "cursor-crosshair"
                  } ${baseUrl ? "" : "relative bg-surface-2"}`}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  onPointerLeave={handlePointerUp}
                  onPointerCancel={handlePointerUp}
                  onClick={(event) => void handleCanvasClick(event)}
                />
                <canvas
                  ref={protectionOverlayRef}
                  width={baseInfo.width}
                  height={baseInfo.height}
                  data-testid="editor-protect-canvas"
                  className="pointer-events-none absolute inset-0 h-full w-full opacity-50"
                />
              </div>
              {segmenting && (
                <p className="absolute bottom-2 left-2 rounded bg-surface px-2 py-1 text-xs text-text-dim">
                  {t("editor.segmenting")}
                </p>
              )}
            </div>
            <p className="text-center text-xs text-text-faint">{t("editor.maskHint")}</p>
            <p className="text-center text-xs text-text-faint">{t("editor.protect.hint")}</p>
            <p className="text-center text-xs text-text-faint">{t("editor.zoomHint")}</p>
            {segmentError && <p className="text-center text-xs text-danger">{segmentError}</p>}
          </main>

          <EditorSidePanel view={panelView}>
            {uploadError && <p className="mb-3 text-sm text-danger">{uploadError}</p>}
            {panelView === "review" && (
              <div className="mb-3">
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
                    className="mt-3 min-h-9 rounded border border-border px-4 py-2 text-sm text-text hover:bg-surface-2"
                  >
                    {t("editor.useAsBase")}
                  </button>
                )}
              </div>
            )}
            <>
                <div
                  className="flex gap-1 rounded border border-border bg-bg p-1"
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
                      className={`min-h-9 flex-1 rounded px-3 py-1.5 text-sm transition-colors ${
                        mode === value ? "bg-surface-2 font-medium text-text" : "text-text-dim hover:text-text"
                      }`}
                    >
                      {t(`editor.mode.${value}`)}
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
                )}
                {isEraseOnlyModel && (
                  <p className="mt-3 text-xs text-text-faint">{t("editor.eraser.hint")}</p>
                )}

                <div className="mt-3">
                  <InsertObjectPanel targetInfo={baseInfo} />
                </div>

                <button
                  type="button"
                  disabled={!canSubmit}
                  onClick={() => void handleGenerate()}
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
                    if (file) {
                      void handleBaseSelected(file);
                    }
                    event.target.value = "";
                  }}
                />
            </>
            {panelView === "action" && (
                <JobCard
                  phase={phase}
                  job={job}
                  fileName={baseInfo.originalFilename}
                  errorMessage={errorMessage}
                  onCancel={cancel}
                />
            )}
          </EditorSidePanel>
        </>
      )}
    </div>
  );
}
