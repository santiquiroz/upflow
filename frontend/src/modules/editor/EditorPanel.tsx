import { useQuery } from "@tanstack/react-query";
import {
  BoxSelect,
  Brush,
  Eraser,
  Hand,
  ImageUp,
  Lasso,
  Maximize,
  MousePointerClick,
  RotateCcw,
  Trash2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
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
  isCommittableStroke,
  rectanglePoints,
  startStroke,
  strokeAddsEditableArea,
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

type Tool = "brush" | "eraser" | "lasso" | "rect" | "tap" | "pan";
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
    (entry) => entry.kind === "segment" || strokeAddsEditableArea(entry.stroke),
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
  mode,
}: {
  models: GenerationModelSummary[];
  value: string | null;
  onChange: (id: string | null) => void;
  label: string;
  mode: EditMode;
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
  const [replaceNegative, setReplaceNegative] = useState(REPLACE_NEGATIVE_PROMPT);
  const [steps, setSteps] = useState(DEFAULT_STEPS);
  const [guidance, setGuidance] = useState(DEFAULT_GUIDANCE);
  const [seedText, setSeedText] = useState("");
  const [strength, setStrength] = useState(1);
  const [segmenting, setSegmenting] = useState(false);
  const [segmentError, setSegmentError] = useState<string | null>(null);

  const [viewport, setViewport] = useState<Viewport>(IDENTITY_VIEWPORT);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
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
  const tapAvailable = editorCapabilities.data?.tapSelect ?? false;
  const busy = phase === "uploading" || phase === "queued" || phase === "running";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      replayHistory(canvas, history, liveStrokeRef.current);
    }
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
    if (tool === "tap" || busy) {
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
    const canvas = canvasRef.current;
    if (canvas) {
      replayHistory(canvas, history, liveStrokeRef.current);
    }
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
    const canvas = canvasRef.current;
    if (canvas) {
      replayHistory(canvas, history, liveStrokeRef.current);
    }
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
      const canvas = canvasRef.current;
      if (canvas) {
        replayHistory(canvas, history, null);
      }
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
    (isEraseOnlyModel || (mode === "erase" ? erasePromptText.trim().length > 0 : prompt.trim().length > 0)) &&
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
            {toolButton("lasso", <Lasso aria-hidden="true" className="h-4 w-4" />, t("editor.tool.lasso"))}
            {toolButton("rect", <BoxSelect aria-hidden="true" className="h-4 w-4" />, t("editor.tool.rect"))}
            {toolButton(
              "tap",
              <MousePointerClick aria-hidden="true" className="h-4 w-4" />,
              t("editor.tool.tap"),
              !tapAvailable,
              tapAvailable ? undefined : t("editor.tool.tapUnavailable"),
            )}
            {toolButton("pan", <Hand aria-hidden="true" className="h-4 w-4" />, t("editor.tool.pan"))}
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
                aria-label={t("editor.zoom.out")}
                title={t("editor.zoom.out")}
                onClick={() => handleZoomStep(-1)}
                className="rounded p-2 text-text-dim hover:bg-surface-2 hover:text-text"
              >
                <ZoomOut aria-hidden="true" className="h-4 w-4" />
              </button>
              <span className="min-w-[3.5rem] text-center font-mono-tabular text-xs text-text-dim">
                {zoomPercent(viewport)}%
              </span>
              <button
                type="button"
                aria-label={t("editor.zoom.in")}
                title={t("editor.zoom.in")}
                onClick={() => handleZoomStep(1)}
                className="rounded p-2 text-text-dim hover:bg-surface-2 hover:text-text"
              >
                <ZoomIn aria-hidden="true" className="h-4 w-4" />
              </button>
              <button
                type="button"
                aria-label={t("editor.zoom.reset")}
                title={t("editor.zoom.reset")}
                disabled={!isZoomed(viewport)}
                onClick={() => setViewport(IDENTITY_VIEWPORT)}
                className="rounded p-2 text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
              >
                <Maximize aria-hidden="true" className="h-4 w-4" />
              </button>
              <span aria-hidden="true" className="mx-1 h-5 w-px bg-border" />
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
              <span aria-hidden="true" className="mx-1 h-5 w-px bg-border" />
              <button
                type="button"
                aria-label={t("editor.newImage")}
                title={t("editor.newImage")}
                disabled={busy}
                onClick={handleCloseImage}
                className="rounded p-2 text-text-dim hover:bg-surface-2 hover:text-text disabled:opacity-40"
              >
                <X aria-hidden="true" className="h-4 w-4" />
              </button>
            </div>
          </div>

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
                    : tool === "tap"
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
            </div>
            {segmenting && (
              <p className="absolute bottom-2 left-2 rounded bg-surface px-2 py-1 text-xs text-text-dim">
                {t("editor.segmenting")}
              </p>
            )}
          </div>
          <p className="text-center text-xs text-text-faint">{t("editor.maskHint")}</p>
          <p className="text-center text-xs text-text-faint">{t("editor.zoomHint")}</p>
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
            <InpaintModelSelect
              models={models}
              value={modelId}
              onChange={setModelId}
              label={t("editor.model.label")}
              mode={mode}
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
            <p className="text-xs text-text-faint">{t("editor.eraser.hint")}</p>
          )}

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
