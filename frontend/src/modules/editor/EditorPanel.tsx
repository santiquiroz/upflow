import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useGenerationCapabilities, useGenerationJob } from "../../hooks/useGenerationJob";
import { useTranslation } from "../../i18n/LocaleProvider";
import { segmentEditorObject } from "../../services/editor";
import {
  canReplace,
  defaultEraseModel,
  effectiveMode,
  hasEraserAuto,
  hasInstalledLama,
  modelForMode,
  shouldUseAutoErase,
} from "./eraseJobParams";
import { EditorDropzone } from "./EditorDropzone";
import { EditorSideContent } from "./EditorSideContent";
import { EditorSidePanel } from "./EditorSidePanel";
import { EditorStageStatus } from "./EditorStageStatus";
import { BRUSH_SIZE_MAX, BRUSH_SIZE_MIN, EditorToolRail, type Tool } from "./EditorToolRail";
import { PeopleOverlay } from "./PeopleOverlay";
import {
  composeCommittedMasks, hasProtection, maskHasContent, pngBlobToAlphaCanvas, type HistoryEntry,
} from "./editorMaskCanvas";
import { localizedErrorMessage } from "./editorErrors";
import { toolLocks } from "./editorToolLocks";
import { ReviewOverlay } from "./ReviewOverlay";
import { useEditorSubmit } from "./useEditorSubmit";
import { useReviewSession } from "./useReviewSession";
import { useZoneRedo } from "./useZoneRedo";
import { lastPeopleBatch } from "./silhouettes";
import { useSilhouetteBatch } from "./useSilhouetteBatch";
import type { EditMode } from "./InpaintModelSelect";
import { useEditorHandoff } from "./useEditorHandoff";
import {
  appendPoint,
  isCommittableStroke,
  rectanglePoints,
  startStroke,
  toImagePoint,
  type BrushStroke,
} from "./maskCanvas";
import { IDENTITY_VIEWPORT, isZoomed, panBy, transformStyle, zoomByStep, zoomPercent, type Viewport } from "./viewport";
import { useStageViewport } from "./useStageViewport";
import { useEditorShortcuts } from "./useEditorShortcuts";
import { usePeopleSelection } from "./usePeopleSelection";
import { useBaseVersions, useObjectUrl } from "./useBaseVersions";
import { useEditorUndo } from "./useEditorUndo";
import { useEdgeAdjust } from "./useEdgeAdjust";
import { editsBase, useHealTool } from "./useHealTool";
import { isRetouchTool, useRetouchTool } from "./useRetouchTool";
import { useMaskOverlay } from "./useMaskOverlay";
import { applyRetouch, type RetouchKind, type RetouchOffset } from "./retouchApply";
import { RetouchCrosshair } from "./RetouchCrosshair";
import { useEditorCapabilities } from "./useEditorCapabilities";

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

export function EditorPanel() {
  const { t } = useTranslation();
  const capabilitiesQuery = useGenerationCapabilities();
  const editorCapabilities = useEditorCapabilities();
  const { phase, job, errorMessage, submit, cancel, reset } = useGenerationJob();

  const baseVersions = useBaseVersions();
  const baseInfo = baseVersions.current;
  const baseUrl = useObjectUrl(baseVersions.currentBlob);
  const originalUrl = useObjectUrl(baseVersions.original?.blob ?? null);
  const retouch = useRetouchTool(handleRetouchCommit);
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
  const [segmentError, setSegmentError] = useState<string | null>(null);

  const [viewport, setViewport] = useState<Viewport>(IDENTITY_VIEWPORT);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const protectionOverlayRef = useRef<HTMLCanvasElement | null>(null);
  const editMaskRef = useRef<HTMLCanvasElement | null>(null);
  const protectMaskRef = useRef<HTMLCanvasElement | null>(null);
  const segmentSequence = useRef(0);
  const segmentPendingRef = useRef(false);
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
  const capabilitiesLoaded = Boolean(capabilitiesQuery.data);
  const peopleEnabled = editorCapabilities.data?.people === true;
  // La detección y las siluetas siempre usan la foto original: las personas no se mueven al retocar.
  const originalInfo = baseVersions.original?.info ?? null;
  const peopleSelection = usePeopleSelection(originalInfo, peopleEnabled);
  const silhouetteBatch = useSilhouetteBatch({
    baseInfo: originalInfo,
    selection: peopleSelection,
    sequenceRef: uploadSequence,
    onDone: handlePeopleBatch,
  });
  const peopleBatch = lastPeopleBatch(history);
  const review = useReviewSession(job);
  const zoneRedo = useZoneRedo({
    session: review.session,
    baseInfo,
    models,
    onDone: review.updateZone,
    defaults: { prompt: erasePromptText, negativePrompt: eraseNegative, steps, guidance, device },
  });
  const resultUrl = review.session?.resultUrl ?? null;
  const shownJob = job && resultUrl ? { ...job, downloadUrl: resultUrl } : job;
  useEffect(() => {
    if (!capabilitiesQuery.data) return;
    setModelId((current) => current ?? defaultEraseModel(models));
  }, [capabilitiesQuery.data, models]);
  useEffect(() => {
    const nextMode = effectiveMode(mode, models, capabilitiesLoaded);
    if (nextMode === mode) return;
    setMode(nextMode);
    setModelId((current) => modelForMode(nextMode, current, models));
  }, [capabilitiesLoaded, mode, models]);
  function handleModeChange(nextMode: EditMode): void {
    setMode(nextMode);
    setModelId((current) => modelForMode(nextMode, current, models));
  }
  const selectedModel = models.find((model) => model.id === modelId) ?? null;
  // Un motor de solo-borrado no sabe poner algo concreto: si el usuario pasa a
  // Reemplazar hay que soltarlo, o el job saldria con un modelo que no aplica.
  const isEraseOnlyModel = selectedModel?.eraseOnly === true;
  const autoErase = shouldUseAutoErase(mode, modelId, models);
  const { preview: edgePreview, ...edgeControls } = useEdgeAdjust({ mode, modelId, models, photo: baseVersions.original });
  const tapAvailable = editorCapabilities.data?.tapSelect ?? false;
  const heal = useHealTool({
    baseInfo, sequenceRef: uploadSequence, onError: setUploadError,
    onAdopt: (info, blob) => recordBase(baseVersions.adopt(info, blob)),
  });
  const { undoLog, appendHistory, dropHistoryEntry, recordBase, handleUndo, clearSelection } = useEditorUndo({
    history, setHistory, undoBase: baseVersions.undo, applying: retouch.applying || heal.healing,
  });
  const jobBusy = baseUploading || phase === "uploading" || phase === "queued" || phase === "running"
    || retouch.applying || heal.healing;
  const editorSubmit = useEditorSubmit({
    baseInfo, models, editMaskRef, protectMaskRef, uploadSequence, segmentPendingRef, review, submit, blocked: jobBusy,
    form: {
      mode, modelId, prompt, erasePromptText, eraseNegative, replaceNegative, steps, guidance, seedText, strength, device,
      edge: edgeControls.edge,
    },
    onError: (message) => setUploadError(message === null ? null : localizedErrorMessage(message, t)),
  });
  const busy = jobBusy || editorSubmit.preparingJob;
  const { last: lastUndo, clear: clearUndo } = undoLog;
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

  // El historial se compone en un lienzo completo solo cuando cambia; pintar un trazo no lo recompone.
  const committed = useMemo(
    () => (baseInfo ? composeCommittedMasks(history, baseInfo.width, baseInfo.height) : null),
    [history, baseInfo?.width, baseInfo?.height],
  );
  useEffect(() => {
    editMaskRef.current = committed?.edit ?? null;
    protectMaskRef.current = committed?.protect ?? null;
  }, [committed]);
  const redrawOverlay = useMaskOverlay({
    masks: committed,
    photoWidth: baseInfo?.width ?? 0,
    photoHeight: baseInfo?.height ?? 0,
    edge: edgePreview,
    redRef: canvasRef,
    greenRef: protectionOverlayRef,
    liveRef: liveStrokeRef,
  });

  useEffect(() => {
    redrawOverlay();
  }, [committed, baseInfo, edgePreview]);

  useEffect(() => {
    if (mode === "replace" && isEraseOnlyModel) {
      setModelId(null);
    }
  }, [mode, isEraseOnlyModel]);

  useStageViewport(stageRef, setViewport, baseInfo);

  // Solo una foto nueva (no un retoque) abre otra sesión: por eso solo aquí se vacía el historial.
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
      clearUndo();
      setSegmentError(null);
      setViewport(IDENTITY_VIEWPORT);
      retouch.reset();
      reset();
      review.clear();
      try {
        await baseVersions.reset(file);
      } catch (error) {
        if (sequence === uploadSequence.current) {
          setUploadError(localizedErrorMessage(error, t));
        }
      } finally {
        if (sequence === uploadSequence.current) setBaseUploading(false);
      }
    },
    [baseVersions.reset, clearUndo, reset, retouch.reset, review.clear, t],
  );

  // Un retoque sube una versión nueva de la base: no toca el historial de máscaras.
  async function pushBase(blob: Blob, sequence = uploadSequence.current) {
    const evicted = await baseVersions.push(blob);
    if (sequence !== uploadSequence.current) return;
    recordBase(evicted);
  }

  async function handleRetouchCommit(kind: RetouchKind, stroke: BrushStroke, offset: RetouchOffset) {
    const { original, currentBlob } = baseVersions;
    if (!original || !currentBlob) return;
    const sequence = uploadSequence.current;
    try {
      const blob = await applyRetouch(kind, stroke, currentBlob, original.blob, offset);
      await pushBase(blob, sequence);
    } catch (error) {
      if (sequence === uploadSequence.current) {
        setUploadError(localizedErrorMessage(error, t));
      }
    }
  }

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
    if (tool === "tap" || tool === "protect" || tool === "people" || busy || showingResult || ((heal.healing || retouch.applying) && editsBase(tool))) {
      return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
      return;
    }
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const radius = scaledBrushRadius();
    if (isRetouchTool(tool)) {
      // El trazo del retoque se dibuja igual que el del pincel; Alt+clic solo fija la fuente y no pinta.
      if (retouch.start(tool, point, event.altKey, radius)) {
        liveStrokeRef.current = startStroke("paint", radius, point);
        redrawOverlay();
      }
      return;
    }
    if (tool === "rect") {
      // El rectángulo es un lazo de 4 puntos que se recalcula al arrastrar:
      // reusa el dibujo/relleno del lazo sin un modo de trazo nuevo.
      rectOriginRef.current = point;
      liveStrokeRef.current = { mode: "lasso", radius: 0, points: [point] };
    } else if (tool === "lasso") {
      liveStrokeRef.current = startStroke("lasso", 0, point);
    } else {
      liveStrokeRef.current = startStroke(tool === "eraser" ? "erase" : "paint", radius, point);
    }
    redrawOverlay();
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
    if (isRetouchTool(tool)) retouch.move(point);
    const rectOrigin = rectOriginRef.current;
    if (rectOrigin) {
      liveStrokeRef.current = { ...liveStrokeRef.current, points: rectanglePoints(rectOrigin, point) };
    } else {
      liveStrokeRef.current = appendPoint(liveStrokeRef.current, point);
    }
    redrawOverlay();
  }

  function handlePointerUp() {
    panOriginRef.current = null;
    rectOriginRef.current = null;
    if (isRetouchTool(tool)) {
      liveStrokeRef.current = null;
      redrawOverlay();
      void retouch.end();
      return;
    }
    const stroke = liveStrokeRef.current;
    if (!stroke) {
      return;
    }
    liveStrokeRef.current = null;
    if (!isCommittableStroke(stroke)) {
      // Un lazo de 1-2 puntos no forma polígono: se descarta y se borra su preview.
      redrawOverlay();
      return;
    }
    // El corrector no entra al historial de máscaras: el trazo va al servicio.
    if (tool === "heal") { redrawOverlay(); void heal.commit(stroke); return; }
    appendHistory({ kind: "stroke", stroke });
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
    baseVersions.clear();
    setHistory([]);
    clearUndo();
    setSegmentError(null);
    setUploadError(null);
    setViewport(IDENTITY_VIEWPORT);
    liveStrokeRef.current = null;
    retouch.reset();
    reset();
    review.clear();
  }

  async function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if ((tool !== "tap" && tool !== "protect") || !tapAvailable || !baseInfo || segmentPendingRef.current || busy || showingResult) {
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
      appendHistory({ kind, mask, point });
    } catch (error) {
      if (sequence === segmentSequence.current) {
        setSegmentError(localizedErrorMessage(error, t));
      }
    } finally {
      if (sequence === segmentSequence.current) {
        segmentPendingRef.current = false;
        setSegmenting(false);
      }
    }
  }

  function handlePeopleBatch(entry: HistoryEntry) {
    if (entry.entries?.length || entry.failed) appendHistory(entry);
    setTool((current) => (current === "people" ? "brush" : current));
  }

  function handleBackToChoosing() {
    if (peopleBatch?.id !== undefined) dropHistoryEntry(peopleBatch.id);
    setTool("people");
  }

  // "Seguir editando este resultado": la base nueva reemplaza los trazos de Quitar en una sola acción de Deshacer; las protecciones siguen.
  async function handleUseResultAsBase() {
    if (!shownJob?.downloadUrl || busy) {
      return;
    }
    const sequence = uploadSequence.current;
    reset();
    review.clear();
    try {
      const response = await fetch(shownJob.downloadUrl);
      const blob = await response.blob();
      if (sequence !== uploadSequence.current) return;
      clearSelection();
      await pushBase(blob, sequence);
    } catch (error) {
      setUploadError(localizedErrorMessage(error, t));
    }
  }

  const canSubmit =
    Boolean(baseInfo) &&
    Boolean(modelId) &&
    maskHasContent(history) &&
    (isEraseOnlyModel || (mode === "erase" ? erasePromptText.trim().length > 0 : prompt.trim().length > 0)) &&
    !busy &&
    !segmenting;
  const panelView = tool === "people" ? "people" : job?.status === "completed" ? "review" : "action";
  const showingResult = panelView === "review" && Boolean(review.zones && resultUrl);
  const disabledTools = toolLocks({ tapAvailable, resultShown: showingResult, lamaInstalled: hasInstalledLama(models) }, t);

  return (
    <div className={baseInfo
      ? "grid min-h-0 grid-cols-1 items-start gap-3 lg:grid-cols-[148px_minmax(0,1fr)_340px]"
      : "flex flex-col gap-4"}>
      {!baseInfo && <EditorDropzone onFile={(file) => void handleBaseSelected(file)} />}

      {baseInfo && (
        <>
          <EditorToolRail
            tool={tool}
            onToolChange={setTool}
            disabledTools={disabledTools}
            toolTitles={{
              protect: t("editor.protect.hint"),
              restore: t("editor.tool.restoreHint"),
              clone: t("editor.tool.cloneHint"),
              heal: t("editor.tool.healHint"),
            }}
            brushSize={brushSize}
            onBrushSizeChange={setBrushSize}
            zoomLabel={`${zoomPercent(viewport)}%`}
            onZoomIn={() => handleZoomStep(1)}
            onZoomOut={() => handleZoomStep(-1)}
            onZoomReset={() => setViewport(IDENTITY_VIEWPORT)}
            canResetZoom={isZoomed(viewport)}
            onUndo={handleUndo}
            canUndo={lastUndo !== null}
            onClear={clearSelection}
            canClear={maskHasContent(history)}
            onClearProtection={() => appendHistory({ kind: "clearProtection" })}
            canClearProtection={hasProtection(history)}
            onNewImage={handleCloseImage}
            busy={busy}
          />

          <div className="flex min-w-0 flex-col items-center gap-3 overflow-auto">
            <div
              ref={stageRef}
              data-testid="editor-stage"
              className="relative w-fit max-w-full touch-none self-center overflow-hidden rounded border border-border"
            >
              {/* El zoom es un transform CSS sobre imagen+canvas juntos: el
                  rectángulo que devuelve el canvas ya viene transformado, así que
                  el mapeo click->píxel y el radio del pincel siguen valiendo. */}
              <div style={{ transform: transformStyle(viewport), transformOrigin: "0 0" }}>
                {showingResult
                  ? <img src={resultUrl!} alt={t("editor.review.result")} className="block max-w-full" />
                  : baseUrl && <img src={baseUrl} alt={baseInfo.originalFilename} className="block max-w-full" />}
                <canvas
                  ref={canvasRef}
                  data-testid="editor-mask-canvas"
                  className={`absolute inset-0 h-full w-full ${showingResult ? "opacity-0" : "opacity-60"} ${
                    tool === "pan"
                      ? "cursor-grab active:cursor-grabbing"
                      : tool === "tap" || tool === "protect" || tool === "people"
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
                  data-testid="editor-protect-canvas"
                  className={`pointer-events-none absolute inset-0 h-full w-full ${showingResult ? "opacity-0" : "opacity-50"}`}
                />
                {tool === "people" && peopleSelection.status === "ready" && (
                  <PeopleOverlay
                    people={peopleSelection.people}
                    choices={peopleSelection.choices}
                    hoveredId={peopleSelection.hoveredId}
                    onHover={peopleSelection.setHoveredId}
                    onToggle={peopleSelection.toggle}
                    imageWidth={baseInfo.width}
                    imageHeight={baseInfo.height}
                  />
                )}
                {tool === "clone" && retouch.source && (
                  <RetouchCrosshair
                    point={retouch.samplePoint ?? retouch.source}
                    imageWidth={baseInfo.width}
                    imageHeight={baseInfo.height}
                  />
                )}
                {panelView === "review" && review.zones && (
                  <ReviewOverlay
                    zones={review.zones}
                    hoveredZone={review.hoveredZone}
                    onHoverZone={review.setHoveredZone}
                    imageWidth={baseInfo.width}
                    imageHeight={baseInfo.height}
                  />
                )}
              </div>
              <EditorStageStatus segmenting={segmenting} applying={retouch.applying} healing={heal.healing} />
            </div>
            <p className="text-center text-xs text-text-faint">{t("editor.maskHint")}</p>
            {tool === "clone" && !retouch.source && (
              <p className="text-center text-xs text-text-dim">{t("editor.retouch.cloneNoSource")}</p>
            )}
            <p className="text-center text-xs text-text-faint">{t("editor.protect.hint")}</p>
            <p className="text-center text-xs text-text-faint">{t("editor.zoomHint")}</p>
            {segmentError && <p className="text-center text-xs text-danger">{segmentError}</p>}
          </div>

          <EditorSidePanel view={panelView}>
            <EditorSideContent
              view={panelView}
              baseInfo={baseInfo}
              baseUrl={baseUrl}
              uploadError={uploadError}
              people={{
                selection: peopleSelection,
                batch: silhouetteBatch,
                personIds: peopleBatch?.personIds ?? [],
                failed: peopleBatch?.failed ?? 0,
                originalUrl,
                onBack: handleBackToChoosing,
              }}
              review={{
                zones: review.zones,
                hoveredZone: review.hoveredZone,
                onHoverZone: review.setHoveredZone,
                onRedo: (zone, engine, text) => void zoneRedo.redo(zone, engine, text),
                sdxlAvailable: canReplace(models),
                eraserAvailable: hasEraserAuto(models),
                redoingIndex: zoneRedo.redoingIndex,
                redoProgress: zoneRedo.progressPct,
                errors: zoneRedo.errors,
                resultUrl,
              }}
              job={{
                phase,
                job: shownJob,
                errorMessage,
                busy,
                onCancel: cancel,
                onUseAsBase: () => void handleUseResultAsBase(),
              }}
              actionControls={{
                baseInfo, models, capabilitiesLoaded, mode, setMode: handleModeChange, prompt, setPrompt,
                modelId, setModelId, device, setDevice, erasePromptText, setErasePromptText,
                eraseNegative, setEraseNegative, replaceNegative, setReplaceNegative, steps, setSteps,
                guidance, setGuidance, seedText, setSeedText, strength, setStrength, isEraseOnlyModel, autoErase, ...edgeControls,
                canSubmit,
                onGenerate: () => void editorSubmit.handleGenerate(),
                onBaseSelected: (file) => void handleBaseSelected(file),
              }}
            />
          </EditorSidePanel>
        </>
      )}
    </div>
  );
}
