import { useEffect, useState } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { useGenerationCapabilities } from "../../hooks/useGenerationJob";
import { useTranslation } from "../../i18n/LocaleProvider";
import type {
  GenerationModelSummary,
  InitImageResponse,
  InsertObjectResponse,
} from "../../lib/apiTypes";
import { insertObject, segmentEditorObject } from "../../services/editor";
import type { InsertObjectParams } from "../../services/editor";
import { uploadGenerationInitImage } from "../../services/generation";
import { toImagePoint } from "./maskCanvas";
import type { BrushPoint } from "./maskCanvas";

const SCALE_MIN = 10;
const SCALE_MAX = 200;
// Por defecto el objeto entra ocupando un tercio del ancho de la imagen destino.
const DEFAULT_TARGET_FRACTION = 1 / 3;
// Con targetMaskToken el backend ignora la geometria: van los defaults del schema.
const REPLACE_GEOMETRY = { x: 0, y: 0, width: 8, height: 8 };

type BusyState = "idle" | "uploading" | "segmenting" | "inserting";
type InsertMode = "point" | "replace";

interface PlacementRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

function clampScale(value: number): number {
  return Math.min(SCALE_MAX, Math.max(SCALE_MIN, Math.round(value)));
}

function defaultScaleFor(targetWidth: number, sourceWidth: number): number {
  if (sourceWidth <= 0) {
    return 100;
  }
  return clampScale(((targetWidth * DEFAULT_TARGET_FRACTION) / sourceWidth) * 100);
}

function clampPosition(value: number, max: number): number {
  return Math.min(max, Math.max(0, Math.round(value)));
}

function scaledSize(source: InitImageResponse, scale: number): { width: number; height: number } {
  return {
    width: Math.max(1, Math.round((source.width * scale) / 100)),
    height: Math.max(1, Math.round((source.height * scale) / 100)),
  };
}

function cornerFromCenter(center: number, size: number, limit: number): number {
  return clampPosition(center - size / 2, Math.max(0, limit - size));
}

function placementRect(
  target: InitImageResponse,
  source: InitImageResponse,
  center: BrushPoint,
  scale: number,
): PlacementRect {
  const size = scaledSize(source, scale);
  return {
    x: cornerFromCenter(center.x, size.width, target.width),
    y: cornerFromCenter(center.y, size.height, target.height),
    width: size.width,
    height: size.height,
  };
}

function toPercent(value: number, total: number): string {
  return `${(value / total) * 100}%`;
}

function installedInpaintModels(models: GenerationModelSummary[]): GenerationModelSummary[] {
  return models.filter((model) => model.status === "installed" && model.inpaintOnly === true);
}

function createObjectUrl(blob: Blob): string | null {
  return typeof URL.createObjectURL === "function" ? URL.createObjectURL(blob) : null;
}

function describeError(cause: unknown, fallback: string): string {
  const message = cause instanceof Error ? cause.message : String(cause);
  return message.trim() !== "" ? message : fallback;
}

export function InsertObjectPanel({ targetInfo }: { targetInfo: InitImageResponse }) {
  const { t } = useTranslation();
  const capabilitiesQuery = useGenerationCapabilities();

  const [mode, setMode] = useState<InsertMode>("point");
  const [sourceInfo, setSourceInfo] = useState<InitImageResponse | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [maskToken, setMaskToken] = useState<string | null>(null);
  const [targetMaskUrl, setTargetMaskUrl] = useState<string | null>(null);
  const [targetMaskToken, setTargetMaskToken] = useState<string | null>(null);
  const [center, setCenter] = useState<BrushPoint>(() => ({
    x: Math.round(targetInfo.width / 2),
    y: Math.round(targetInfo.height / 2),
  }));
  const [scale, setScale] = useState(100);
  const [matchColor, setMatchColor] = useState(true);
  const [harmonize, setHarmonize] = useState(false);
  const [modelId, setModelId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState<BusyState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [composite, setComposite] = useState<InsertObjectResponse | null>(null);

  const inpaintModels = installedInpaintModels(capabilitiesQuery.data?.models ?? []);
  const compositeDataUrl = composite ? `data:image/png;base64,${composite.compositePngBase64}` : null;
  const ghostRect =
    mode === "point" && sourceInfo && maskToken
      ? placementRect(targetInfo, sourceInfo, center, scale)
      : null;
  const canInsert =
    sourceInfo !== null &&
    maskToken !== null &&
    (mode === "point" || targetMaskToken !== null) &&
    busy === "idle" &&
    (!harmonize || inpaintModels.some((model) => model.id === modelId));

  useEffect(() => {
    setCenter({ x: Math.round(targetInfo.width / 2), y: Math.round(targetInfo.height / 2) });
    setComposite(null);
    setTargetMaskToken(null);
    setTargetMaskUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return null;
    });
  }, [targetInfo]);

  useEffect(() => () => {
    if (sourceUrl) {
      URL.revokeObjectURL(sourceUrl);
    }
  }, [sourceUrl]);

  useEffect(() => () => {
    if (maskUrl) {
      URL.revokeObjectURL(maskUrl);
    }
  }, [maskUrl]);

  useEffect(() => () => {
    if (targetMaskUrl) {
      URL.revokeObjectURL(targetMaskUrl);
    }
  }, [targetMaskUrl]);

  useEffect(() => {
    if (harmonize && !modelId && inpaintModels.length > 0) {
      setModelId(inpaintModels[0].id);
    }
  }, [harmonize, modelId, inpaintModels]);

  function switchMode(next: InsertMode) {
    setMode(next);
    setError(null);
    setComposite(null);
  }

  async function handleSourceSelected(file: File) {
    setBusy("uploading");
    setError(null);
    setComposite(null);
    setMaskToken(null);
    setMaskUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return null;
    });
    try {
      const response = await uploadGenerationInitImage(file);
      setSourceInfo(response);
      setScale(defaultScaleFor(targetInfo.width, response.width));
      setSourceUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return createObjectUrl(file);
      });
    } catch (cause) {
      setError(describeError(cause, t("editor.insert.error")));
    } finally {
      setBusy("idle");
    }
  }

  async function handleSourceClick(event: React.MouseEvent<HTMLImageElement>) {
    if (!sourceInfo || busy !== "idle") {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const point = toImagePoint(event.clientX, event.clientY, rect, sourceInfo.width, sourceInfo.height);
    setBusy("segmenting");
    setError(null);
    try {
      const blob = await segmentEditorObject(sourceInfo.initImageToken, point.x, point.y);
      setMaskUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return createObjectUrl(blob);
      });
      const maskFile = new File([blob], "mask.png", { type: "image/png" });
      const maskResponse = await uploadGenerationInitImage(maskFile);
      setMaskToken(maskResponse.initImageToken);
    } catch (cause) {
      setError(describeError(cause, t("editor.insert.error")));
    } finally {
      setBusy("idle");
    }
  }

  function handleMapClick(event: React.MouseEvent<HTMLButtonElement>) {
    if (busy !== "idle") {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const point = toImagePoint(event.clientX, event.clientY, rect, targetInfo.width, targetInfo.height);
    if (mode === "point") {
      setCenter({ x: Math.round(point.x), y: Math.round(point.y) });
      return;
    }
    void segmentTargetObject(point);
  }

  async function segmentTargetObject(point: BrushPoint) {
    setBusy("segmenting");
    setError(null);
    try {
      const blob = await segmentEditorObject(targetInfo.initImageToken, point.x, point.y);
      setTargetMaskUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return createObjectUrl(blob);
      });
      const maskFile = new File([blob], "target-mask.png", { type: "image/png" });
      const maskResponse = await uploadGenerationInitImage(maskFile);
      setTargetMaskToken(maskResponse.initImageToken);
    } catch (cause) {
      setError(describeError(cause, t("editor.insert.error")));
    } finally {
      setBusy("idle");
    }
  }

  function buildInsertParams(): InsertObjectParams | null {
    if (!sourceInfo || !maskToken) {
      return null;
    }
    const base = {
      targetToken: targetInfo.initImageToken,
      sourceToken: sourceInfo.initImageToken,
      sourceMaskToken: maskToken,
      matchColor,
      harmonize,
      modelId: harmonize && modelId ? modelId : undefined,
      prompt: harmonize && prompt.trim() !== "" ? prompt.trim() : undefined,
    };
    if (mode === "replace") {
      if (!targetMaskToken) {
        return null;
      }
      return { ...base, ...REPLACE_GEOMETRY, targetMaskToken };
    }
    return { ...base, ...placementRect(targetInfo, sourceInfo, center, scale) };
  }

  async function handleInsert() {
    const params = buildInsertParams();
    if (!params) {
      return;
    }
    setBusy("inserting");
    setError(null);
    try {
      const response = await insertObject(params);
      setComposite(response);
    } catch (cause) {
      setError(describeError(cause, t("editor.insert.error")));
    } finally {
      setBusy("idle");
    }
  }

  return (
    <AccordionSection
      title={t("editor.insert.title")}
      summary={t("editor.insert.summary")}
      tooltip={t("editor.insert.tooltip")}
    >
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <label
            htmlFor="insert-object-source-input"
            className={`cursor-pointer rounded border border-border px-4 py-2 text-sm text-text hover:bg-surface-2 ${
              busy !== "idle" ? "pointer-events-none opacity-40" : ""
            }`}
          >
            {t("editor.insert.uploadSource")}
          </label>
          <input
            id="insert-object-source-input"
            type="file"
            accept="image/*"
            className="sr-only"
            aria-label={t("editor.insert.sourceInputLabel")}
            disabled={busy !== "idle"}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void handleSourceSelected(file);
              }
              event.target.value = "";
            }}
          />
          {busy === "uploading" && (
            <span className="text-xs text-text-dim">{t("editor.insert.uploading")}</span>
          )}
        </div>

        {sourceInfo && sourceUrl && (
          <>
            <div className="relative w-fit max-w-full self-center">
              <img
                src={sourceUrl}
                alt={sourceInfo.originalFilename}
                data-testid="insert-source-image"
                className="block max-h-64 max-w-full cursor-crosshair rounded border border-border"
                onClick={(event) => void handleSourceClick(event)}
              />
              {maskUrl && (
                <img
                  src={maskUrl}
                  alt=""
                  data-testid="insert-mask-overlay"
                  className="pointer-events-none absolute inset-0 h-full w-full opacity-60 mix-blend-screen"
                />
              )}
            </div>
            <p className="text-center text-xs text-text-faint">{t("editor.insert.clickHint")}</p>
          </>
        )}
        {busy === "segmenting" && (
          <p className="text-center text-xs text-text-dim">{t("editor.insert.segmenting")}</p>
        )}

        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="radio"
              name="insert-object-mode"
              checked={mode === "point"}
              disabled={busy !== "idle"}
              onChange={() => switchMode("point")}
            />
            {t("editor.insert.modePoint")}
          </label>
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="radio"
              name="insert-object-mode"
              checked={mode === "replace"}
              disabled={busy !== "idle"}
              onChange={() => switchMode("replace")}
            />
            {t("editor.insert.modeReplace")}
          </label>
        </div>

        <button
          type="button"
          data-testid="insert-placement-map"
          aria-label={t("editor.insert.mapLabel")}
          disabled={busy !== "idle"}
          onClick={handleMapClick}
          className="relative block w-full max-w-md cursor-crosshair self-center overflow-hidden rounded border border-border bg-surface-2 p-0"
          style={{ aspectRatio: `${targetInfo.width} / ${targetInfo.height}` }}
        >
          {ghostRect && sourceUrl && maskUrl && (
            <img
              src={sourceUrl}
              alt=""
              data-testid="insert-ghost"
              className="pointer-events-none absolute opacity-60"
              style={{
                left: toPercent(ghostRect.x, targetInfo.width),
                top: toPercent(ghostRect.y, targetInfo.height),
                width: toPercent(ghostRect.width, targetInfo.width),
                height: toPercent(ghostRect.height, targetInfo.height),
                maskImage: `url(${maskUrl})`,
                WebkitMaskImage: `url(${maskUrl})`,
                maskMode: "luminance",
                maskSize: "100% 100%",
                WebkitMaskSize: "100% 100%",
              }}
            />
          )}
          {mode === "replace" && targetMaskUrl && (
            <img
              src={targetMaskUrl}
              alt=""
              data-testid="insert-target-mask-overlay"
              className="pointer-events-none absolute inset-0 h-full w-full opacity-50 mix-blend-screen"
            />
          )}
        </button>
        <p className="text-center text-xs text-text-faint">
          {mode === "point" ? t("editor.insert.mapCaption") : t("editor.insert.mapCaptionReplace")}
        </p>

        {mode === "point" && (
          <label className="flex max-w-xs flex-col gap-1 text-sm text-text">
            <span>
              {t("editor.insert.scale")}{" "}
              <span className="font-mono-tabular text-text-dim">{scale}</span>
            </span>
            <input
              type="range"
              min={SCALE_MIN}
              max={SCALE_MAX}
              value={scale}
              disabled={busy !== "idle"}
              aria-label={t("editor.insert.scale")}
              onChange={(event) => setScale(Number(event.target.value))}
            />
          </label>
        )}

        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="checkbox"
              checked={matchColor}
              disabled={busy !== "idle"}
              onChange={(event) => setMatchColor(event.target.checked)}
            />
            {t("editor.insert.matchColor")}
          </label>
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="checkbox"
              checked={harmonize}
              disabled={busy !== "idle"}
              onChange={(event) => setHarmonize(event.target.checked)}
            />
            {t("editor.insert.harmonize")}
          </label>
        </div>

        {harmonize && inpaintModels.length === 0 && (
          <p className="text-xs text-danger">{t("editor.insert.needsInpaintModel")}</p>
        )}
        {harmonize && inpaintModels.length > 0 && (
          <label className="flex flex-col gap-1 text-sm text-text">
            {t("editor.insert.model")}
            <select
              className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
              value={modelId ?? ""}
              disabled={busy !== "idle"}
              onChange={(event) => setModelId(event.target.value || null)}
            >
              {inpaintModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {harmonize && (
          <label className="flex flex-col gap-1 text-sm text-text">
            {t("editor.insert.integrationPrompt")}
            <textarea
              rows={2}
              value={prompt}
              disabled={busy !== "idle"}
              placeholder={t("editor.insert.integrationPromptPlaceholder")}
              onChange={(event) => setPrompt(event.target.value)}
              className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
            />
          </label>
        )}

        {error && <p className="text-sm text-danger">{error}</p>}

        <button
          type="button"
          disabled={!canInsert}
          onClick={() => void handleInsert()}
          className="self-start rounded bg-accent px-5 py-2 text-sm font-medium text-surface disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy === "inserting" ? t("editor.insert.working") : t("editor.insert.insert")}
        </button>

        {composite && compositeDataUrl && (
          <div className="flex flex-col gap-2">
            <img
              src={compositeDataUrl}
              alt={t("editor.insert.resultAlt")}
              data-testid="insert-composite"
              className="max-w-full self-center rounded border border-border"
            />
            <div className="flex items-center gap-3">
              <a
                href={compositeDataUrl}
                download="insert-object.png"
                className="rounded border border-border px-4 py-2 text-sm text-text hover:bg-surface-2"
              >
                {t("editor.insert.download")}
              </a>
              {composite.jobId && (
                <p className="text-xs text-text-dim">{t("editor.insert.harmonizing")}</p>
              )}
            </div>
          </div>
        )}
      </div>
    </AccordionSection>
  );
}
