import { Eye } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { cctvFilterPreviewUrl, cctvFrameUrl, type CctvStepRequest } from "../../../services/cctv";
import type { FrameSize } from "./cctvBoxes";
import { useImageStatus } from "./useImageStatus";

interface FilterPreviewProps {
  token: string;
  frame: number;
  steps: readonly CctvStepRequest[];
  skipsAiSteps: boolean;
  displaySize: FrameSize;
}

interface PreviewRequest {
  frame: number;
  steps: readonly CctvStepRequest[];
  key: string;
}

const DEFAULT_BLEND = 50;

function requestKey(frame: number, steps: readonly CctvStepRequest[]): string {
  return `${frame}:${JSON.stringify(steps)}`;
}

function PreviewStatus({ status }: { status: "loading" | "ready" | "failed" }) {
  const { t } = useTranslation();
  if (status === "loading") {
    return <p role="status" className="text-xs text-text-dim">{t("cctv.preview.loading")}</p>;
  }
  if (status === "failed") {
    return <p role="alert" className="text-xs text-danger">{t("cctv.preview.failed")}</p>;
  }
  return null;
}

function BlendedFrames({ token, request, displaySize }: { token: string; request: PreviewRequest; displaySize: FrameSize }) {
  const { t } = useTranslation();
  const [blend, setBlend] = useState(DEFAULT_BLEND);
  const processedSrc = cctvFilterPreviewUrl(token, request.frame, request.steps);
  const processed = useImageStatus(processedSrc);
  return (
    <div className="flex flex-col gap-2">
      <div
        className="relative w-full overflow-hidden rounded border border-border bg-black"
        style={{ aspectRatio: `${displaySize.width} / ${displaySize.height}` }}
      >
        <img
          src={cctvFrameUrl(token, request.frame)}
          alt={t("cctv.preview.alt.original", { frame: request.frame })}
          className="absolute inset-0 h-full w-full object-fill"
        />
        <img
          src={processedSrc}
          alt={t("cctv.preview.alt.processed", { frame: request.frame })}
          onLoad={processed.onLoad}
          onError={processed.onError}
          style={{ opacity: blend / 100 }}
          className="absolute inset-0 h-full w-full object-fill"
        />
      </div>
      <PreviewStatus status={processed.status} />
      <label className="flex items-center gap-2 text-xs text-text-dim">
        <span>{t("cctv.preview.original")}</span>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={blend}
          aria-label={t("cctv.preview.blend")}
          aria-valuetext={t("cctv.preview.blendValue", { percent: blend })}
          onChange={(event) => setBlend(Number(event.target.value))}
          className="flex-1 accent-accent"
        />
        <span>{t("cctv.preview.processed")}</span>
      </label>
    </div>
  );
}

export function CctvFilterPreview({ token, frame, steps, skipsAiSteps, displaySize }: FilterPreviewProps) {
  const { t } = useTranslation();
  const [request, setRequest] = useState<PreviewRequest | null>(null);
  const currentKey = requestKey(frame, steps);
  const isStale = request !== null && request.key !== currentKey;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => setRequest({ frame, steps, key: currentKey })}
          disabled={steps.length === 0}
          className="inline-flex w-fit items-center gap-2 rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text transition-colors duration-fast hover:border-accent disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Eye aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("cctv.preview.action")}
        </button>
        <span className="text-xs text-text-faint">{t("cctv.preview.hint")}</span>
      </div>
      {skipsAiSteps && <p className="text-xs text-text-dim">{t("cctv.preview.classicOnly")}</p>}
      {isStale && <p role="status" className="text-xs text-warn">{t("cctv.preview.stale")}</p>}
      {request && <BlendedFrames key={request.key} token={token} request={request} displaySize={displaySize} />}
    </div>
  );
}
