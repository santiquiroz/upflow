import { ChevronLeft, ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { useDebouncedValue } from "../../../hooks/useDebouncedValue";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { cctvFrameUrl, type CctvFrameIndex } from "../../../services/cctv";
import type { FrameSize } from "./cctvBoxes";
import { clampFrame, frameTimecode } from "./cctvFrames";
import { useImageStatus } from "./useImageStatus";

// Cada cuadro es un ffmpeg en el backend: el deslizador pide la imagen cuando se queda quieto.
export const FRAME_REQUEST_DELAY_MS = 250;

const BUTTON_CLASS =
  "inline-flex h-8 w-8 items-center justify-center rounded-sm border border-border bg-surface text-text-dim transition-colors duration-fast hover:text-text disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const FIELD_CLASS =
  "font-mono-tabular w-24 rounded-sm border border-border bg-surface px-2 py-1 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

interface FrameScrubberProps {
  token: string;
  frameCount: number;
  frame: number;
  index: CctvFrameIndex | null;
  displaySize: FrameSize;
  onFrameChange: (frame: number) => void;
  overlay?: ReactNode;
}

function FrameImage({ token, frame, displaySize, overlay }: { token: string; frame: number; displaySize: FrameSize; overlay?: ReactNode }) {
  const { t } = useTranslation();
  const src = cctvFrameUrl(token, frame);
  const image = useImageStatus(src);
  return (
    <div
      className="relative w-full overflow-hidden rounded border border-border bg-black"
      style={{ aspectRatio: `${displaySize.width} / ${displaySize.height}` }}
    >
      <img
        src={src}
        alt={t("cctv.frames.alt", { frame })}
        draggable={false}
        onLoad={image.onLoad}
        onError={image.onError}
        className="absolute inset-0 h-full w-full select-none object-fill"
      />
      {overlay}
      {image.status === "loading" && (
        <p role="status" className="pointer-events-none absolute left-2 top-2 rounded-sm bg-surface px-2 py-1 text-xs text-text-dim">
          {t("cctv.frames.loading")}
        </p>
      )}
      {image.status === "failed" && (
        <p role="alert" className="pointer-events-none absolute left-2 top-2 rounded-sm bg-surface px-2 py-1 text-xs text-danger">
          {t("cctv.frames.loadFailed")}
        </p>
      )}
    </div>
  );
}

function FrameTimecode({ index, frame }: { index: CctvFrameIndex | null; frame: number }) {
  const { t } = useTranslation();
  const timecode = frameTimecode(index, frame);
  if (timecode === null) {
    return null;
  }
  return (
    <span className="font-mono-tabular text-xs text-text-dim" title={t("cctv.frames.timecode")}>
      {timecode}
    </span>
  );
}

export function FrameScrubber({ token, frameCount, frame, index, displaySize, onFrameChange, overlay }: FrameScrubberProps) {
  const { t } = useTranslation();
  const shownFrame = useDebouncedValue(frame, FRAME_REQUEST_DELAY_MS);
  const last = Math.max(0, frameCount - 1);
  const go = (next: number) => onFrameChange(clampFrame(next, frameCount));
  return (
    <div className="flex flex-col gap-2">
      <FrameImage token={token} frame={shownFrame} displaySize={displaySize} overlay={overlay} />
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => go(frame - 1)} disabled={frame <= 0} aria-label={t("cctv.frames.previous")} className={BUTTON_CLASS}>
          <ChevronLeft aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        </button>
        <input
          type="range"
          min={0}
          max={last}
          step={1}
          value={frame}
          aria-label={t("cctv.frames.slider")}
          onChange={(event) => go(Number(event.target.value))}
          className="min-w-40 flex-1 accent-accent"
        />
        <button type="button" onClick={() => go(frame + 1)} disabled={frame >= last} aria-label={t("cctv.frames.next")} className={BUTTON_CLASS}>
          <ChevronRight aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        </button>
        <input
          type="number"
          min={0}
          max={last}
          step={1}
          value={frame}
          aria-label={t("cctv.frames.number")}
          onChange={(event) => go(Number(event.target.value))}
          className={FIELD_CLASS}
        />
        <span className="text-xs text-text-dim">{t("cctv.frames.position", { frame, last })}</span>
        <FrameTimecode index={index} frame={frame} />
      </div>
    </div>
  );
}
