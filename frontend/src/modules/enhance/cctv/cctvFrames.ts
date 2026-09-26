import type { CctvAnalysis, CctvFrameIndex } from "../../../services/cctv";
import type { FrameSize } from "./cctvBoxes";

export type TrimRange = readonly [number, number];

const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_MINUTE = 60;
const MILLISECONDS = 1000;

export function clampFrame(frame: number, frameCount: number): number {
  if (!Number.isFinite(frame) || frameCount <= 0) {
    return 0;
  }
  return Math.min(frameCount - 1, Math.max(0, Math.trunc(frame)));
}

export function storedSizeOf(analysis: CctvAnalysis): FrameSize {
  return { width: analysis.video.width, height: analysis.video.height };
}

// Un "1080p Lite" guarda 960x1080 y se ve a 1920x1080: la vista respeta el aspecto real.
export function displaySizeOf(analysis: CctvAnalysis): FrameSize {
  const display = analysis.video.lite?.displaySize;
  return display ? { width: display[0], height: display[1] } : storedSizeOf(analysis);
}

function frameDelta(index: CctvFrameIndex): number | null {
  if (index.medianDelta) {
    return index.medianDelta;
  }
  return index.measuredFps ? 1 / index.measuredFps : null;
}

function gapExtra(index: CctvFrameIndex, frame: number, delta: number): number {
  return index.gaps
    .filter((gap) => gap.afterFrame < frame)
    .reduce((total, gap) => total + Math.max(0, gap.end - gap.start - delta), 0);
}

// Tiempo desde el primer cuadro: paso mediano del indice mas lo que duran los huecos.
export function frameElapsedSeconds(index: CctvFrameIndex | null, frame: number): number | null {
  const delta = index ? frameDelta(index) : null;
  if (!index || delta === null) {
    return null;
  }
  return frame * delta + gapExtra(index, frame, delta);
}

function twoDigits(value: number): string {
  return String(value).padStart(2, "0");
}

export function formatTimecode(seconds: number): string {
  const totalMs = Math.round(seconds * MILLISECONDS);
  const hours = Math.floor(totalMs / (SECONDS_PER_HOUR * MILLISECONDS));
  const minutes = Math.floor(totalMs / (SECONDS_PER_MINUTE * MILLISECONDS)) % SECONDS_PER_MINUTE;
  const wholeSeconds = Math.floor(totalMs / MILLISECONDS) % SECONDS_PER_MINUTE;
  const ms = String(totalMs % MILLISECONDS).padStart(3, "0");
  return `${twoDigits(hours)}:${twoDigits(minutes)}:${twoDigits(wholeSeconds)}.${ms}`;
}

// Con fps variables el paso mediano es una aproximacion: el prefijo lo dice.
export function frameTimecode(index: CctvFrameIndex | null, frame: number): string | null {
  const seconds = frameElapsedSeconds(index, frame);
  if (seconds === null) {
    return null;
  }
  return `${index?.isVfr ? "≈ " : ""}${formatTimecode(seconds)}`;
}

export function fullRange(frameCount: number): TrimRange {
  return [0, Math.max(0, frameCount - 1)];
}

export function isTrimValid(trim: TrimRange | null, frameCount: number): boolean {
  if (trim === null) {
    return true;
  }
  const [first, last] = trim;
  return Number.isInteger(first) && Number.isInteger(last) && first >= 0 && first <= last && last < frameCount;
}

export function trimFrameCount(trim: TrimRange): number {
  return trim[1] - trim[0] + 1;
}
