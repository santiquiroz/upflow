import { AlertTriangle } from "lucide-react";
import type { VideoEncoder } from "../lib/apiTypes";

// Measured on an RX 7800 XT: libx265 -preset slow encodes 5120x2880 (a 4x upscale
// of 720p) at 5.15 fps. Encode speed tracks output pixel count, so scaling that
// anchor by (4/scale)^2 gives a defensible estimate for the other ratios.
const MEASURED_SLOW_FPS_AT_4X = 5.15;
const REFERENCE_EPISODE_FRAMES = 24 * 60 * 24; // a 24-minute episode at 24 fps
const SLOW_PRESETS = new Set(["slow", "veryslow"]);

export function estimateSoftwareEncodeMinutes(scale: number): number {
  const fps = MEASURED_SLOW_FPS_AT_4X * (16 / (scale * scale));
  return Math.round(REFERENCE_EPISODE_FRAMES / fps / 60);
}

export function shouldWarnAboutSlowPreset(
  encoder: VideoEncoder,
  preset: string,
  scale: number | null,
): boolean {
  // Only the software path pays the preset cost; the GPU encoders ignore
  // x264/x265 presets entirely. Below 3x the output is small enough not to nag.
  return encoder === "software" && SLOW_PRESETS.has(preset) && (scale ?? 0) >= 3;
}

interface SlowPresetCostHintProps {
  encoder: VideoEncoder;
  preset: string;
  scale: number | null;
}

export function SlowPresetCostHint({ encoder, preset, scale }: SlowPresetCostHintProps) {
  if (!shouldWarnAboutSlowPreset(encoder, preset, scale)) {
    return null;
  }
  const minutes = estimateSoftwareEncodeMinutes(scale ?? 4);
  return (
    <p
      role="note"
      className="mt-3 flex gap-2 rounded border border-accent bg-surface-2 px-3 py-2 text-xs text-text-dim"
    >
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-accent" strokeWidth={1.75} />
      <span>
        Heads up: <span className="text-text">{preset}</span> on the software encoder at {scale}x is slow —
        roughly <span className="text-text">{minutes} min of encoding alone</span> for a 24-minute episode.
        Switching to <span className="text-text">Auto (GPU)</span> brings that down to minutes.
      </span>
    </p>
  );
}
