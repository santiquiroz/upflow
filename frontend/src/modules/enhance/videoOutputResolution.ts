import type { TranslationParams } from "../../i18n";

export const TARGET_HEIGHT_PRESETS = [480, 720, 1080, 1440, 2160] as const;
export const OUTPUT_PIXEL_WARNING_THRESHOLD = 33_000_000;

export type VideoOutputMode = "resolution" | "multiplier";

export interface VideoDimensions {
  width: number;
  height: number;
}

export interface VideoOutputWarning {
  code: "disproportionate_output";
  key: string;
  params: TranslationParams;
}

function toEven(value: number): number {
  return value % 2 === 0 ? value : value + 1;
}

function divideAndRoundHalfEven(numerator: number, denominator: number): number {
  const quotient = Math.floor(numerator / denominator);
  const doubledRemainder = (numerator % denominator) * 2;
  if (doubledRemainder < denominator) {
    return quotient;
  }
  if (doubledRemainder > denominator) {
    return quotient + 1;
  }
  return quotient % 2 === 0 ? quotient : quotient + 1;
}

export function dimensionsForTargetHeight(
  source: VideoDimensions,
  targetHeight: number,
): VideoDimensions {
  return {
    // Python's round() uses half-to-even; mirror it before the backend's even
    // yuv420p adjustment so the preview cannot drift by two pixels at a .5 tie.
    width: toEven(divideAndRoundHalfEven(source.width * targetHeight, source.height)),
    height: toEven(targetHeight),
  };
}

export function dimensionsForScale(
  source: VideoDimensions,
  scale: number,
): VideoDimensions {
  return {
    width: toEven(source.width * scale),
    height: toEven(source.height * scale),
  };
}

export function outputDimensions(
  source: VideoDimensions | null,
  mode: VideoOutputMode,
  scale: number | null,
  targetHeight: number,
): VideoDimensions | null {
  if (source === null) {
    return null;
  }
  if (mode === "resolution") {
    return dimensionsForTargetHeight(source, targetHeight);
  }
  return scale === null ? null : dimensionsForScale(source, scale);
}

function formatMegapixels(dimensions: VideoDimensions): string {
  return ((dimensions.width * dimensions.height) / 1_000_000).toFixed(1);
}

// Warnings carry translation data, not assembled copy, matching the model
// installation warning contract. Missing source measurements fail open.
export function buildVideoOutputWarnings(
  source: VideoDimensions | null,
  mode: VideoOutputMode,
  scale: number | null,
): VideoOutputWarning[] {
  if (source === null || mode !== "multiplier" || scale === null) {
    return [];
  }

  const output = dimensionsForScale(source, scale);
  if (output.width * output.height <= OUTPUT_PIXEL_WARNING_THRESHOLD) {
    return [];
  }

  const commonParams: TranslationParams = {
    width: output.width,
    height: output.height,
    megapixels: formatMegapixels(output),
  };

  if (source.height >= 2160) {
    return [
      {
        code: "disproportionate_output",
        key: "video.output.warning.keepSource",
        params: {
          ...commonParams,
          sourceHeight: source.height,
        },
      },
    ];
  }

  return [
    {
      code: "disproportionate_output",
      key: "video.output.warning.target",
      params: {
        ...commonParams,
        suggestedHeight: 2160,
      },
    },
  ];
}

// /video/analyze currently stages the upload and returns tracks, but not source
// dimensions. Browser metadata provides the intrinsic size without another API.
export function readVideoDimensions(file: File): Promise<VideoDimensions> {
  return new Promise((resolve, reject) => {
    if (typeof URL.createObjectURL !== "function") {
      reject(new Error("Video metadata is unavailable"));
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement("video");

    const cleanup = () => {
      video.onloadedmetadata = null;
      video.onerror = null;
      video.removeAttribute("src");
      URL.revokeObjectURL(objectUrl);
    };

    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const dimensions = {
        width: video.videoWidth,
        height: video.videoHeight,
      };
      cleanup();
      if (dimensions.width > 0 && dimensions.height > 0) {
        resolve(dimensions);
      } else {
        reject(new Error("Video metadata has no dimensions"));
      }
    };
    video.onerror = () => {
      cleanup();
      reject(new Error("Could not read video metadata"));
    };
    video.src = objectUrl;
  });
}
