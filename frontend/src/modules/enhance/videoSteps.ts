export const VIDEO_STEP_IDS = ["upscale", "interpolate", "audio", "subtitles"] as const;

export type VideoStepId = (typeof VIDEO_STEP_IDS)[number];

export interface VideoStepConfig {
  modelName: string | null;
  scale: number | null;
  fpsMultiplier: number;
  targetFps: string | null;
  interpEngine: string;
  keepAudio: boolean;
  audioEnhance: string | null;
  audioRestore: string | null;
  keepSubtitles: boolean;
}

export interface UpscaleVideoStep {
  id: "upscale";
  modelName: string | null;
  scale: number;
}

export interface InterpolateVideoStep {
  id: "interpolate";
  interpEngine: string;
  fpsMultiplier: number;
  targetFps: string | null;
}

export interface AudioVideoStep {
  id: "audio";
  audioEnhance: string | null;
  audioRestore: string | null;
}

export interface SubtitlesVideoStep {
  id: "subtitles";
}

export type VideoStep =
  | UpscaleVideoStep
  | InterpolateVideoStep
  | AudioVideoStep
  | SubtitlesVideoStep;

export interface VideoStepDefaults {
  upscaleScale?: number;
  fpsMultiplier?: number;
  audioEnhance?: string;
}

function isInterpolationActive(config: VideoStepConfig): boolean {
  return config.fpsMultiplier > 1 || config.targetFps !== null;
}

function isAudioEnhancementActive(config: VideoStepConfig): boolean {
  return config.keepAudio && (config.audioEnhance !== null || config.audioRestore !== null);
}

// The backend fixes this semantic execution order, so the UI intentionally has no reorder control.
export function deriveVideoSteps(config: VideoStepConfig): VideoStep[] {
  const steps: VideoStep[] = [];

  if (config.scale !== null) {
    steps.push({
      id: "upscale",
      modelName: config.modelName,
      scale: config.scale,
    });
  }

  if (isInterpolationActive(config)) {
    steps.push({
      id: "interpolate",
      interpEngine: config.interpEngine,
      fpsMultiplier: config.fpsMultiplier,
      targetFps: config.targetFps,
    });
  }

  if (isAudioEnhancementActive(config)) {
    steps.push({
      id: "audio",
      audioEnhance: config.audioEnhance,
      audioRestore: config.audioRestore,
    });
  }

  if (config.keepSubtitles) {
    steps.push({ id: "subtitles" });
  }

  return steps;
}

export function removeVideoStep(config: VideoStepConfig, stepId: VideoStepId): VideoStepConfig {
  switch (stepId) {
    case "upscale":
      return { ...config, scale: null };
    case "interpolate":
      return { ...config, fpsMultiplier: 1, targetFps: null };
    case "audio":
      return { ...config, audioEnhance: null, audioRestore: null };
    case "subtitles":
      return { ...config, keepSubtitles: false };
  }
}

export function addVideoStep(
  config: VideoStepConfig,
  stepId: VideoStepId,
  defaults: VideoStepDefaults = {},
): VideoStepConfig {
  switch (stepId) {
    case "upscale":
      return config.scale !== null
        ? config
        : { ...config, scale: defaults.upscaleScale ?? 2 };
    case "interpolate":
      return isInterpolationActive(config)
        ? config
        : {
            ...config,
            fpsMultiplier: defaults.fpsMultiplier ?? 2,
            targetFps: null,
          };
    case "audio": {
      if (isAudioEnhancementActive(config)) {
        return config;
      }
      const hasSelectedMode = config.audioEnhance !== null || config.audioRestore !== null;
      return {
        ...config,
        keepAudio: true,
        audioEnhance: hasSelectedMode ? config.audioEnhance : (defaults.audioEnhance ?? "deepfilter"),
      };
    }
    case "subtitles":
      return config.keepSubtitles ? config : { ...config, keepSubtitles: true };
  }
}

export function getAddableVideoStepIds(config: VideoStepConfig): VideoStepId[] {
  const activeIds = new Set(deriveVideoSteps(config).map((step) => step.id));
  return VIDEO_STEP_IDS.filter((stepId) => !activeIds.has(stepId));
}
