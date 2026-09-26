import type { JobResponse } from "../../../lib/apiTypes";
import type {
  RestoreAnalysis,
  RestoreCapabilities,
  RestoreFace,
  RestoreStepCapability,
} from "../../../lib/restoreApiTypes";

export function makeFace(overrides: Partial<RestoreFace> = {}): RestoreFace {
  return {
    index: 0,
    box: [10, 10, 60, 60],
    eyePx: 40,
    sharpness: 0.2,
    confidence: 0.98,
    enabled: true,
    blend: 0.6,
    thumbnailUrl: "/api/v1/restore/analysis/tok-1/face-0.jpg",
    ...overrides,
  };
}

export function makeAnalysis(overrides: Partial<RestoreAnalysis> = {}): RestoreAnalysis {
  return {
    token: "tok-1",
    originalName: "grandma.jpg",
    sha256: "a".repeat(64),
    width: 1200,
    height: 800,
    bitDepth: 8,
    hasIcc: false,
    geometry: { rotate90: 0, crop: null, angle: 0 },
    previewUrl: "/api/v1/restore/analysis/tok-1/preview.jpg",
    diagnosis: { findings: [], suggestedPresets: ["gentle"], toneKind: "bw" },
    proposedPreset: "gentle",
    proposedSteps: ["repair"],
    proposedOptions: { repair: { sensitivity: 0.5 } },
    presetSelections: {
      gentle: { steps: ["repair"], options: { repair: { sensitivity: 0.5 } } },
    },
    faces: [makeFace()],
    damage: { coverage: 0.04, probUrl: null, largeHoles: 0 },
    damageOverFaces: false,
    eta: { gpuSeconds: 40, cpuSeconds: 240, perStep: { repair: { gpuSeconds: 40, cpuSeconds: 240 } } },
    ...overrides,
  };
}

const CHAIN_STEPS: [id: string, pack: string | null, warningKey: string | null][] = [
  ["descreen", null, null],
  ["repair", "restore-core", "restore.warning.inventsDetail"],
  ["deblock", "restore-core", null],
  ["denoise", "restore-core", null],
  ["tone", null, null],
  ["faces", "restore-faces", "restore.warning.faces"],
  ["colorize", "restore-colorize", "restore.warning.colorize"],
];

export function makeStepCapability([id, pack, warningKey]: (typeof CHAIN_STEPS)[number]): RestoreStepCapability {
  return {
    id,
    phase: id === "faces" || id === "colorize" ? "output" : "native",
    strategy: pack ? "model" : "dsp",
    labelKey: `restore.step.${id}`,
    descriptionKey: `restore.step.${id}.description`,
    inventsDetail: warningKey !== null,
    warningKey,
    pack,
    installed: true,
  };
}

export function makeCapabilities(missingPacks: string[] = []): RestoreCapabilities {
  const steps = CHAIN_STEPS.map(makeStepCapability).map((step) => ({
    ...step,
    installed: step.pack === null || !missingPacks.includes(step.pack),
  }));
  const presets = ["gentle", "heavy_damage", "newspaper", "faded_color_print", "portrait"].map((id) => ({
    id,
    labelKey: `restore.preset.${id}`,
    descriptionKey: `restore.preset.${id}.description`,
    steps: [],
  }));
  return { steps, presets, halftoneDenoiseLimit: 0.3 };
}

export function makeRestoreMetadata(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    artifacts: ["preview", "view", "beforeafter", "sidecar"],
    downloadNames: {
      restored: "grandma_restored.png",
      uncolored: "grandma_uncolored.png",
      before_after: "grandma_before-after.jpg",
      sidecar: "grandma_restore.json",
    },
    viewFullResolution: true,
    compositeReasons: [],
    digitalSourceType: "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced",
    ...overrides,
  };
}

export function makeCompletedRestoreJob(restore: Record<string, unknown> | undefined): JobResponse {
  return {
    jobId: "job-1",
    status: "completed",
    originalFilename: "grandma.jpg",
    modelName: "",
    scale: 1,
    outputFormat: "png",
    modelId: null,
    device: "cpu",
    createdAt: "2026-09-25T10:00:00Z",
    startedAt: "2026-09-25T10:00:01Z",
    finishedAt: "2026-09-25T10:00:40Z",
    error: null,
    ownerId: null,
    metadata: restore === undefined ? {} : { restore },
    progressPct: 100,
    downloadUrl: "/api/v1/jobs/job-1/download",
    restoreSteps: ["repair"],
  };
}
