import type { RestoreAnalysis, RestoreFace } from "../../../lib/restoreApiTypes";

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
    faces: [makeFace()],
    damage: { coverage: 0.04, probUrl: null, largeHoles: 0 },
    damageOverFaces: false,
    eta: { gpuSeconds: 40, cpuSeconds: 240 },
    ...overrides,
  };
}
