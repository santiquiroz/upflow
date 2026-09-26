// Contrato de /api/v1/restore (app/schemas_restore.py). Vive aparte de
// apiTypes.ts porque ese archivo ya pasa el tope de tamaño del repo.

export type CropBox = [x: number, y: number, width: number, height: number];

export type Corner = [x: number, y: number];
export type Quad = [topLeft: Corner, topRight: Corner, bottomRight: Corner, bottomLeft: Corner];

export interface RestoreGeometry {
  rotate90: number;
  crop: CropBox | null;
  angle: number;
  // Esquinas de perspectiva sobre la foto girada; excluyentes con angle.
  corners?: Quad | null;
}

export interface RestoreCapture {
  autoCrop: RestoreGeometry | null;
  photos: RestoreGeometry[];
  perspective: RestoreGeometry | null;
  frameWidth: number;
  frameHeight: number;
}

export interface RestoreStepProposal {
  stepId: string;
  options: Record<string, unknown>;
  enabled: boolean;
}

export interface RestoreFinding {
  key: string;
  value: number | string | null;
  reasonKey: string;
  params: Record<string, string | number>;
  proposes: RestoreStepProposal[];
  missingPack: string | null;
  message: string | null;
}

export interface RestoreDiagnosis {
  findings: RestoreFinding[];
  suggestedPresets: string[];
  toneKind: string;
}

export interface RestoreFace {
  index: number;
  box: [number, number, number, number] | null;
  eyePx: number | null;
  sharpness: number | null;
  confidence: number | null;
  enabled: boolean;
  blend: number;
  thumbnailUrl: string | null;
}

export interface RestoreDamage {
  coverage: number | null;
  probUrl: string | null;
  largeHoles: number;
}

export interface StepEta {
  gpuSeconds: number;
  cpuSeconds: number;
}

export interface RestoreEta extends StepEta {
  perStep: Record<string, StepEta>;
}

export type RestoreStepOptions = Record<string, unknown>;

export interface RestorePresetSelection {
  steps: string[];
  options: Record<string, RestoreStepOptions>;
}

export interface RestoreAnalysis {
  token: string;
  originalName: string;
  sha256: string;
  width: number;
  height: number;
  bitDepth: number;
  hasIcc: boolean;
  geometry: RestoreGeometry;
  previewUrl: string;
  diagnosis: RestoreDiagnosis;
  proposedPreset: string;
  proposedSteps: string[];
  proposedOptions: Record<string, RestoreStepOptions>;
  presetSelections: Record<string, RestorePresetSelection>;
  faces: RestoreFace[];
  damage: RestoreDamage;
  damageOverFaces: boolean;
  eta: RestoreEta;
  capture?: RestoreCapture;
}

export interface RestoreMaskResponse {
  coverage: number;
  width: number;
  height: number;
}

export interface RestoreStepCapability {
  id: string;
  phase: string;
  strategy: string;
  labelKey: string;
  descriptionKey: string;
  inventsDetail: boolean;
  warningKey: string | null;
  pack: string | null;
  installed: boolean;
}

export interface RestorePreset {
  id: string;
  labelKey: string;
  descriptionKey: string;
  steps: string[];
}

export interface RestoreCapabilities {
  steps: RestoreStepCapability[];
  presets: RestorePreset[];
  halftoneDenoiseLimit: number;
}

export type RestoreUpscaleMode = "none" | "classic" | "ai";

// Las opciones por paso viajan en snake_case: el backend las valida con
// extra="forbid" (RestoreOptions) y no tienen alias.
export interface RestoreOptions {
  preset?: string;
  geometry?: RestoreGeometry;
  descreen?: Record<string, unknown>;
  repair?: Record<string, unknown>;
  deblock?: Record<string, unknown>;
  denoise?: Record<string, unknown>;
  tone?: Record<string, unknown>;
  upscale_mode?: RestoreUpscaleMode;
  faces?: Record<string, unknown>;
  colorize?: Record<string, unknown>;
  preview_crop?: CropBox;
  badge?: boolean;
  keep_gps?: boolean;
  photo_date?: string;
  // Foto de un lote: el backend rechaza lo elegido sobre otra foto (indices de caras, geometria...).
  batch?: boolean;
}

export interface RecomposeFaceChoice {
  enabled: boolean;
  blend: number;
}

export interface RecomposeResponse {
  sidecar: Record<string, unknown>;
}

export type RestoreArtifactName =
  | "preview"
  | "view"
  | "beforeafter"
  | "uncolored"
  | "sidecar"
  | `face:${number}:${"before" | "after"}`;
