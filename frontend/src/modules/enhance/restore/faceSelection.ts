import type { RestoreFace, RestoreOptions, RestoreStepOptions } from "../../../lib/restoreApiTypes";

export type FaceTier = "restore" | "alreadyClear" | "small" | "tooSmallFaithful" | "tooSmall";

// Espejo de face_geometry.default_face_policy (§2.4, umbrales [propuesta] hasta P1-ID).
const RESTORE_MIN_EYE_PX = 32;
const SMALL_MIN_EYE_PX = 16;
const FORCEABLE_MIN_EYE_PX = 8;
const SHARP_FACE_MIN = 0.06;
const DEFAULT_FACE_BLEND = 0.6;

type FaceMeasures = Pick<RestoreFace, "eyePx" | "sharpness">;

function largeFaceTier(sharpness: number): FaceTier {
  return sharpness >= SHARP_FACE_MIN ? "alreadyClear" : "restore";
}

// Sin medida no hay como saber si la cara da para restaurar: se trata como la mas chica.
export function faceTier(face: FaceMeasures): FaceTier {
  const eyePx = face.eyePx ?? 0;
  if (eyePx >= RESTORE_MIN_EYE_PX) return largeFaceTier(face.sharpness ?? 0);
  if (eyePx >= SMALL_MIN_EYE_PX) return "small";
  if (eyePx >= FORCEABLE_MIN_EYE_PX) return "tooSmallFaithful";
  return "tooSmall";
}

export function isSelectable(tier: FaceTier): boolean {
  return tier !== "tooSmall";
}

export function needsConfirmation(tier: FaceTier): boolean {
  return tier === "tooSmallFaithful";
}

export function tierLabelKey(tier: FaceTier): string {
  return `restore.face.${tier}`;
}

export function bySize(faces: readonly RestoreFace[]): RestoreFace[] {
  return [...faces].sort((a, b) => (b.eyePx ?? 0) - (a.eyePx ?? 0));
}

export function isFaceSelected(face: RestoreFace): boolean {
  return face.enabled && isSelectable(faceTier(face));
}

export function selectedFaceIndices(faces: readonly RestoreFace[]): number[] {
  return faces.filter(isFaceSelected).map((face) => face.index);
}

// La mezcla del paso vale para las caras que la politica encendio y nadie retoco;
// una cara opcional conserva la suya, como en photo_restore_runners._default_blend.
export function effectiveFaceBlend(current: RestoreFace, proposed: RestoreFace | undefined, stepBlend: number): number {
  if (proposed === undefined || !proposed.enabled || current.blend !== proposed.blend) {
    return current.blend;
  }
  return stepBlend;
}

export function stepBlendOf(options: RestoreStepOptions): number {
  return typeof options.blend === "number" ? options.blend : DEFAULT_FACE_BLEND;
}

function proposedFace(proposed: readonly RestoreFace[], index: number): RestoreFace | undefined {
  return proposed.find((face) => face.index === index);
}

function perFaceBlends(
  current: readonly RestoreFace[],
  proposed: readonly RestoreFace[],
  stepBlend: number,
): Record<string, number> {
  return Object.fromEntries(
    current
      .filter(isFaceSelected)
      .map((face) => [String(face.index), effectiveFaceBlend(face, proposedFace(proposed, face.index), stepBlend)]),
  );
}

// Con sesion el backend restaura las caras de la sesion con su propia mezcla:
// se nombran todas para que corra justo lo que muestra la grilla.
export function withFaceChoices(
  options: RestoreOptions,
  current: readonly RestoreFace[],
  proposed: readonly RestoreFace[],
): RestoreOptions {
  if (options.faces === undefined) {
    return options;
  }
  const stepBlend = stepBlendOf(options.faces);
  return {
    ...options,
    faces: {
      ...options.faces,
      selected: selectedFaceIndices(current),
      per_face: perFaceBlends(current, proposed, stepBlend),
    },
  };
}
