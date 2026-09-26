import type { RecomposeFaceChoice } from "../../../lib/restoreApiTypes";

// Caras de job.metadata.restore.faces o del sidecar que devuelve /recompose:
// JSON sin tipar, asi que cada campo se valida antes de usarlo.

export interface RestoredFace {
  index: number;
  enabled: boolean;
  blend: number;
}

export type FaceDraft = Readonly<Record<number, RecomposeFaceChoice>>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isUnit(value: unknown): value is number {
  return typeof value === "number" && value >= 0 && value <= 1;
}

function readRestoredFace(value: unknown): RestoredFace | null {
  if (!isRecord(value) || value.restored !== true) {
    return null;
  }
  const { index, enabled, blend } = value;
  if (!Number.isInteger(index) || typeof enabled !== "boolean" || !isUnit(blend)) {
    return null;
  }
  return { index: index as number, enabled, blend };
}

export function readRestoredFaces(value: unknown): RestoredFace[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map(readRestoredFace).filter((face): face is RestoredFace => face !== null);
}

export function draftOf(faces: readonly RestoredFace[]): FaceDraft {
  return Object.fromEntries(faces.map((face) => [face.index, { enabled: face.enabled, blend: face.blend }]));
}

export function withFaceChoice(draft: FaceDraft, index: number, patch: Partial<RecomposeFaceChoice>): FaceDraft {
  return { ...draft, [index]: { ...draft[index], ...patch } };
}

function sameChoice(face: RestoredFace, choice: RecomposeFaceChoice | undefined): boolean {
  return choice !== undefined && choice.enabled === face.enabled && choice.blend === face.blend;
}

export function isDraftChanged(faces: readonly RestoredFace[], draft: FaceDraft): boolean {
  return faces.some((face) => !sameChoice(face, draft[face.index]));
}
