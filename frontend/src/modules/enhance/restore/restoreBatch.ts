import type { RestoreOptions } from "../../../lib/restoreApiTypes";
import type { CreateRestoreJobParams } from "../../../services/restore";

const FACES_STEP = "faces";
// Modelo y mezcla valen para cualquier foto; que cara se elige lo decide la politica en cada una.
const SHARED_FACE_KEYS = new Set(["model", "blend"]);

export function batchSteps(steps: readonly string[], withFaces: boolean): string[] {
  return withFaces ? [...steps] : steps.filter((step) => step !== FACES_STEP);
}

function withoutKey(record: Record<string, unknown> | undefined, key: string): Record<string, unknown> | undefined {
  if (record === undefined) {
    return undefined;
  }
  const { [key]: _dropped, ...rest } = record;
  return rest;
}

export function batchFaceOptions(faces: Record<string, unknown> | undefined): Record<string, unknown> {
  return Object.fromEntries(Object.entries(faces ?? {}).filter(([key]) => SHARED_FACE_KEYS.has(key)));
}

// Geometria, recorte de vista previa, mascara pintada, punto gris e indices de caras son de
// la primera foto: en otra caerian en cualquier lado. Cada foto usa su deteccion automatica.
export function batchOptions(options: RestoreOptions, withFaces: boolean): RestoreOptions {
  const { geometry: _geometry, preview_crop: _previewCrop, faces, repair, tone, ...rest } = options;
  const batchRepair = withoutKey(repair, "use_user_mask");
  const batchTone = withoutKey(tone, "gray_point");
  return {
    ...rest,
    ...(batchRepair === undefined ? {} : { repair: batchRepair }),
    ...(batchTone === undefined ? {} : { tone: batchTone }),
    ...(withFaces ? { faces: batchFaceOptions(faces) } : {}),
    batch: true,
  };
}

export function batchJobParams(
  base: CreateRestoreJobParams,
  files: readonly File[],
  withFaces: boolean,
): CreateRestoreJobParams[] {
  const steps = batchSteps(base.steps, withFaces);
  const options = batchOptions(base.options, withFaces);
  return files.map((file) => ({ ...base, source: { file }, steps, options }));
}

export function batchOffersFaces(base: CreateRestoreJobParams): boolean {
  return base.steps.includes(FACES_STEP);
}

export function canBatch(base: CreateRestoreJobParams): boolean {
  return base.steps.length > 0;
}

// Si la foto solo restauro caras y el lote las deja afuera, no queda nada que correr.
export function hasBatchSteps(base: CreateRestoreJobParams, withFaces: boolean): boolean {
  return batchSteps(base.steps, withFaces).length > 0;
}
