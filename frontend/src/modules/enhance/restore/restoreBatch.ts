import type { RestoreOptions } from "../../../lib/restoreApiTypes";
import type { CreateRestoreJobParams } from "../../../services/restore";

// Las caras en lote son P4-BATCH: sin la grilla no se restaura una cara que nadie vio.
const FACES_STEP = "faces";

export function batchSteps(steps: readonly string[]): string[] {
  return steps.filter((step) => step !== FACES_STEP);
}

function withoutKey(record: Record<string, unknown> | undefined, key: string): Record<string, unknown> | undefined {
  if (record === undefined) {
    return undefined;
  }
  const { [key]: _dropped, ...rest } = record;
  return rest;
}

// Geometria, recorte de vista previa, mascara pintada y punto gris son de la primera
// foto: en otra caerian en cualquier lado. Cada foto usa su mascara automatica.
export function batchOptions(options: RestoreOptions): RestoreOptions {
  const { geometry: _geometry, preview_crop: _previewCrop, faces: _faces, repair, tone, ...rest } = options;
  const perPhoto: RestoreOptions = { ...rest };
  const batchRepair = withoutKey(repair, "use_user_mask");
  const batchTone = withoutKey(tone, "gray_point");
  return {
    ...perPhoto,
    ...(batchRepair === undefined ? {} : { repair: batchRepair }),
    ...(batchTone === undefined ? {} : { tone: batchTone }),
  };
}

export function batchJobParams(base: CreateRestoreJobParams, files: readonly File[]): CreateRestoreJobParams[] {
  const steps = batchSteps(base.steps);
  const options = batchOptions(base.options);
  return files.map((file) => ({ ...base, source: { file }, steps, options }));
}

// Si la foto solo restauro caras, en lote no queda nada que correr.
export function canBatch(base: CreateRestoreJobParams): boolean {
  return batchSteps(base.steps).length > 0;
}

export function batchSkipsFaces(base: CreateRestoreJobParams): boolean {
  return base.steps.includes(FACES_STEP);
}
