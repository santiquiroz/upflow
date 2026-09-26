/** Lo minimo que hace falta: sirve para un `JobStage` y para un `StepperItem`. */
export interface LabelledStage {
  key: string;
  label: string;
}

// El backend arma el texto de cada etapa en ingles (app/services/progress.py) y
// no tiene forma de saber en que idioma se lee la pantalla. La CLAVE si viaja
// (`stages[].key`), asi que la traduccion se hace por clave y el label del
// backend queda como respaldo: una etapa nueva del servidor sigue mostrandose
// con su nombre en vez de con la clave cruda.

const CLEANUP_STAGE_PREFIX = "cleanup_";
const CLEANUP_LABEL_SEPARATOR = ": ";

export function stageTranslationKey(stageKey: string): string {
  return `job.stage.${stageKey}`;
}

// La cadena de limpieza produce UNA etapa por pasada, con clave dinamica
// (`cleanup_<modelo>`) y el nombre propio del modelo dentro del label. Ese
// nombre no se traduce (es una marca); lo que se traduce es el molde.
function cleanupModelName(stage: LabelledStage): string {
  const separator = stage.label.indexOf(CLEANUP_LABEL_SEPARATOR);
  if (separator === -1) {
    return stage.key.slice(CLEANUP_STAGE_PREFIX.length);
  }
  return stage.label.slice(separator + CLEANUP_LABEL_SEPARATOR.length);
}

export type StageTranslator = (key: string, params?: Record<string, string>) => string;

export interface StageCount {
  done: number;
  total: number;
}

export interface StageCounterMetadata {
  stage?: string | null;
  framesDone?: number | null;
  framesTotal?: number | null;
}

// El backend reusa framesDone/framesTotal para las areas de relleno y las caras
// de la restauracion; solo valen para la etapa que los reporto.
export function activeStageCount(stageKey: string, metadata: StageCounterMetadata | undefined): StageCount | null {
  if (!metadata || metadata.stage !== stageKey) {
    return null;
  }
  const { framesDone: done, framesTotal: total } = metadata;
  if (typeof done !== "number" || typeof total !== "number" || total <= 0 || done > total) {
    return null;
  }
  return { done, total };
}

// `translate` devuelve la clave tal cual cuando no existe en el catalogo.
function translateOrNull(t: StageTranslator, key: string, params?: Record<string, string>): string | null {
  const translated = t(key, params);
  return translated === key ? null : translated;
}

function countedStageLabel(stageKey: string, count: StageCount, t: StageTranslator): string | null {
  const params = { done: String(count.done), total: String(count.total) };
  return translateOrNull(t, `${stageTranslationKey(stageKey)}.count`, params);
}

export function translateStageLabel(stage: LabelledStage, t: StageTranslator, count?: StageCount | null): string {
  if (stage.key.startsWith(CLEANUP_STAGE_PREFIX)) {
    return t("job.stage.cleanup", { model: cleanupModelName(stage) });
  }
  const counted = count ? countedStageLabel(stage.key, count, t) : null;
  return counted ?? translateOrNull(t, stageTranslationKey(stage.key)) ?? stage.label;
}
