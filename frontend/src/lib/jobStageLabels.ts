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

export function translateStageLabel(stage: LabelledStage, t: StageTranslator): string {
  if (stage.key.startsWith(CLEANUP_STAGE_PREFIX)) {
    return t("job.stage.cleanup", { model: cleanupModelName(stage) });
  }
  const key = stageTranslationKey(stage.key);
  const translated = t(key);
  // `translate` devuelve la clave tal cual cuando no existe en el catalogo.
  return translated === key ? stage.label : translated;
}
