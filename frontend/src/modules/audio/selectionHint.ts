import type { Locale } from "../../i18n";

export interface AudioSectionAvailability {
  masteringAvailable: boolean;
  restoreAvailable: boolean;
  /** Hay al menos un modelo de limpieza instalado. */
  cleanupAvailable?: boolean;
}

// Denoise y Voz estan siempre en pantalla; Limpieza, Acabado y Restore aparecen
// solo si el backend reporta pasos, presets o modelos instalados. El mensaje que
// explica por que el boton esta apagado se arma desde ESTA misma disponibilidad
// para que no pueda nombrar una seccion que el usuario no ve, ni callarse una
// que si esta. El orden es el mismo en que se renderizan las secciones.
export function selectableSectionKeys({
  masteringAvailable,
  restoreAvailable,
  cleanupAvailable = false,
}: AudioSectionAvailability): readonly string[] {
  return [
    ...(cleanupAvailable ? ["audio.section.cleanup"] : []),
    "audio.section.denoise",
    ...(masteringAvailable ? ["audio.section.mastering"] : []),
    ...(restoreAvailable ? ["audio.section.restore"] : []),
    "voice.sectionTitle",
  ];
}

export function joinAsChoices(labels: readonly string[], locale: Locale): string {
  return new Intl.ListFormat(locale, { style: "long", type: "disjunction" }).format(labels);
}

// Espejo de UNAMBIGUOUS_SOURCE_FORMATS del backend (app/services/audio_conversion.py):
// extensiones cuyo contenedor determina el codec sin ambiguedad. `.m4a` no esta
// porque puede traer ALAC o AAC, asi que "m4a -> m4a" NO se considera un
// trabajo vacio ni aca ni alla.
const UNAMBIGUOUS_SOURCE_FORMATS: Readonly<Record<string, string>> = {
  wav: "wav",
  wave: "wav",
  flac: "flac",
  mp3: "mp3",
};

// Extensiones de las que se puede AFIRMAR que el origen no tiene perdida. `.m4a`
// queda afuera por lo mismo de arriba: no se avisa de una perdida que no se
// puede probar.
const LOSSLESS_SOURCE_EXTENSIONS: ReadonlySet<string> = new Set([
  "wav",
  "wave",
  "flac",
  "aif",
  "aiff",
]);

const LOSSY_OUTPUT_FORMATS: ReadonlySet<string> = new Set(["mp3", "m4a"]);

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  return dot === -1 ? "" : filename.slice(dot + 1).toLowerCase();
}

/** El formato de salida equivalente al del archivo, o null si no se puede afirmar. */
export function audioSourceFormat(filename: string): string | null {
  return UNAMBIGUOUS_SOURCE_FORMATS[extensionOf(filename)] ?? null;
}

/** Cuantos de los archivos elegidos cambiarian de formato con esta salida. */
export function convertibleFileCount(
  files: readonly { name: string }[],
  outputFormat: string,
): number {
  return files.filter((file) => audioSourceFormat(file.name) !== outputFormat).length;
}

/**
 * True cuando se puede afirmar que la conversion descarta informacion para
 * siempre: origen sin perdida y destino con perdida. No bloquea nada — es el
 * dato que la UI necesita para decirlo antes de que se apriete el boton.
 */
export function losesQualityIrreversibly(
  files: readonly { name: string }[],
  outputFormat: string,
): boolean {
  if (!LOSSY_OUTPUT_FORMATS.has(outputFormat)) {
    return false;
  }
  return files.some((file) => LOSSLESS_SOURCE_EXTENSIONS.has(extensionOf(file.name)));
}

export function isLossyFormat(outputFormat: string): boolean {
  return LOSSY_OUTPUT_FORMATS.has(outputFormat);
}
