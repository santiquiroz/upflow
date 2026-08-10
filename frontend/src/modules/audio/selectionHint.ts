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
