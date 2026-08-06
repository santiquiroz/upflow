/**
 * Los Whisper que terminan en `.en` entienden UN idioma: inglés.
 *
 * Reportado el 2026-08-06 desde una instalación real: con un audio en castellano
 * y `whisper-tiny.en_timestamped` elegido, la transcripción moría con un error
 * crudo de la librería. La pantalla te dejaba armar esa combinación.
 *
 * El sufijo puede llevar cola (`whisper-tiny.en_timestamped`), así que se busca
 * `.en` seguido de algo que no sea letra, no al final del nombre.
 */
const ENGLISH_ONLY = /\.en(?![a-z])/i;

export function isEnglishOnly(modelId: string): boolean {
  const nombre = modelId.split("/").pop() ?? modelId;
  return ENGLISH_ONLY.test(nombre);
}
