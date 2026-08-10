export interface DetailItem {
  labelKey: string;
  value: string;
  /** Numeros y medidas: se alinean a la derecha con cifras tabulares. */
  isNumeric?: boolean;
  /** Prompt, error, codigo: ocupan las dos columnas y se envuelven. */
  isLong?: boolean;
}

/**
 * Tres grupos y no una lista plana: el usuario pregunta cosas distintas ("que
 * pedi", "por donde va", "que me quedo") y mezclarlas convierte el modal en un
 * volcado. `timing` se pinta dentro del bloque de progreso.
 */
export interface JobDetailSections {
  parameters: DetailItem[];
  timing: DetailItem[];
  result: DetailItem[];
}

export type CatalogName =
  | "cleanupStep"
  | "voiceStep"
  | "voiceDelivery"
  | "masteringPreset"
  | "separationModel";

export interface DetailContext {
  t: (key: string, params?: Record<string, string>) => string;
  /** Nombre real de la placa a partir de su id ("dml:0" no dice nada). */
  deviceLabel: (deviceId: string) => string;
  /** El que usa el servidor cuando el job no fijo ninguno. */
  defaultDeviceId: string | null;
  /** Id -> copia de los catalogos de audio. Cae al id cuando no hay catalogo. */
  labelFor: (catalog: CatalogName, id: string) => string;
  /** Reloj del "lleva corriendo"; lo tickea quien renderiza. */
  nowMs: number;
}

export function emptySections(): JobDetailSections {
  return { parameters: [], timing: [], result: [] };
}

/** Un valor ausente NO se muestra: una fila vacia es ruido, no informacion. */
export function push(
  items: DetailItem[],
  labelKey: string,
  value: string | null | undefined,
  extra: Omit<DetailItem, "labelKey" | "value"> = {},
): void {
  if (value === null || value === undefined || value === "") {
    return;
  }
  items.push({ labelKey, value, ...extra });
}

export function pushNumber(
  items: DetailItem[],
  labelKey: string,
  value: number | null | undefined,
  format: (value: number) => string = String,
): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return;
  }
  push(items, labelKey, format(value), { isNumeric: true });
}

export function pushFlag(
  items: DetailItem[],
  labelKey: string,
  value: boolean | null | undefined,
  t: DetailContext["t"],
): void {
  if (typeof value !== "boolean") {
    return;
  }
  push(items, labelKey, t(value ? "job.detail.yes" : "job.detail.no"));
}
