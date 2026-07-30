import type { TranslationParams } from "../../i18n";
import type { UpscalerPreflightResponse } from "../../lib/apiTypes";

export type UpscalerWarningCode =
  | "degraded"
  | "gated"
  | "incompatible"
  | "disk_low";

export interface UpscalerWarning {
  code: UpscalerWarningCode;
  key: string;
  params: TranslationParams;
}

function formatGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

// Convertir deja el archivo descargado Y el .onnx resultante en disco a la vez,
// asi que hace falta el doble. Un repo que ya trae .onnx no convierte nada: se
// baja a un nombre de staging y se renombra, que es UNA copia. Aplicar el doble
// en ese caso avisaria de un espacio que nunca se usa.
export const UPSCALER_CONVERSION_DISK_FACTOR = 2;
export const UPSCALER_DIRECT_DISK_FACTOR = 1;

function diskFactor(compat: string | null): number {
  return compat === "needs_conversion"
    ? UPSCALER_CONVERSION_DISK_FACTOR
    : UPSCALER_DIRECT_DISK_FACTOR;
}

export function buildUpscalerWarnings(
  preflight: UpscalerPreflightResponse,
): UpscalerWarning[] {
  if (preflight.degraded) {
    return [
      {
        code: "degraded",
        key: "upscaler.warning.degraded",
        params: {},
      },
    ];
  }

  const warnings: UpscalerWarning[] = [];

  if (preflight.compat === "gated") {
    warnings.push({
      code: "gated",
      key: "upscaler.warning.gated",
      params: {},
    });
  }

  if (preflight.compat === "incompatible") {
    warnings.push({
      code: "incompatible",
      key:
        preflight.compatReasonKey ??
        "upscaler.warning.incompatibleFallback",
      params: preflight.compatReasonParams,
    });
  }

  if (preflight.disk !== null && preflight.downloadBytes !== null) {
    const neededBytes = preflight.downloadBytes * diskFactor(preflight.compat);
    if (preflight.disk.freeBytes < neededBytes) {
      warnings.push({
        code: "disk_low",
        key: "upscaler.warning.diskLow",
        params: {
          free: formatGb(preflight.disk.freeBytes),
          path: preflight.disk.targetPath,
          needed: formatGb(neededBytes),
        },
      });
    }
  }

  return warnings;
}
