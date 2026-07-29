import type {
  CheckpointCandidate,
  PreflightResponse,
  Precision,
} from "../../lib/apiTypes";
import type { TranslationParams } from "../../i18n";

export type WarningCode =
  | "degraded"
  | "gated"
  | "incompatible"
  | "disk_low"
  | "ram_low"
  | "device_wont_fit"
  | "cpu_slow";

export interface Warning {
  code: WarningCode;
  key: string;
  params?: TranslationParams;
}

function formatGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

// Medido de punta a punta el 2026-07-29: un checkpoint de 2034 MB dejo un pico
// de 8140 MB en disco. Se conserva como constante nombrada para que revisarlo
// con otra medicion sea un cambio de una linea.
export const SINGLE_FILE_DISK_PEAK_FACTOR = 4;

// El servidor manda hechos medidos; la decision de que amerita un aviso vive
// aca, para que cambiar de precision no re-consulte el backend. `null` en una
// medicion significa "no se pudo medir": nunca genera aviso.
export function buildWarnings(
  preflight: PreflightResponse,
  precision: Precision,
  checkpoint?: CheckpointCandidate,
): Warning[] {
  if (preflight.degraded) {
    return [
      {
        code: "degraded",
        key: "generation.warning.degraded",
      },
    ];
  }

  const warnings: Warning[] = [];

  if (preflight.compat === "gated") {
    warnings.push({
      code: "gated",
      key: "generation.warning.gated",
    });
  }
  if (preflight.compat === "incompatible") {
    warnings.push(
      preflight.compatReason
        ? {
            code: "incompatible",
            key: "generation.warning.incompatible.reason",
            params: { reason: preflight.compatReason },
          }
        : {
            code: "incompatible",
            key: "generation.warning.incompatible.fallback",
          },
    );
  }

  const cost = preflight.precisions.find((p) => p.precision === precision);

  if (cost && preflight.disk && preflight.disk.freeBytes < cost.downloadBytes) {
    warnings.push({
      code: "disk_low",
      key: "generation.warning.diskLow",
      params: {
        free: formatGb(preflight.disk.freeBytes),
        path: preflight.disk.targetPath,
        needed: formatGb(cost.downloadBytes),
      },
    });
  }

  // Un checkpoint suelto no pasa por `precisions`, asi que la rama de arriba
  // nunca lo cubria: quedaba sin ningun aviso de disco. Y el pico no es el
  // tamano de la descarga -- convertirlo deja tres copias en disco a la vez.
  // MEDIDO de punta a punta el 2026-07-29 sobre un SD1.5 real: 2034 MB de
  // checkpoint dejaron un pico de 8140 MB (4.0x) = el checkpoint + 4069 MB de
  // arbol diffusers + 2037 MB de ONNX fp16. Exportar en fp32 lo sube mas.
  if (checkpoint && preflight.disk) {
    const peakBytes = checkpoint.sizeBytes * SINGLE_FILE_DISK_PEAK_FACTOR;
    if (preflight.disk.freeBytes < peakBytes) {
      warnings.push({
        code: "disk_low",
        key: "generation.warning.diskLow.singleFile",
        params: {
          peak: formatGb(peakBytes),
          path: preflight.disk.targetPath,
          free: formatGb(preflight.disk.freeBytes),
        },
      });
    }
  }

  if (
    checkpoint &&
    preflight.freeRamBytes !== null &&
    preflight.freeRamBytes < checkpoint.sizeBytes
  ) {
    warnings.push({
      code: "ram_low",
      key: "generation.warning.ramLow",
      params: {
        needed: formatGb(checkpoint.sizeBytes),
        free: formatGb(preflight.freeRamBytes),
      },
    });
  }

  if (cost) {
    const tooSmall = preflight.devices.filter(
      (d) => d.kind === "gpu" && d.freeVramBytes !== null && d.freeVramBytes < cost.estimatedPeakBytes,
    );
    for (const device of tooSmall) {
      warnings.push({
        code: "device_wont_fit",
        key: "generation.warning.deviceWontFit",
        params: {
          device: device.name,
          needed: formatGb(cost.estimatedPeakBytes),
          width: preflight.referenceWidth,
          height: preflight.referenceHeight,
          free: formatGb(device.freeVramBytes as number),
        },
      });
    }
  }

  if (preflight.devices.every((d) => d.kind !== "gpu")) {
    warnings.push({
      code: "cpu_slow",
      key: "generation.warning.cpuOnly",
    });
  } else if (preflight.devices.some((d) => d.kind === "cpu")) {
    warnings.push({
      code: "cpu_slow",
      key: "generation.warning.cpuSlow",
    });
  }

  return warnings;
}
