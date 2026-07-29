import type {
  CheckpointCandidate,
  PreflightResponse,
  Precision,
} from "../../lib/apiTypes";

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
  message: string;
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
        message: "No se pudo evaluar este modelo. Podés instalarlo igual.",
      },
    ];
  }

  const warnings: Warning[] = [];

  if (preflight.compat === "gated") {
    warnings.push({
      code: "gated",
      message: "Repo con acceso restringido: necesitás un token de Hugging Face y aceptar la licencia.",
    });
  }
  if (preflight.compat === "incompatible") {
    warnings.push({
      code: "incompatible",
      message: preflight.compatReason ?? "No parece un pipeline diffusers.",
    });
  }

  const cost = preflight.precisions.find((p) => p.precision === precision);

  if (cost && preflight.disk && preflight.disk.freeBytes < cost.downloadBytes) {
    warnings.push({
      code: "disk_low",
      message:
        `Quedan ${formatGb(preflight.disk.freeBytes)} libres en ${preflight.disk.targetPath} ` +
        `y hace falta ${formatGb(cost.downloadBytes)}.`,
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
        message:
          `Convertir este checkpoint necesita ~${formatGb(peakBytes)} de pico en ` +
          `${preflight.disk.targetPath} (deja el checkpoint, el pipeline y el ONNX ` +
          `en disco a la vez) y quedan ${formatGb(preflight.disk.freeBytes)} libres.`,
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
      message:
        `La conversión carga el checkpoint completo en RAM: ${formatGb(checkpoint.sizeBytes)} ` +
        `requeridos y ${formatGb(preflight.freeRamBytes)} libres.`,
    });
  }

  if (cost) {
    const tooSmall = preflight.devices.filter(
      (d) => d.kind === "gpu" && d.freeVramBytes !== null && d.freeVramBytes < cost.estimatedPeakBytes,
    );
    for (const device of tooSmall) {
      warnings.push({
        code: "device_wont_fit",
        message:
          `${device.name}: no entra. Necesita ~${formatGb(cost.estimatedPeakBytes)} estimados ` +
          `a ${preflight.referenceWidth}×${preflight.referenceHeight} y tiene ` +
          `${formatGb(device.freeVramBytes as number)} libres.`,
      });
    }
  }

  if (preflight.devices.every((d) => d.kind !== "gpu")) {
    warnings.push({
      code: "cpu_slow",
      message: "Sin GPU compatible: generar en CPU tarda varios minutos por imagen.",
    });
  } else if (preflight.devices.some((d) => d.kind === "cpu")) {
    warnings.push({
      code: "cpu_slow",
      message: "En CPU tarda varios minutos por imagen.",
    });
  }

  return warnings;
}
