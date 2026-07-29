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
