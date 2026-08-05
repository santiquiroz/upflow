import { apiGet, apiPostForm } from "../lib/api";
import type { UploadOptions } from "../lib/uploadRequest";

export interface Printer {
  id: string;
  bedMm: [number, number, number];
}

export interface PrintCheckResult {
  canPrint: boolean;
  sizeMm: [number, number, number];
  triangleCount: number;
  watertight: boolean;
  manifold: boolean;
  volumeMm3: number | null;
  overhangRatio: number;
  /** Lo que impide imprimir. Vacío = se puede. */
  blockers: string[];
  /** Lo que se imprime igual pero saldría mejor de otra forma. */
  advice: string[];
}

export interface CheckPrintParams {
  file: File;
  printer: string;
  targetAxis?: string;
  targetMm?: number;
}

export function fetchPrinters(): Promise<{ printers: Printer[] }> {
  return apiGet<{ printers: Printer[] }>("/print/printers");
}

export function checkPrint(
  params: CheckPrintParams,
  options: UploadOptions = {},
): Promise<PrintCheckResult> {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("printer", params.printer);
  // Los dos van juntos o ninguno: un eje sin medida no dice a cuánto escalar.
  if (params.targetAxis && typeof params.targetMm === "number") {
    formData.append("target_axis", params.targetAxis);
    formData.append("target_mm", String(params.targetMm));
  }
  return apiPostForm<PrintCheckResult>("/print/check", formData, options);
}

export interface MeshRepairResult {
  canPrint: boolean;
  watertight: boolean;
  manifold: boolean;
  triangleCount: number;
  volumeMm3: number | null;
  blockers: string[];
  /** La malla reparada. Se entrega aunque NO haya quedado cerrada. */
  downloadUrl: string;
}

export function repairMesh(
  file: File,
  options: UploadOptions = {},
): Promise<MeshRepairResult> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostForm<MeshRepairResult>("/print/repair", formData, options);
}
