import { apiGet, apiPostForm, apiPostJson } from "../lib/api";
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

export interface PartParam {
  name: string;
  labelKey: string;
  default: number;
  minimum: number;
}

export interface PartKind {
  id: string;
  labelKey: string;
  descriptionKey: string;
  params: PartParam[];
}

export interface GeneratedPart {
  canPrint: boolean;
  sizeMm: [number, number, number];
  volumeMm3: number | null;
  triangleCount: number;
  overhangRatio: number;
  blockers: string[];
  advice: string[];
  /** Se entrega siempre: si no entra en esa cama, la pieza igual está bien hecha. */
  downloadUrl: string;
}

export function fetchPartKinds(): Promise<{ kinds: PartKind[] }> {
  return apiGet<{ kinds: PartKind[] }>("/print/parts");
}

export function generatePart(body: {
  kind: string;
  params: Record<string, number>;
  printer: string;
}): Promise<GeneratedPart> {
  return apiPostJson<GeneratedPart>("/print/parts", body);
}

export type Shape3dStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface Shape3dJob {
  id: string;
  status: Shape3dStatus;
  prompt: string;
  printer: string;
  /**
   * "mesh" = forma sin cotas. "photo" = interpretación de una foto, tampoco
   * con cotas. "cad" = código OpenSCAD con cotas.
   */
  source: string;
  /** Solo en "cad": el código, que es la pieza editable. */
  code: string | null;
  retries: number;
  /** A cuánto se escaló el lado más largo. */
  targetMm: number | null;
  /**
   * De dónde salió esa medida: "user", "estimate" (la sugirió el modelo y el
   * usuario la confirmó) o "default" (la puso el programa). El número solo
   * dejaría creer que alguien lo eligió.
   */
  targetMmSource: string | null;
  /** Contra qué objeto la comparó el modelo, si vino de una estimación. */
  targetMmReference: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  canPrint: boolean | null;
  sizeMm: [number, number, number] | null;
  triangleCount: number | null;
  blockers: string[];
  advice: string[];
  error: string | null;
  downloadUrl: string | null;
}

export function createShape3dJob(body: {
  /** Vacío solo en "photo": ahí la entrada es la foto. */
  prompt?: string;
  printer: string;
  source?: "mesh" | "photo" | "cad";
  targetMm?: number;
  /**
   * De dónde salió `targetMm`. Lo declara el cliente porque es el único que
   * sabe si el usuario escribió la medida o aceptó la sugerida.
   */
  targetMmSource?: "user" | "estimate";
  targetMmReference?: string;
  expectedSize?: [number, number, number];
  /** Solo en "photo": token devuelto por /generation/init-image. */
  imageToken?: string;
}): Promise<Shape3dJob> {
  return apiPostJson<Shape3dJob>("/print/generate", {
    prompt: body.prompt ?? "",
    printer: body.printer,
    source: body.source ?? "mesh",
    image_token: body.imageToken ?? null,
    target_mm: body.targetMm ?? null,
    target_mm_source: body.targetMmSource ?? null,
    target_mm_reference: body.targetMmReference ?? null,
    expected_size: body.expectedSize ?? null,
  });
}

/** Cuánto mide de verdad el objeto descrito. Es una SUGERENCIA, no una cota. */
export interface SizeEstimate {
  longestMm: number;
  /** Contra qué objeto lo comparó el modelo. Puede venir vacío. */
  reference: string;
}

export function estimatePrintSize(prompt: string): Promise<SizeEstimate> {
  return apiPostJson<SizeEstimate>("/print/estimate-size", { prompt });
}

export function getShape3dJob(jobId: string): Promise<Shape3dJob> {
  return apiGet<Shape3dJob>(`/print/generate/${jobId}`);
}

export function cancelShape3dJob(jobId: string): Promise<Shape3dJob> {
  return apiPostJson<Shape3dJob>(`/print/generate/${jobId}/cancel`, {});
}
