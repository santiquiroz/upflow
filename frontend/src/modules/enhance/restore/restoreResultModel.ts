import type { JobResponse } from "../../../lib/apiTypes";
import { restoreArtifactUrl } from "../../../services/restore";
import { readRestoredFaces, type RestoredFace } from "./faceResultModel";

// Lectura de job.metadata.restore (restore_outputs.restore_summary). Llega como
// JSON sin tipar, asi que cada campo se valida antes de usarlo.

export interface RestoreDownloadNames {
  restored: string;
  uncolored: string;
  beforeAfter: string;
  sidecar: string;
}

export interface RestoreResultSummary {
  artifacts: string[];
  downloadNames: RestoreDownloadNames;
  viewFullResolution: boolean;
  compositeReasons: string[];
  faces: RestoredFace[];
  recomposeAvailable: boolean;
}

export interface EditorSource {
  url: string;
  fileName: string;
}

// Lo que cuenta la tarjeta: caras, color y rellenos. Un SR generativo solo no.
const INFO_CARD_REASONS = new Set(["faces", "colorize", "largeFill", "fillOverFace"]);
const BROWSER_FORMATS = new Set(["png", "jpg", "jpeg", "webp"]);
const VIEW_EXTENSION = ".jpg";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function stringField(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function readDownloadNames(value: unknown): RestoreDownloadNames | null {
  if (!isRecord(value)) {
    return null;
  }
  const restored = stringField(value, "restored");
  const uncolored = stringField(value, "uncolored");
  const beforeAfter = stringField(value, "before_after");
  const sidecar = stringField(value, "sidecar");
  if (restored === null || uncolored === null || beforeAfter === null || sidecar === null) {
    return null;
  }
  return { restored, uncolored, beforeAfter, sidecar };
}

export function readRestoreSummary(job: JobResponse): RestoreResultSummary | null {
  if (job.status !== "completed") {
    return null;
  }
  // Respuesta sin validar del backend: un job viejo o de otra familia puede no traer metadata.
  const restore: unknown = job.metadata?.restore;
  if (!isRecord(restore)) {
    return null;
  }
  const artifacts = stringList(restore.artifacts);
  const downloadNames = readDownloadNames(restore.downloadNames);
  // Una corrida de "Preview this area" solo deja la vista previa del recorte.
  if (!artifacts.includes("view") || downloadNames === null) {
    return null;
  }
  return {
    artifacts,
    downloadNames,
    viewFullResolution: restore.viewFullResolution === true,
    compositeReasons: stringList(restore.compositeReasons),
    faces: readRestoredFaces(restore.faces),
    recomposeAvailable: restore.recomposeAvailable === true,
  };
}

// Recomponer puede apagar todas las caras: la tarjeta y la grilla siguen al sidecar nuevo.
export function withRecomposedSidecar(
  summary: RestoreResultSummary,
  sidecar: Record<string, unknown>,
): RestoreResultSummary {
  return {
    ...summary,
    faces: readRestoredFaces(sidecar.faces),
    compositeReasons: stringList(sidecar.compositeReasons),
  };
}

export function hasUncolored(summary: RestoreResultSummary): boolean {
  return summary.artifacts.includes("uncolored");
}

export function showsInfoCard(summary: RestoreResultSummary): boolean {
  return summary.compositeReasons.some((reason) => INFO_CARD_REASONS.has(reason));
}

function withExtension(fileName: string, extension: string): string {
  const dot = fileName.lastIndexOf(".");
  return `${dot > 0 ? fileName.slice(0, dot) : fileName}${extension}`;
}

// El Editor dibuja en un canvas: un TIFF o un 16 bits no se ven, la copia "view" si.
export function editorSource(job: JobResponse, summary: RestoreResultSummary): EditorSource {
  if (job.downloadUrl && BROWSER_FORMATS.has(job.outputFormat.toLowerCase())) {
    return { url: job.downloadUrl, fileName: summary.downloadNames.restored };
  }
  return {
    url: restoreArtifactUrl(job.jobId, "view"),
    fileName: withExtension(summary.downloadNames.restored, VIEW_EXTENSION),
  };
}
