import type { JobResponse } from "../apiTypes";
import { type DetailContext, type DetailItem, push } from "./types";

// Lectura de job.metadata.restore (photo_restore_pipeline.restore_metadata +
// restore_outputs.restore_summary). Llega como JSON sin tipar: cada campo se
// valida y lo que no tiene forma se descarta en vez de romper el modal.

interface ModelUse {
  id: string;
  device: string | null;
  precision: string | null;
}

interface StepRun {
  id: string;
  models: ModelUse[];
}

interface UpscaleRun {
  mode: string;
  scale: number | null;
  model: string | null;
  generative: boolean;
}

interface CpuFallbackNote {
  model: string;
  reason: string;
}

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function records(value: unknown): UnknownRecord[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function text(record: UnknownRecord, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value !== "" ? value : null;
}

function readModelUse(value: unknown): ModelUse | null {
  if (!isRecord(value)) {
    return null;
  }
  const id = text(value, "id");
  return id === null ? null : { id, device: text(value, "device"), precision: text(value, "precision") };
}

function readStepRun(record: UnknownRecord): StepRun | null {
  const id = text(record, "id");
  if (id === null) {
    return null;
  }
  const candidates = [record.model, ...records(record.auxiliaryModels)];
  const models = candidates.map(readModelUse).filter((model): model is ModelUse => model !== null);
  return { id, models };
}

function readRestore(job: JobResponse): UnknownRecord | null {
  const restore: unknown = job.metadata?.restore;
  return isRecord(restore) ? restore : null;
}

function readStepRuns(restore: UnknownRecord): StepRun[] {
  return records(restore.steps)
    .map(readStepRun)
    .filter((step): step is StepRun => step !== null);
}

function readFaceCounts(restore: UnknownRecord): { restored: number; total: number } | null {
  const faces = records(restore.faces);
  if (faces.length === 0) {
    return null;
  }
  return { restored: faces.filter((face) => face.restored === true).length, total: faces.length };
}

function readUpscaleRun(restore: UnknownRecord): UpscaleRun | null {
  const upscale = restore.upscale;
  if (!isRecord(upscale)) {
    return null;
  }
  const mode = text(upscale, "mode");
  if (mode === null || mode === "none") {
    return null;
  }
  const scale = typeof upscale.scale === "number" ? upscale.scale : null;
  return { mode, scale, model: text(upscale, "model"), generative: upscale.generative === true };
}

function readCpuFallbacks(restore: UnknownRecord): CpuFallbackNote[] {
  return records(restore.cpuFallback).flatMap((entry) => {
    const model = text(entry, "model");
    return model === null ? [] : [{ model, reason: text(entry, "reason") ?? "" }];
  });
}

function formatModelUse(model: ModelUse, context: DetailContext): string {
  const device = model.device ? context.deviceLabel(model.device) : null;
  return [model.id, device, model.precision].filter((part): part is string => part !== null).join(" · ");
}

function formatStepRun(step: StepRun, context: DetailContext): string {
  if (step.models.length === 0) {
    return context.t("job.detail.restore.noModel");
  }
  return step.models.map((model) => formatModelUse(model, context)).join("; ");
}

function upscaleMethod(upscale: UpscaleRun, context: DetailContext): string {
  if (upscale.mode === "classic") {
    return context.t("job.detail.restore.classicUpscale");
  }
  return upscale.model ?? upscale.mode;
}

function formatUpscaleRun(upscale: UpscaleRun, context: DetailContext): string {
  const scale = upscale.scale === null ? null : `${upscale.scale}x`;
  const tag = context.t(upscale.generative ? "model.tag.generative" : "model.tag.nonGenerative");
  return [upscaleMethod(upscale, context), scale, tag].filter((part): part is string => part !== null).join(" · ");
}

function formatCpuFallback(note: CpuFallbackNote): string {
  return note.reason ? `${note.model}: ${note.reason}` : note.model;
}

export function pushRestoreSteps(items: DetailItem[], job: JobResponse, context: DetailContext): void {
  const steps = job.restoreSteps ?? [];
  const labels = steps.map((step) => context.t(`restore.step.${step}`));
  push(items, "job.detail.field.restoreSteps", labels.join(", "));
}

function pushStepRuns(items: DetailItem[], restore: UnknownRecord, context: DetailContext): void {
  for (const step of readStepRuns(restore)) {
    push(items, `restore.step.${step.id}`, formatStepRun(step, context));
  }
}

function pushFaceCount(items: DetailItem[], restore: UnknownRecord, context: DetailContext): void {
  const faces = readFaceCounts(restore);
  if (faces === null) {
    return;
  }
  const params = { restored: String(faces.restored), total: String(faces.total) };
  push(items, "job.detail.field.restoreFaces", context.t("job.detail.restore.facesRestored", params));
}

export function pushRestoreResult(items: DetailItem[], job: JobResponse, context: DetailContext): void {
  const restore = readRestore(job);
  if (restore === null) {
    return;
  }
  pushStepRuns(items, restore, context);
  pushFaceCount(items, restore, context);
  const upscale = readUpscaleRun(restore);
  push(items, "job.detail.field.restoreUpscale", upscale ? formatUpscaleRun(upscale, context) : null);
  const fallbacks = readCpuFallbacks(restore).map(formatCpuFallback).join("; ");
  push(items, "job.detail.field.restoreCpuFallback", fallbacks, { isLong: true });
}
