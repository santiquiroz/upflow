import type { JobStatus } from "../apiTypes";
import { formatDuration } from "../formatDuration";
import { formatElapsed } from "../jobElapsed";
import { jobKindLabelKey, jobStatusLabelKey } from "../jobStatus";
import type { AnyQueuedJob, JobKind } from "../jobTypeGuards";
import { type DetailContext, type DetailItem, push } from "./types";

interface TimestampedJob {
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
}

export function formatTimestamp(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime()) ? iso : parsed.toLocaleString();
}

export function pushIdentity(
  items: DetailItem[],
  kind: JobKind,
  status: JobStatus,
  context: DetailContext,
): void {
  push(items, "job.detail.field.type", context.t(jobKindLabelKey(kind)));
  push(items, "job.detail.field.status", context.t(jobStatusLabelKey(status)));
}

/**
 * El dispositivo EFECTIVO, no solo el que se eligio a mano. Un job sin device
 * corre igual en alguna placa; mostrar la fila solo cuando se eligio explicito
 * escondia justo el dato que explica por que un trabajo fue lento.
 */
export function pushDevice(
  items: DetailItem[],
  device: string | null | undefined,
  context: DetailContext,
): void {
  if (device) {
    push(items, "job.detail.field.device", context.deviceLabel(device));
    return;
  }
  if (!context.defaultDeviceId) {
    return;
  }
  push(
    items,
    "job.detail.field.device",
    context.t("job.detail.device.byDefault", {
      device: context.deviceLabel(context.defaultDeviceId),
    }),
  );
}

export function buildTiming(job: TimestampedJob, context: DetailContext): DetailItem[] {
  const items: DetailItem[] = [];
  push(items, "job.detail.field.createdAt", formatTimestamp(job.createdAt));
  push(items, "job.detail.field.startedAt", job.startedAt ? formatTimestamp(job.startedAt) : null);
  push(items, "job.detail.field.finishedAt", job.finishedAt ? formatTimestamp(job.finishedAt) : null);
  pushRuntime(items, job, context);
  return items;
}

// Terminado: cuanto tardo. Corriendo: cuanto lleva. Antes solo existia lo
// primero, asi que durante la espera -- que es cuando importa -- no habia nada.
function pushRuntime(items: DetailItem[], job: TimestampedJob, context: DetailContext): void {
  if (job.finishedAt) {
    push(items, "job.detail.field.duration", formatDuration(job.startedAt, job.finishedAt), {
      isNumeric: true,
    });
    return;
  }
  push(items, "job.detail.field.elapsed", formatElapsed(job.startedAt, context.nowMs), {
    isNumeric: true,
  });
}

export function readMetadataString(job: AnyQueuedJob, key: string): string | null {
  if (!("metadata" in job) || !job.metadata) {
    return null;
  }
  const value = job.metadata[key];
  return typeof value === "string" && value !== "" ? value : null;
}

export function readMetadataNumber(job: AnyQueuedJob, key: string): number | null {
  if (!("metadata" in job) || !job.metadata) {
    return null;
  }
  const value = job.metadata[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
