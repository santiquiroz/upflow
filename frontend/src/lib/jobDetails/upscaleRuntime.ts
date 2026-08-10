import type { AnyQueuedJob } from "../jobTypeGuards";
import { readMetadataString } from "./common";
import { type DetailContext, type DetailItem, push } from "./types";

/**
 * Por que este trabajo fue rapido o lento. Medido a 1080p->4x en una RX 7800
 * XT: onnx fp16 1.58 fps, ncnn 0.72, onnx fp32 0.32 — sin esta fila, "en mi
 * maquina va lentisimo" no se puede diagnosticar.
 */
export function pushUpscaleRuntime(
  items: DetailItem[],
  job: AnyQueuedJob,
  context: DetailContext,
): void {
  const backend = readMetadataString(job, "upscaleBackend");
  if (!backend) {
    return;
  }
  const precision = readMetadataString(job, "upscalePrecision");
  const base = precision ? `${backend} ${precision}` : backend;
  const tiled = "metadata" in job && job.metadata?.upscaleTiled === true;
  push(
    items,
    "job.detail.field.upscaleRuntime",
    tiled ? `${base} · ${context.t("job.detail.tiled")}` : base,
  );
}
