import type { JobResponse } from "../apiTypes";
import { buildTiming, pushDevice, pushIdentity } from "./common";
import { type DetailContext, type DetailItem, type JobDetailSections, push, pushNumber } from "./types";
import { pushUpscaleRuntime } from "./upscaleRuntime";

export function buildImageSections(job: JobResponse, context: DetailContext): JobDetailSections {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "image", job.status, context);
  push(parameters, "job.detail.field.file", job.originalFilename);
  push(parameters, "job.detail.field.model", job.modelName);
  pushNumber(parameters, "job.detail.field.scale", job.scale, (value) => `${value}x`);
  push(parameters, "job.detail.field.format", job.outputFormat.toUpperCase());
  pushDevice(parameters, job.device, context);

  const result: DetailItem[] = [];
  pushUpscaleRuntime(result, job, context);

  return { parameters, timing: buildTiming(job, context), result };
}
