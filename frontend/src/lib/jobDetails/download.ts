import type { DownloadJob } from "../apiTypes";
import { formatModelSize } from "../sizeFormat";
import { buildTiming, pushIdentity } from "./common";
import {
  type DetailContext,
  type DetailItem,
  type JobDetailSections,
  push,
  pushNumber,
} from "./types";

// "12.4 MB / 380.1 MB" y no dos filas sueltas: lo que se pregunta es cuanto
// falta, y eso solo se lee comparando. Sin total conocido va solo lo bajado.
function transferredValue(job: DownloadJob): string | null {
  if (job.downloadedBytes <= 0 && !job.totalBytes) {
    return null;
  }
  const downloaded = formatModelSize(job.downloadedBytes);
  return job.totalBytes ? `${downloaded} / ${formatModelSize(job.totalBytes)}` : downloaded;
}

export function buildDownloadSections(
  job: DownloadJob,
  context: DetailContext,
): JobDetailSections {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "download", job.status, context);
  push(parameters, "job.detail.field.title", job.mediaTitle);
  push(parameters, "job.detail.field.uploader", job.mediaUploader);
  push(parameters, "job.detail.field.extractor", job.extractor);
  push(parameters, "job.detail.field.sourceUrl", job.url, { isLong: true });
  if (job.audioOnly) {
    push(parameters, "job.detail.field.audioOnly", job.audioFormat.toUpperCase());
    pushNumber(parameters, "job.detail.field.audioBitrate", job.audioBitrateKbps, (value) => `${value} kbps`);
  } else {
    pushNumber(parameters, "job.detail.field.maxHeight", job.maxHeight, (value) => `${value}p`);
    push(parameters, "job.detail.field.container", job.videoContainer.toUpperCase());
  }

  const result: DetailItem[] = [];
  push(result, "job.detail.field.transferred", transferredValue(job), { isNumeric: true });
  if (job.outputFiles.length > 0) {
    push(result, "job.detail.field.outputFiles", job.outputFiles.join(", "), { isLong: true });
  }
  push(result, "job.detail.field.outputDirectory", job.outputDirectory, { isLong: true });

  return { parameters, timing: buildTiming(job, context), result };
}
