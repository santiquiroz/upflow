import type { GenerationJob } from "../apiTypes";
import {
  type AnyQueuedJob,
  isAudioJob,
  isDownloadJob,
  isShape3dJob,
  isTranscribeJob,
  isVideoJob,
} from "../jobTypeGuards";
import { buildAudioSections } from "./audio";
import { buildDownloadSections } from "./download";
import { buildGenerationSections } from "./generation";
import { buildImageSections } from "./image";
import { buildShape3dSections } from "./shape3d";
import { buildTranscribeSections } from "./transcribe";
import { buildVideoSections } from "./video";
import { type DetailContext, type JobDetailSections, emptySections } from "./types";

export type { CatalogName, DetailContext, DetailItem, JobDetailSections } from "./types";

// Un solo punto de entrada por familia: la pantalla no decide que campos van ni
// tiene que saber que forma tiene cada respuesta. El ORDEN de los guards
// importa: un job 3D tambien trae `prompt`, asi que se descarta antes que el de
// generacion.
export function buildJobDetailSections(
  job: AnyQueuedJob | undefined,
  context: DetailContext,
): JobDetailSections {
  if (!job) {
    return emptySections();
  }
  if (isShape3dJob(job)) {
    return buildShape3dSections(job, context);
  }
  if (isDownloadJob(job)) {
    return buildDownloadSections(job, context);
  }
  if (isTranscribeJob(job)) {
    return buildTranscribeSections(job, context);
  }
  if (isVideoJob(job)) {
    return buildVideoSections(job, context);
  }
  if (isAudioJob(job)) {
    return buildAudioSections(job, context);
  }
  if ("prompt" in job) {
    return buildGenerationSections(job as GenerationJob, context);
  }
  return buildImageSections(job, context);
}
