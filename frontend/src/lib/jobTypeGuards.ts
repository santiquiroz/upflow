import type {
  AudioJob,
  DownloadJob,
  GenerationJob,
  JobResponse,
  Shape3dJob,
  TranscribeJob,
  VideoJobResponse,
} from "./apiTypes";

export type AnyJobResponse = JobResponse | VideoJobResponse | AudioJob | GenerationJob;

/** Las 7 familias que pueden llegar a la cola y al modal de detalle. */
export type AnyQueuedJob = AnyJobResponse | TranscribeJob | DownloadJob | Shape3dJob;

// Un job 3D TAMBIEN tiene `prompt`, asi que este guard solo es seguro sobre la
// union de 4 familias que usan JobCard y los paneles. Sobre AnyQueuedJob hay
// que descartar antes el 3D — de eso se encarga jobKindOf.
export function isGenerationJob(job: AnyJobResponse): job is GenerationJob {
  return "prompt" in job;
}

export function isVideoJob(job: AnyQueuedJob): job is VideoJobResponse {
  return "videoCodec" in job;
}

export function isAudioJob(job: AnyQueuedJob): job is AudioJob {
  return "denoise" in job;
}

export function isShape3dJob(job: AnyQueuedJob): job is Shape3dJob {
  return "printer" in job;
}

export function isDownloadJob(job: AnyQueuedJob): job is DownloadJob {
  return "outputFiles" in job;
}

export function isTranscribeJob(job: AnyQueuedJob): job is TranscribeJob {
  return "text" in job;
}

export function isImageJob(job: AnyQueuedJob): job is JobResponse {
  return jobKindOf(job) === "image";
}

export type JobKind =
  | "image"
  | "video"
  | "audio"
  | "generation"
  | "transcribe"
  | "download"
  | "shape3d";

// El orden importa: cada rama descarta un campo compartido con la siguiente
// (`prompt` lo tienen generacion y 3D; `modelId` lo tienen imagen, video,
// generacion y transcripcion). "image" queda de ultimo como el caso base.
export function jobKindOf(job: AnyQueuedJob): JobKind {
  if (isShape3dJob(job)) {
    return "shape3d";
  }
  if (isDownloadJob(job)) {
    return "download";
  }
  if (isTranscribeJob(job)) {
    return "transcribe";
  }
  if (isVideoJob(job)) {
    return "video";
  }
  if (isAudioJob(job)) {
    return "audio";
  }
  if ("prompt" in job) {
    return "generation";
  }
  return "image";
}
