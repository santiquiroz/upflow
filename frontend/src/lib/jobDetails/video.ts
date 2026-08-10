import type { VideoJobResponse } from "../apiTypes";
import { formatFps } from "../formatFps";
import { buildTiming, pushDevice, pushIdentity, readMetadataString } from "./common";
import {
  type DetailContext,
  type DetailItem,
  type JobDetailSections,
  push,
  pushFlag,
  pushNumber,
} from "./types";
import { pushUpscaleRuntime } from "./upscaleRuntime";

const AUDIO_ENHANCE_LABELS: Record<string, string> = {
  rnnoise: "RNNoise",
  deepfilter: "DeepFilterNet",
};

const AUDIO_RESTORE_LABELS: Record<string, string> = {
  apollo: "Apollo",
  audiosr: "AudioSR",
};

// Lo que de verdad va a salir: el fps efectivo si el pipeline ya lo escribio, el
// pedido si todavia no, y el multiplicador cuando no hay ninguno de los dos.
function fpsValue(job: VideoJobResponse): string | null {
  const outputFps = job.metadata.outputFps;
  if (typeof outputFps === "string" && outputFps) {
    return formatFps(outputFps);
  }
  if (job.targetFps) {
    return formatFps(job.targetFps);
  }
  return job.fpsMultiplier > 1 ? `${job.fpsMultiplier}x` : null;
}

function pushAudio(items: DetailItem[], job: VideoJobResponse, context: DetailContext): void {
  if (!job.keepAudio) {
    push(items, "job.detail.field.audio", context.t("job.detail.audio.dropped"));
    return;
  }
  push(items, "job.detail.field.audio", context.t("job.detail.audio.kept"));
  if (job.audioEnhance) {
    push(
      items,
      "job.detail.field.audioEnhance",
      AUDIO_ENHANCE_LABELS[job.audioEnhance] ?? job.audioEnhance,
    );
  }
  if (job.audioRestore) {
    push(
      items,
      "job.detail.field.audioRestore",
      AUDIO_RESTORE_LABELS[job.audioRestore] ?? job.audioRestore,
    );
  }
  const tracks = job.audioTrackIndices;
  if (tracks && tracks.length > 0) {
    push(items, "job.detail.field.audioTracks", tracks.join(", "), { isNumeric: true });
  }
  if (job.audioOutputFormat && job.audioOutputFormat !== "auto") {
    push(items, "job.detail.field.audioCodec", job.audioOutputFormat.toUpperCase());
  }
}

function pushEncoding(items: DetailItem[], job: VideoJobResponse, context: DetailContext): void {
  push(items, "job.detail.field.container", job.outputContainer.toUpperCase());
  push(items, "job.detail.field.videoCodec", job.videoCodec);
  push(
    items,
    "job.detail.field.videoEncoder",
    job.videoEncoder ? context.t(`job.detail.encoder.${job.videoEncoder}`) : null,
  );
  push(items, "job.detail.field.videoPreset", job.videoPreset);
  pushNumber(items, "job.detail.field.crf", job.crf);
}

export function buildVideoSections(
  job: VideoJobResponse,
  context: DetailContext,
): JobDetailSections {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "video", job.status, context);
  push(parameters, "job.detail.field.file", job.originalFilename);
  push(parameters, "job.detail.field.model", job.modelName);
  pushNumber(parameters, "job.detail.field.scale", job.scale, (value) => `${value}x`);
  pushNumber(parameters, "job.detail.field.targetHeight", job.targetHeight, (value) => `${value}p`);
  pushEncoding(parameters, job, context);
  push(parameters, "job.detail.field.fps", fpsValue(job), { isNumeric: true });
  // El motor de interpolacion solo dice algo cuando de verdad se interpola.
  if (job.fpsMultiplier > 1 || job.targetFps) {
    push(parameters, "job.detail.field.interpEngine", job.interpEngine);
  }
  pushAudio(parameters, job, context);
  pushFlag(parameters, "job.detail.field.subtitles", job.keepSubtitles, context.t);
  push(
    parameters,
    "job.detail.field.backend",
    job.backend ? context.t(`job.detail.backend.${job.backend}`) : null,
  );
  pushDevice(parameters, job.device, context);

  const result: DetailItem[] = [];
  pushUpscaleRuntime(result, job, context);
  // Por que el .mp4 pedido salio .mkv, y por que el camino rapido no corrio.
  // Estaban anotados en la metadata y no llegaban a ninguna pantalla.
  push(result, "job.detail.field.containerUpgraded", readMetadataString(job, "containerUpgradedReason"), {
    isLong: true,
  });
  push(result, "job.detail.field.slowerPipeline", readMetadataString(job, "streamPipelineFallback"), {
    isLong: true,
  });

  return { parameters, timing: buildTiming(job, context), result };
}
