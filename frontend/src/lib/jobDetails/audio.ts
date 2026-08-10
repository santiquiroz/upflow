import type { AudioJob } from "../apiTypes";
import { denoiseLabel, restoreLabel } from "../audioLabels";
import {
  buildTiming,
  pushDevice,
  pushIdentity,
  readMetadataNumber,
  readMetadataString,
} from "./common";
import { type DetailContext, type DetailItem, type JobDetailSections, push } from "./types";

const CHAIN_SEPARATOR = " → ";

function pushSeparation(items: DetailItem[], job: AudioJob, context: DetailContext): void {
  if (!job.separate) {
    return;
  }
  push(
    items,
    "job.detail.field.separationModel",
    job.separationModel ? context.labelFor("separationModel", job.separationModel) : null,
  );
  const stems = job.stems ?? [];
  if (stems.length > 0) {
    push(items, "job.detail.field.stems", stems.map((stem) => context.t(stem.labelKey)).join(", "));
  }
}

// La cadena se muestra EN ORDEN y con flechas: el orden tiene causalidad (quitar
// ruido -> quitar eco -> quitar reverb) y una lista suelta la pierde.
function pushCleanupChain(items: DetailItem[], job: AudioJob, context: DetailContext): void {
  const steps = job.cleanupSteps ?? [];
  if (steps.length === 0) {
    return;
  }
  push(
    items,
    "job.detail.field.cleanupChain",
    steps.map((id) => context.labelFor("cleanupStep", id)).join(CHAIN_SEPARATOR),
  );
}

function pushVoiceChain(items: DetailItem[], job: AudioJob, context: DetailContext): void {
  const steps = job.voiceSteps ?? [];
  if (steps.length > 0) {
    push(
      items,
      "job.detail.field.voiceSteps",
      steps.map((id) => context.labelFor("voiceStep", id)).join(CHAIN_SEPARATOR),
    );
  }
  push(
    items,
    "job.detail.field.voiceDelivery",
    job.voiceDelivery ? context.labelFor("voiceDelivery", job.voiceDelivery) : null,
  );
  if (typeof job.voicePresenceDb === "number" && job.voicePresenceDb !== 0) {
    push(items, "job.detail.field.voicePresence", `${job.voicePresenceDb} dB`, { isNumeric: true });
  }
}

// De cuanto a cuanto se movio el volumen. Es la unica forma de saber si el
// acabado hizo algo y cuanto: el nombre del preset solo no lo dice.
function pushLoudnessMove(items: DetailItem[], job: AudioJob, context: DetailContext): void {
  const before = readMetadataNumber(job, "loudnessBefore");
  const target = readMetadataNumber(job, "loudnessTarget");
  if (before === null || target === null) {
    return;
  }
  push(
    items,
    "job.detail.field.loudness",
    context.t("job.detail.loudness.move", {
      before: before.toFixed(1),
      target: target.toFixed(1),
    }),
    { isNumeric: true },
  );
}

export function buildAudioSections(job: AudioJob, context: DetailContext): JobDetailSections {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "audio", job.status, context);
  push(parameters, "job.detail.field.file", job.originalFilename);
  pushSeparation(parameters, job, context);
  pushCleanupChain(parameters, job, context);
  push(parameters, "job.detail.field.denoise", job.denoise ? denoiseLabel(job.denoise) : null);
  push(parameters, "job.detail.field.restore", job.restore ? restoreLabel(job.restore) : null);
  pushVoiceChain(parameters, job, context);
  push(
    parameters,
    "job.detail.field.mastering",
    job.master ? context.labelFor("masteringPreset", job.master) : null,
  );
  push(parameters, "job.detail.field.format", job.outputFormat?.toUpperCase());
  pushDevice(parameters, job.device, context);

  const result: DetailItem[] = [];
  pushLoudnessMove(result, job, context);
  push(result, "job.detail.field.masteringSkipped", readMetadataString(job, "masteringSkipped"), {
    isLong: true,
  });
  push(
    result,
    "job.detail.field.voiceLoudnessSkipped",
    readMetadataString(job, "voiceLoudnessSkipped"),
    { isLong: true },
  );
  push(result, "job.detail.field.overprocessed", readMetadataString(job, "cleanupOverprocessed"), {
    isLong: true,
  });

  return { parameters, timing: buildTiming(job, context), result };
}
