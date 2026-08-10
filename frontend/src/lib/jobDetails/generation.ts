import type { GenerationJob } from "../apiTypes";
import { buildTiming, pushDevice, pushIdentity } from "./common";
import {
  type DetailContext,
  type DetailItem,
  type JobDetailSections,
  push,
  pushFlag,
  pushNumber,
} from "./types";

// Ritmo real del trabajo terminado: permite comparar corridas y maquinas
// ("~2 s/paso aca, ~90 s/paso en la maquina que cayo a CPU").
function stepPace(job: GenerationJob): string | null {
  if (!job.startedAt || !job.finishedAt || job.steps <= 0) {
    return null;
  }
  const elapsedSeconds = (new Date(job.finishedAt).getTime() - new Date(job.startedAt).getTime()) / 1000;
  if (!Number.isFinite(elapsedSeconds) || elapsedSeconds <= 0) {
    return null;
  }
  const pace = elapsedSeconds / job.steps;
  return pace >= 10 ? `~${Math.round(pace)} s/step` : `~${pace.toFixed(1)} s/step`;
}

// Una semilla aleatoria y una elegida se ven igual: sin la marca, repetir un
// resultado parece posible cuando no lo es.
function seedValue(job: GenerationJob, context: DetailContext): string | null {
  if (job.seed === null || job.seed === undefined) {
    return null;
  }
  return job.seedWasRandom
    ? context.t("job.detail.seed.random", { seed: String(job.seed) })
    : String(job.seed);
}

export function buildGenerationSections(
  job: GenerationJob,
  context: DetailContext,
): JobDetailSections {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "generation", job.status, context);
  push(parameters, "job.detail.field.prompt", job.prompt, { isLong: true });
  push(parameters, "job.detail.negativePrompt", job.negativePrompt, { isLong: true });
  push(parameters, "job.detail.field.model", job.modelId);
  pushNumber(parameters, "job.detail.field.steps", job.steps);
  pushNumber(parameters, "job.detail.field.guidance", job.guidance);
  pushNumber(parameters, "job.detail.field.strength", job.strength);
  push(parameters, "job.detail.field.size", `${job.width}x${job.height}`, { isNumeric: true });
  push(parameters, "job.detail.field.seed", seedValue(job, context), { isNumeric: true });
  push(parameters, "job.detail.field.scheduler", job.scheduler);
  push(parameters, "job.detail.field.precision", job.precision);
  push(parameters, "job.detail.field.executionProvider", job.executionProvider);
  push(parameters, "job.detail.field.speedClass", job.speedClass);
  pushFlag(parameters, "job.detail.field.autoUpscale", job.autoUpscale, context.t);
  pushDevice(parameters, job.device, context);

  const result: DetailItem[] = [];
  push(result, "job.detail.field.pace", stepPace(job), { isNumeric: true });
  // El job termina COMPLETO aunque el agrandado falle: sin esta fila el usuario
  // recibe una imagen mas chica de la que pidio y nada se lo dice.
  push(result, "job.detail.field.upscaleError", job.upscaleError, { isLong: true });

  return { parameters, timing: buildTiming(job, context), result };
}
