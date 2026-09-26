import type { CctvJobSummary, VideoJobResponse } from "../apiTypes";
import { buildTiming, pushDevice, pushIdentity } from "./common";
import { type DetailContext, type DetailItem, type JobDetailSections, push } from "./types";

// Un id que el catalogo no conoce (un preset nuevo del backend) se muestra tal cual.
function translatedOr(context: DetailContext, key: string, fallback: string): string {
  const text = context.t(key);
  return text === key ? fallback : text;
}

function catalogLabel(context: DetailContext, prefix: string, id: string | null): string | null {
  return id ? translatedOr(context, `${prefix}.${id}`, id) : null;
}

function onScreenTextValue(summary: CctvJobSummary, context: DetailContext): string | null {
  if (summary.noOsd) {
    return context.t("cctv.osd.none");
  }
  return summary.osdBoxesConfirmed ? context.t("job.detail.cctv.osdConfirmed") : null;
}

function warningsValue(summary: CctvJobSummary, context: DetailContext): string {
  return summary.warnings.map((key) => translatedOr(context, key, key)).join(" ");
}

function cctvParameters(job: VideoJobResponse, summary: CctvJobSummary, context: DetailContext): DetailItem[] {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "video", job.status, context);
  push(parameters, "job.detail.field.file", job.originalFilename);
  push(parameters, "job.detail.field.cctvTask", catalogLabel(context, "cctv.task", summary.task));
  push(parameters, "job.detail.field.cctvLane", catalogLabel(context, "cctv.lane", summary.lane));
  push(parameters, "job.detail.field.videoPreset", catalogLabel(context, "cctv.preset", summary.preset));
  push(parameters, "job.detail.field.onScreenText", onScreenTextValue(summary, context));
  pushDevice(parameters, job.device, context);
  return parameters;
}

function cctvResult(summary: CctvJobSummary, context: DetailContext): DetailItem[] {
  const result: DetailItem[] = [];
  push(result, "job.detail.field.sourceSha256", summary.sourceSha256, { isLong: true });
  push(result, "job.detail.field.cctvWarnings", warningsValue(summary, context), { isLong: true });
  return result;
}

// Un job CCTV no reescala: modelo, escala y codificacion del pipeline de video no dicen nada de el.
export function buildCctvSections(
  job: VideoJobResponse,
  summary: CctvJobSummary,
  context: DetailContext,
): JobDetailSections {
  return {
    parameters: cctvParameters(job, summary, context),
    timing: buildTiming(job, context),
    result: cctvResult(summary, context),
  };
}
