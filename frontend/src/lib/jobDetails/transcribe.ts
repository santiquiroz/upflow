import type { TranscribeJob } from "../apiTypes";
import { buildTiming, pushDevice, pushIdentity } from "./common";
import {
  type DetailContext,
  type DetailItem,
  type JobDetailSections,
  push,
  pushNumber,
} from "./types";

export function buildTranscribeSections(
  job: TranscribeJob,
  context: DetailContext,
): JobDetailSections {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "transcribe", job.status, context);
  push(parameters, "job.detail.field.file", job.originalFilename);
  push(parameters, "job.detail.field.model", job.modelId);
  push(
    parameters,
    "job.detail.field.language",
    job.language ?? context.t("job.detail.language.auto"),
  );
  push(
    parameters,
    "job.detail.field.outputMode",
    job.outputMode ? context.t(`job.detail.outputMode.${job.outputMode}`) : null,
  );
  push(parameters, "job.detail.field.targetLanguage", job.targetLanguage);
  pushDevice(parameters, job.device, context);

  const result: DetailItem[] = [];
  pushNumber(result, "job.detail.field.transcriptChars", job.text ? job.text.length : null);
  // Un doblaje con lineas que no entraron suena corrido: avisarlo es la unica
  // alternativa a entregarlo en silencio.
  pushNumber(result, "job.detail.field.dubOverflow", job.dubOverflowSegments);

  return { parameters, timing: buildTiming(job, context), result };
}
