import type { Shape3dJob } from "../apiTypes";
import { buildTiming, pushIdentity } from "./common";
import {
  type DetailContext,
  type DetailItem,
  type JobDetailSections,
  push,
  pushNumber,
} from "./types";

const KNOWN_TARGET_SOURCES = ["user", "estimate", "default"];

/**
 * La medida y DE DONDE salio van juntas o no va ninguna: "80 mm" solo deja
 * creer que alguien la eligio cuando puede ser el relleno del programa.
 */
function targetSizeValue(job: Shape3dJob, context: DetailContext): string | null {
  if (typeof job.targetMm !== "number") {
    return null;
  }
  const size = `${job.targetMm} mm`;
  const source = job.targetMmSource;
  if (!source || !KNOWN_TARGET_SOURCES.includes(source)) {
    return size;
  }
  if (source === "estimate" && job.targetMmReference) {
    return `${size} · ${context.t("job.detail.targetMm.estimate", {
      reference: job.targetMmReference,
    })}`;
  }
  return `${size} · ${context.t(`job.detail.targetMm.${source}`)}`;
}

function verdictValue(job: Shape3dJob, context: DetailContext): string | null {
  if (job.canPrint === null || job.canPrint === undefined) {
    return null;
  }
  return context.t(job.canPrint ? "job.detail.verdict.printable" : "job.detail.verdict.blocked");
}

export function buildShape3dSections(job: Shape3dJob, context: DetailContext): JobDetailSections {
  const parameters: DetailItem[] = [];
  pushIdentity(parameters, "shape3d", job.status, context);
  push(parameters, "job.detail.field.source", context.t(`job.detail.shapeSource.${job.source}`));
  push(parameters, "job.detail.field.prompt", job.prompt, { isLong: true });
  push(parameters, "job.detail.field.printer", job.printer);
  push(parameters, "job.detail.field.targetSize", targetSizeValue(job, context));

  const result: DetailItem[] = [];
  push(result, "job.detail.field.verdict", verdictValue(job, context));
  push(
    result,
    "job.detail.field.sizeMm",
    job.sizeMm ? job.sizeMm.map((axis) => axis.toFixed(1)).join(" × ") : null,
    { isNumeric: true },
  );
  pushNumber(result, "job.detail.field.triangles", job.triangleCount);
  if (job.blockers.length > 0) {
    push(result, "job.detail.field.blockers", job.blockers.join(" · "), { isLong: true });
  }
  if (job.advice.length > 0) {
    push(result, "job.detail.field.advice", job.advice.join(" · "), { isLong: true });
  }
  pushNumber(result, "job.detail.field.retries", job.retries > 0 ? job.retries : null);
  // El codigo ES la pieza editable en el carril CAD: entregar solo el STL seria
  // entregar algo que no se puede ajustar.
  push(result, "job.detail.field.code", job.code, { isLong: true });

  return { parameters, timing: buildTiming(job, context), result };
}
