import type { CctvRoiSummary } from "../../../lib/apiTypes";
import type {
  CctvBox,
  CctvRoiKind,
  CctvRoiMethod,
  CctvRoiReferenceRequest,
  CctvRoiRequest,
  CctvStepRequest,
  CctvStepSchema,
} from "../../../services/cctv";
import { trimFrameCount, type TrimRange } from "./cctvFrames";
import type { TranslatedReason } from "./cctvLanes";

export interface RoiChoice {
  first: number | null;
  last: number | null;
  reference: number | null;
  box: CctvBox | null;
  kind: CctvRoiKind;
  scale: number;
  method: CctvRoiMethod;
}

export const ROI_KINDS: readonly CctvRoiKind[] = ["plate", "face_or_object"];
export const ROI_SCALES: readonly number[] = [2, 3, 4];
export const ROI_METHODS: readonly CctvRoiMethod[] = ["median", "trimmed_mean"];
// Igual que CCTV_ROI_MAX_FRAMES por defecto; el backend valida el tope configurado.
export const MAX_ROI_FRAMES = 60;
// Lectura de Axis sobre IEC 62676-4: unos 40 px sobre una cara para identificar (spec §4.9).
const FACE_MIN_WIDTH_PX = 40;
// La fusion solo desentrelaza y desbloquea antes de alinear: el denoise temporal correlaciona los cuadros.
const ROI_PREFILTER_STEP_IDS: ReadonlySet<string> = new Set(["deinterlace", "deblock"]);
// "Frames used" es un dato del resultado que se muestra aparte, no un aviso.
const FRAMES_USED_KEY = "cctv.roi.framesUsed";

export const EMPTY_ROI: RoiChoice = {
  first: null,
  last: null,
  reference: null,
  box: null,
  kind: "plate",
  scale: 2,
  method: "median",
};

export function roiCatalog(catalog: readonly CctvStepSchema[]): CctvStepSchema[] {
  return catalog.filter((step) => ROI_PREFILTER_STEP_IDS.has(step.id));
}

// La caja se dibuja sobre el cuadro que se ve: ese pasa a ser el cuadro de referencia.
export function withRoiBox(roi: RoiChoice, box: CctvBox | null, frame: number): RoiChoice {
  return box ? { ...roi, box, reference: frame } : { ...roi, box: null, reference: null };
}

export function roiRange(roi: RoiChoice): TrimRange | null {
  if (roi.first === null || roi.last === null || roi.last < roi.first) {
    return null;
  }
  return [roi.first, roi.last];
}

export function roiFrameCount(roi: RoiChoice): number | null {
  const range = roiRange(roi);
  return range ? trimFrameCount(range) : null;
}

function isRangeValid(range: TrimRange | null, frameCount: number): boolean {
  return range !== null && range[0] >= 0 && range[1] < frameCount && trimFrameCount(range) <= MAX_ROI_FRAMES;
}

function isReferenceInRange(range: TrimRange | null, reference: number | null): boolean {
  return range !== null && reference !== null && range[0] <= reference && reference <= range[1];
}

export function roiBlockerKey(roi: RoiChoice, frameCount: number): string | null {
  const range = roiRange(roi);
  const checks: ReadonlyArray<readonly [boolean, string]> = [
    [roi.box === null, "cctv.roi.blocked.box"],
    [!isRangeValid(range, frameCount), "cctv.roi.blocked.range"],
    [!isReferenceInRange(range, roi.reference), "cctv.roi.blocked.reference"],
  ];
  return checks.find(([blocked]) => blocked)?.[1] ?? null;
}

// "Suggest reference frame" mide la caja en cada cuadro del rango: hace falta la caja y un rango valido.
export function roiReferenceRequest(
  roi: RoiChoice,
  frameCount: number,
  steps: readonly CctvStepRequest[],
): CctvRoiReferenceRequest | null {
  const range = roiRange(roi);
  if (roi.box === null || range === null || !isRangeValid(range, frameCount)) {
    return null;
  }
  return { firstFrame: range[0], lastFrame: range[1], box: [...roi.box], steps: [...steps] };
}

export function roiRequest(roi: RoiChoice): CctvRoiRequest | null {
  if (roi.box === null || roi.first === null || roi.last === null || roi.reference === null) {
    return null;
  }
  return {
    firstFrame: roi.first,
    lastFrame: roi.last,
    referenceFrame: roi.reference,
    box: [...roi.box],
    kind: roi.kind,
    scale: roi.scale,
    method: roi.method,
  };
}

// Mismo criterio que roi_fusion.density_notices, en pixeles guardados (la caja se dibuja sobre el cuadro guardado).
export function roiDensityNotice(kind: CctvRoiKind, box: CctvBox | null): TranslatedReason | null {
  if (box === null) {
    return null;
  }
  if (kind === "plate") {
    return { key: "cctv.roi.densityPlate", params: { px: box[3] } };
  }
  return box[2] < FACE_MIN_WIDTH_PX ? { key: "cctv.roi.densityFace", params: { px: box[2] } } : null;
}

function rejectedNotice(rejected: readonly number[]): TranslatedReason[] {
  return rejected.length > 0 ? [{ key: "cctv.roi.result.rejected", params: { frames: rejected.join(", ") } }] : [];
}

export function roiResultNotices(roi: CctvRoiSummary): TranslatedReason[] {
  const notices = roi.notices
    .filter((notice) => notice.key !== FRAMES_USED_KEY)
    .map((notice) => ({ key: notice.key, params: notice.params }));
  return [...notices, ...rejectedNotice(roi.rejectedFrames)];
}
