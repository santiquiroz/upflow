import type { JobMetadata, JobStage, StageStatus } from "./apiTypes";

const INTERPOLATION_STAGE_KEY = "interpolating_frames";

export type StepIconState = "done" | "active" | "pending";

export interface StepperItem {
  key: string;
  label: string;
  iconState: StepIconState;
}

function iconStateForStageStatus(status: StageStatus): StepIconState {
  if (status === "done") {
    return "done";
  }
  if (status === "active") {
    return "active";
  }
  return "pending";
}

export function deriveStepper(stages: JobStage[] | undefined): StepperItem[] {
  if (!stages) {
    return [];
  }
  return stages.map((stage) => ({
    key: stage.key,
    label: stage.label,
    iconState: iconStateForStageStatus(stage.status),
  }));
}

export function isProgressDeterminate(progressPct: number | null | undefined): progressPct is number {
  return typeof progressPct === "number" && Number.isFinite(progressPct);
}

// Progress is best-effort and must never appear to move backward in the UI
// (e.g. a stage-transition recompute can transiently report a lower fraction
// than what was already shown) -- callers keep the highest value seen so far
// per job and pass it in here as `previousMax`.
export function toMonotonicProgressPct(previousMax: number, candidate: number | null | undefined): number {
  if (!isProgressDeterminate(candidate)) {
    return previousMax;
  }
  return Math.max(previousMax, candidate);
}

// During interpolating_frames the backend counts framesDone against
// interpFramesTotal (source x multiplier, larger than the source framesTotal),
// so dividing by framesTotal there would render an impossible "800 / 400".
export function resolveFramesDenominator(metadata: JobMetadata | undefined): number | null | undefined {
  if (!metadata) {
    return undefined;
  }
  if (metadata.stage === INTERPOLATION_STAGE_KEY) {
    return metadata.interpFramesTotal ?? metadata.framesTotal;
  }
  return metadata.framesTotal;
}

export function areFramesReportable(
  framesDone: number | null | undefined,
  framesTotal: number | null | undefined,
): framesTotal is number {
  return (
    typeof framesDone === "number" &&
    typeof framesTotal === "number" &&
    framesTotal > 0 &&
    framesDone <= framesTotal
  );
}
