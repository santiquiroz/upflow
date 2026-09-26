import type { VideoCapabilities } from "../../../lib/apiTypes";
import type { CctvAnalysis, CctvLane, CctvTask } from "../../../services/cctv";

export interface TranslatedReason {
  key: string;
  params: Record<string, string | number>;
}

export interface AiLaneState {
  available: boolean;
  reason: TranslatedReason | null;
}

export interface StartBlocker {
  key: string;
}

export interface StartInputs {
  modeAvailable: boolean;
  decodeFailed: boolean;
  lane: CctvLane;
  aiAvailable: boolean;
  task: CctvTask;
  noOsd: boolean;
  osdBoxesConfirmed: boolean;
  osdBoxCount: number;
  incompleteStepIds: readonly string[];
  trimValid: boolean;
  caseDetailsValid: boolean;
  roiBlockerKey: string | null;
}

export const LANE_TASKS: Readonly<Record<CctvLane, readonly CctvTask[]>> = {
  classic: ["clarify", "roi_fusion"],
  ai: ["enhance", "roi_fusion"],
};

// Las tareas que producen video: piden decision sobre el OSD y recorte; la foto multi-cuadro usa su rango.
const VIDEO_TASKS: ReadonlySet<CctvTask> = new Set<CctvTask>(["clarify", "enhance"]);
// DRUNet en CPU: unos 15 s por cuadro 1080p (derivado, spec 4.7), escalado por pixeles.
const AI_CPU_SECONDS_PER_1080P_FRAME = 15;
const PIXELS_1080P = 1920 * 1080;
const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_MINUTE = 60;
const NEEDS_GPU_REASON = "capability.setup.needsGpu";

export function isVideoTask(task: CctvTask): boolean {
  return VIDEO_TASKS.has(task);
}

// Solo un job que genera video con IA pide el modal: la foto multi-cuadro es clasica en los dos carriles.
export function needsAiConfirmation(task: CctvTask): boolean {
  return task === "enhance";
}

export function defaultTask(lane: CctvLane): CctvTask {
  return LANE_TASKS[lane][0];
}

export function estimateAiCpuSeconds(frameCount: number, width: number, height: number): number {
  return (frameCount * AI_CPU_SECONDS_PER_1080P_FRAME * width * height) / PIXELS_1080P;
}

export function formatRoughDuration(seconds: number): string {
  if (seconds >= SECONDS_PER_HOUR) {
    return `${Math.round(seconds / SECONDS_PER_HOUR)} h`;
  }
  return `${Math.max(1, Math.round(seconds / SECONDS_PER_MINUTE))} min`;
}

function cpuBlockedReason(analysis: CctvAnalysis | null): TranslatedReason | null {
  const frameCount = analysis?.frameIndex?.frameCount;
  if (!analysis || !frameCount) {
    return null;
  }
  const seconds = estimateAiCpuSeconds(frameCount, analysis.video.width, analysis.video.height);
  return { key: "cctv.ai.cpuBlocked", params: { eta: formatRoughDuration(seconds) } };
}

function unavailableReason(caps: VideoCapabilities, analysis: CctvAnalysis | null): TranslatedReason {
  const key = caps.cctvAiReasonKey ?? NEEDS_GPU_REASON;
  const cpuBlocked = key === NEEDS_GPU_REASON ? cpuBlockedReason(analysis) : null;
  return cpuBlocked ?? { key, params: {} };
}

export function aiLaneState(caps: VideoCapabilities | undefined, analysis: CctvAnalysis | null): AiLaneState {
  if (!caps) {
    return { available: false, reason: null };
  }
  if (caps.cctvAiAvailable) {
    return { available: true, reason: null };
  }
  return { available: false, reason: unavailableReason(caps, analysis) };
}

function hasOsdDecision(inputs: StartInputs): boolean {
  return !isVideoTask(inputs.task) || inputs.noOsd || (inputs.osdBoxesConfirmed && inputs.osdBoxCount > 0);
}

function roiBlocker(inputs: StartInputs): StartBlocker | null {
  return inputs.task === "roi_fusion" && inputs.roiBlockerKey !== null ? { key: inputs.roiBlockerKey } : null;
}

export function startBlocker(inputs: StartInputs): StartBlocker | null {
  const checks: ReadonlyArray<readonly [boolean, string]> = [
    [!inputs.modeAvailable, "cctv.blocked.modeUnavailable"],
    [inputs.decodeFailed, "cctv.undecodable"],
    [inputs.lane === "ai" && !inputs.aiAvailable, "cctv.blocked.aiUnavailable"],
    [inputs.incompleteStepIds.length > 0, "cctv.blocked.incompleteSteps"],
    [!inputs.trimValid, "cctv.trim.invalid"],
    [!inputs.caseDetailsValid, "cctv.case.offsetInvalid"],
    [!hasOsdDecision(inputs), "cctv.osd.confirm"],
  ];
  const failed = checks.find(([blocked]) => blocked);
  return failed ? { key: failed[1] } : roiBlocker(inputs);
}
