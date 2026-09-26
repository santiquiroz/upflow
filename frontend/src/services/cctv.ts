import { apiGet, apiPostForm, apiPostJson } from "../lib/api";

const API_BASE = "/api/v1";
import type { VideoJobResponse } from "../lib/apiTypes";
import type { UploadOptions } from "../lib/uploadRequest";

export type CctvLane = "classic" | "ai";
export type CctvTask = "clarify" | "enhance" | "roi_fusion";
export type CctvParamValue = number | string;
export type CctvParams = Readonly<Record<string, CctvParamValue>>;
export type CctvBox = readonly [number, number, number, number];
export type CctvStepCondition = "always" | "interlaced" | "anamorphic";
export type CctvRoiKind = "plate" | "face_or_object";
export type CctvRoiMethod = "median" | "trimmed_mean";

export interface CctvEnumParamSchema {
  name: string;
  type: "enum";
  choices: CctvParamValue[];
  default: CctvParamValue;
}

export interface CctvNumberParamSchema {
  name: string;
  type: "int" | "float";
  min: number;
  max: number;
  default: number | null;
  even?: boolean;
  odd?: boolean;
}

export type CctvParamSchema = CctvEnumParamSchema | CctvNumberParamSchema;

export interface CctvFilterSchema {
  name: string;
  descriptionKey: string;
  description: string;
  docUrl: string | null;
  ffmpegFilters: string[];
  params: CctvParamSchema[];
  available: boolean;
  unavailableReasonKey: string | null;
  unavailableReason: string | null;
}

export interface CctvStepSchema {
  id: string;
  labelKey: string;
  label: string;
  category: "classic" | "ai" | "label";
  defaultOn: boolean;
  placement: string | null;
  filters: CctvFilterSchema[];
  available: boolean;
}

export interface CctvPresetStep {
  id: string;
  params: CctvParams;
  when: CctvStepCondition;
}

export interface CctvPreset {
  id: string;
  labelKey: string;
  label: string;
  descriptionKey: string;
  description: string;
  aiUpscaleHint: number | null;
  lanes: Record<CctvLane, CctvPresetStep[]>;
}

export interface CctvUnavailableFilter {
  stepId: string;
  filter: string;
  missing: string[];
  reasonKey: string;
  reason: string;
}

export interface CctvAiUpscaleModel {
  id: string;
  label: string;
  scales: number[];
  generative: boolean;
  generativeLabel: string;
}

export interface CctvPresetsResponse {
  modeAvailable: boolean;
  modeUnavailableReason: string | null;
  ffmpeg: { version: string | null; gpl: boolean; binarySha256: string | null };
  presets: CctvPreset[];
  steps: Record<CctvLane, CctvStepSchema[]>;
  unavailableSteps: string[];
  unavailableFilters: CctvUnavailableFilter[];
  // Solo los reescaladores builtin con export ONNX instalado: los unicos que corren dentro del stream IA.
  aiUpscaleModels: CctvAiUpscaleModel[];
}

export interface CctvLiteAspect {
  storedSize: [number, number];
  displaySize: [number, number];
  sar: string;
  filter: string;
}

export interface CctvVideoFacts {
  codec: string;
  width: number;
  height: number;
  headerRate: string | null;
  lite: CctvLiteAspect | null;
}

export interface CctvAudioTrack {
  codec: string;
  sampleRate: number | null;
  channels: number | null;
  family: string;
}

export interface CctvGop {
  keyframes: number;
  minLength: number | null;
  maxLength: number | null;
  medianLength: number | null;
  pRatio: number;
  bRatio: number;
}

export interface CctvFrameIndex {
  csvSha256: string;
  frameCount: number;
  measuredFps: number | null;
  medianDelta: number | null;
  isVfr: boolean;
  gaps: { afterFrame: number; start: number; end: number }[];
  probableDuplicates: number;
  gop: CctvGop;
}

export interface CctvMetricSummary {
  samples: number;
  mean: number;
  median: number;
  p90: number;
}

export interface CctvQuality {
  interlace: { interlaced: boolean; fieldOrder: "tff" | "bff" | null } | null;
  blocking: CctvMetricSummary | null;
  blur: CctvMetricSummary | null;
  freezes: { start: number; end: number | null }[];
  frameStats: {
    samples: number;
    lumaMean: number;
    chromaDeviation: number;
    clippedHighPct: number;
    clippedLowPct: number;
    monochrome: boolean;
  } | null;
}

export interface CctvStepRequest {
  id: string;
  params: CctvParams;
}

export interface CctvAnalysis {
  token: string;
  sourceSha256: string;
  receivedAt: { utc: string; local: string };
  originalName: string;
  sizeBytes: number;
  container: { kind: string; label: string };
  video: CctvVideoFacts;
  audio: CctvAudioTrack[];
  frameIndex: CctvFrameIndex | null;
  gop: CctvGop | null;
  quality: CctvQuality | null;
  decodeFailed: boolean;
  suggestedPreset: string | null;
  proposedSteps: Record<CctvLane, CctvStepRequest[]> | null;
  modeAvailable: boolean;
  modeUnavailableReason: string | null;
  unavailableSteps: string[];
  unavailableFilters: CctvUnavailableFilter[];
  warnings: string[];
}

export type CctvAnalysisStatus = "running" | "completed" | "failed";

export interface CctvAnalysisJob {
  analysisJobId: string;
  status: CctvAnalysisStatus;
  statusUrl: string;
  result: CctvAnalysis | null;
  error: string | null;
  errorKey: string | null;
}

export type CctvAnalyzeReply =
  | { kind: "done"; analysis: CctvAnalysis }
  | { kind: "pending"; analysisJobId: string };

export interface CctvAcquisition {
  recorderMake?: string;
  recorderModel?: string;
  channel?: string;
  clockOffsetSeconds?: number;
  clockOffsetMethod?: string;
}

export interface CctvCaseRequest {
  caseLabel?: string;
  operatorName?: string;
  acquisition?: CctvAcquisition;
}

export interface CctvRoiRequest {
  firstFrame: number;
  lastFrame: number;
  referenceFrame: number;
  box: [number, number, number, number];
  kind: CctvRoiKind;
  scale: number;
  method: CctvRoiMethod;
}

export interface CctvJobRequest extends CctvCaseRequest {
  token: string;
  task: CctvTask;
  preset: string | null;
  steps: CctvStepRequest[];
  osdBoxes: CctvBox[];
  osdBoxesConfirmed: boolean;
  noOsd: boolean;
  trim: [number, number] | null;
  roi?: CctvRoiRequest;
  modelId?: string;
  scale?: number;
}

export interface CctvVerifyResult {
  ok: boolean;
  checked: number;
  mismatches: string[];
  missing: string[];
}

export interface CctvOsdBoxCheck {
  box: number[];
  frames: number;
  contrast: number;
  staticFraction: number | null;
  looksLikeText: boolean;
  warningKey: string | null;
}

export interface CctvOsdCheckResponse {
  checks: CctvOsdBoxCheck[];
  warnings: string[];
}

export function toAnalyzeReply(body: CctvAnalysis | CctvAnalysisJob): CctvAnalyzeReply {
  // Un clip largo no termina dentro de la ventana sincronica: el backend responde 202 con el id.
  if ("analysisJobId" in body) {
    return { kind: "pending", analysisJobId: body.analysisJobId };
  }
  return { kind: "done", analysis: body };
}

export async function analyzeCctv(file: File, options: UploadOptions = {}): Promise<CctvAnalyzeReply> {
  const formData = new FormData();
  formData.append("file", file);
  const body = await apiPostForm<CctvAnalysis | CctvAnalysisJob>("/video/cctv/analyze", formData, options);
  return toAnalyzeReply(body);
}

export function getCctvAnalysis(analysisJobId: string): Promise<CctvAnalysisJob> {
  return apiGet<CctvAnalysisJob>(`/video/cctv/analysis/${encodeURIComponent(analysisJobId)}`);
}

export function getCctvPresets(): Promise<CctvPresetsResponse> {
  return apiGet<CctvPresetsResponse>("/video/cctv/presets");
}

export function createCctvJob(request: CctvJobRequest): Promise<VideoJobResponse> {
  return apiPostJson<VideoJobResponse>("/video/cctv/jobs", request);
}

function previewPath(token: string): string {
  return `${API_BASE}/video/cctv/${encodeURIComponent(token)}/preview`;
}

// El navegador no reproduce PS ni H.265: cada cuadro lo decodifica el backend como PNG.
export function cctvFrameUrl(token: string, frame: number): string {
  return `${previewPath(token)}?frame=${frame}`;
}

export function cctvFilterPreviewUrl(token: string, frame: number, steps: readonly CctvStepRequest[]): string {
  const query = new URLSearchParams({ frame: String(frame), steps: JSON.stringify(steps) });
  return `${previewPath(token)}?${query.toString()}`;
}

export function checkCctvOsd(token: string, boxes: readonly CctvBox[], frame: number): Promise<CctvOsdCheckResponse> {
  return apiPostJson<CctvOsdCheckResponse>(`/video/cctv/${encodeURIComponent(token)}/osd-check`, {
    boxes: boxes.map((box) => [...box]),
    frame,
  });
}

export interface CctvRoiReferenceRequest {
  firstFrame: number;
  lastFrame: number;
  box: number[];
  steps: CctvStepRequest[];
}

export interface CctvRoiReferenceResponse {
  referenceFrame: number;
}

export function suggestCctvRoiReference(token: string, request: CctvRoiReferenceRequest): Promise<CctvRoiReferenceResponse> {
  return apiPostJson<CctvRoiReferenceResponse>(`/video/cctv/${encodeURIComponent(token)}/roi/reference`, request);
}

export function verifyCctvFiles(jobId: string): Promise<CctvVerifyResult> {
  return apiPostJson<CctvVerifyResult>(`/video/jobs/${encodeURIComponent(jobId)}/verify`, {});
}
