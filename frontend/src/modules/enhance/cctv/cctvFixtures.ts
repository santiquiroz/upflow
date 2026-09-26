import type { CctvRoiSummary } from "../../../lib/apiTypes";
import type {
  CctvAiUpscaleModel,
  CctvAnalysis,
  CctvFilterSchema,
  CctvPreset,
  CctvPresetsResponse,
  CctvStepSchema,
} from "../../../services/cctv";

function filter(name: string, params: CctvFilterSchema["params"] = [], available = true): CctvFilterSchema {
  return {
    name,
    descriptionKey: `cctv.filter.${name}.description`,
    description: `Applied ${name}.`,
    docUrl: `https://ffmpeg.org/ffmpeg-filters.html#${name}`,
    ffmpegFilters: [name],
    params,
    available,
    unavailableReasonKey: available ? null : "cctv.filterUnavailable",
    unavailableReason: available ? null : `This ffmpeg build doesn't include ${name}, so this step is off.`,
  };
}

function step(
  id: string,
  label: string,
  filters: CctvFilterSchema[],
  category: CctvStepSchema["category"] = "classic",
): CctvStepSchema {
  return {
    id,
    labelKey: `cctv.step.${id}`,
    label,
    category,
    defaultOn: false,
    placement: null,
    filters,
    available: filters.some((candidate) => candidate.available),
  };
}

const TRIM = step("trim", "Trim", [
  filter("trim", [
    { name: "start_frame", type: "int", min: 0, max: 2147483647, default: null },
    { name: "end_frame", type: "int", min: 0, max: 2147483647, default: null },
  ]),
]);
const ASPECT = step("aspect", "Correct aspect ratio", [
  filter("setsar", [
    { name: "num", type: "int", min: 1, max: 100, default: null },
    { name: "den", type: "int", min: 1, max: 100, default: null },
  ]),
]);
const DEINTERLACE = step("deinterlace", "Deinterlace", [
  filter("bwdif", [{ name: "mode", type: "enum", choices: ["send_frame"], default: "send_frame" }]),
  filter("yadif", [{ name: "mode", type: "enum", choices: ["send_frame"], default: "send_frame" }]),
]);
const DEBLOCK = step("deblock", "Reduce blocking", [
  filter("deblock", [
    { name: "filter_type", type: "enum", choices: ["weak", "strong"], default: "weak" },
    { name: "block", type: "int", min: 4, max: 512, default: 8 },
  ]),
  filter("fspp", [{ name: "quality", type: "int", min: 4, max: 5, default: 4 }], false),
]);
const AI_DEBLOCK = step(
  "ai_deblock",
  "AI deblock",
  [filter("drunet_deblock", [{ name: "strength", type: "int", min: 0, max: 100, default: 40 }])],
  "ai",
);
const DENOISE = step("denoise", "Reduce noise", [
  filter("hqdn3d", [
    { name: "luma_spatial", type: "float", min: 0, max: 50, default: 4 },
    { name: "luma_tmp", type: "float", min: 0, max: 50, default: 6 },
  ], false),
  filter("atadenoise", [{ name: "s", type: "int", min: 5, max: 129, default: 9, odd: true }]),
]);
const CROP = step("crop", "Crop", [
  filter("crop", [
    { name: "w", type: "int", min: 2, max: 16384, default: null, even: true },
    { name: "h", type: "int", min: 2, max: 16384, default: null, even: true },
  ]),
]);
const GRAY = step("gray", "Grayscale", [filter("gray")]);
const SCALE = step("scale", "Enlarge", [
  filter("scale", [
    { name: "factor", type: "int", min: 1, max: 8, default: 2 },
    { name: "flags", type: "enum", choices: ["neighbor", "bicubic", "lanczos"], default: "neighbor" },
  ]),
]);
const AI_UPSCALE = step("ai_upscale", "AI upscale", [filter("onnx_upscale")], "ai");
const SHARPEN = step("sharpen", "Sharpen", [filter("cas", [{ name: "strength", type: "float", min: 0, max: 1, default: 0.5 }])]);
const OSD_PROTECT = step("osd_protect", "Protect on-screen text", [filter("osd_restore")]);
const K_PARAM = { type: "float", min: -1, max: 1, default: 0 } as const;
const LENSCORRECTION: CctvFilterSchema = {
  ...filter("lenscorrection", [
    { name: "k1", ...K_PARAM },
    { name: "k2", ...K_PARAM },
  ]),
  presets: [
    { name: "wide", labelKey: "cctv.filter.lenscorrection.preset.wide", label: "Wide angle (about 4 mm)", params: { k1: -0.12, k2: -0.01 } },
    { name: "very_wide", labelKey: "cctv.filter.lenscorrection.preset.very_wide", label: "Very wide angle (2.8 mm)", params: { k1: -0.22, k2: -0.02 } },
  ],
};
export const LENS = step("lens", "Lens correction", [LENSCORRECTION, filter("v360")]);
const AI_LABEL = step("ai_label", "AI label", [filter("label_band")], "label");

export const CLASSIC_STEPS: CctvStepSchema[] = [
  TRIM, ASPECT, DEINTERLACE, DEBLOCK, DENOISE, CROP, GRAY, SCALE, SHARPEN, OSD_PROTECT,
];

export const AI_STEPS: CctvStepSchema[] = [
  TRIM, ASPECT, DEINTERLACE, DEBLOCK, AI_DEBLOCK, DENOISE, CROP, GRAY, SCALE, AI_UPSCALE, SHARPEN, OSD_PROTECT, AI_LABEL,
];

export const DAY_PRESET: CctvPreset = {
  id: "day",
  labelKey: "cctv.preset.day",
  label: "Day",
  descriptionKey: "cctv.preset.day.description",
  description: "For color footage with moderate compression blocking.",
  aiUpscaleHint: null,
  lanes: {
    classic: [
      { id: "aspect", params: {}, when: "anamorphic" },
      { id: "deinterlace", params: { filter: "bwdif", mode: "send_frame" }, when: "interlaced" },
      { id: "deblock", params: { filter: "deblock", filter_type: "weak", block: 8 }, when: "always" },
      { id: "denoise", params: { filter: "atadenoise", s: 9 }, when: "always" },
      { id: "osd_protect", params: {}, when: "always" },
    ],
    ai: [
      { id: "aspect", params: {}, when: "anamorphic" },
      { id: "ai_deblock", params: { strength: 40 }, when: "always" },
      { id: "osd_protect", params: {}, when: "always" },
    ],
  },
};

export const NIGHT_PRESET: CctvPreset = {
  id: "night_ir",
  labelKey: "cctv.preset.night_ir",
  label: "Night / IR",
  descriptionKey: "cctv.preset.night_ir.description",
  description: "For dark or infrared footage with no real color.",
  aiUpscaleHint: null,
  lanes: {
    classic: [
      { id: "deblock", params: { filter: "deblock", filter_type: "strong", block: 8 }, when: "always" },
      { id: "denoise", params: { filter: "hqdn3d", luma_spatial: 3 }, when: "always" },
      { id: "gray", params: {}, when: "always" },
      { id: "osd_protect", params: {}, when: "always" },
    ],
    ai: [
      { id: "ai_deblock", params: { strength: 60 }, when: "always" },
      { id: "gray", params: {}, when: "always" },
    ],
  },
};

export const AI_UPSCALE_MODELS: CctvAiUpscaleModel[] = [
  {
    id: "realesrgan-x4plus",
    label: "Real-ESRGAN x4plus",
    scales: [2, 4],
    generative: true,
    generativeLabel: "Generative (invents texture)",
  },
  {
    id: "plain-x2",
    label: "Plain x2",
    scales: [2],
    generative: false,
    generativeLabel: "Non-generative",
  },
];

export const PRESETS_RESPONSE: CctvPresetsResponse = {
  modeAvailable: true,
  modeUnavailableReason: null,
  ffmpeg: { version: "N-123588", gpl: true, binarySha256: "ab".repeat(32) },
  presets: [DAY_PRESET, NIGHT_PRESET],
  steps: { classic: CLASSIC_STEPS, ai: AI_STEPS },
  unavailableSteps: [],
  unavailableFilters: [
    { stepId: "deblock", filter: "fspp", missing: ["fspp"], reasonKey: "cctv.filterUnavailable", reason: "missing fspp" },
  ],
  aiUpscaleModels: AI_UPSCALE_MODELS,
};

export const ANALYSIS: CctvAnalysis = {
  token: "tok-1",
  sourceSha256: "cd".repeat(32),
  receivedAt: { utc: "2026-09-25T20:00:00.000Z", local: "2026-09-25T15:00:00.000-05:00" },
  originalName: "ch01_20260920.mp4",
  sizeBytes: 1048576,
  container: { kind: "hikvision_ps", label: "Hikvision PS" },
  video: { codec: "hevc", width: 960, height: 1080, headerRate: "25/1", lite: {
    storedSize: [960, 1080],
    displaySize: [1920, 1080],
    sar: "2:1",
    filter: "setsar=2",
  } },
  audio: [{ codec: "pcm_alaw", sampleRate: 8000, channels: 1, family: "g711" }],
  frameIndex: {
    csvSha256: "ef".repeat(32),
    frameCount: 750,
    measuredFps: 12.5,
    medianDelta: 0.08,
    isVfr: true,
    gaps: [{ afterFrame: 10, start: 0.8, end: 1.6 }],
    probableDuplicates: 3,
    gop: { keyframes: 15, minLength: 50, maxLength: 50, medianLength: 50, pRatio: 0.98, bRatio: 0 },
  },
  gop: { keyframes: 15, minLength: 50, maxLength: 50, medianLength: 50, pRatio: 0.98, bRatio: 0 },
  quality: {
    interlace: { interlaced: true, fieldOrder: "tff" },
    blocking: { samples: 30, mean: 20, median: 21.5, p90: 40 },
    blur: { samples: 30, mean: 4, median: 4.25, p90: 8 },
    freezes: [],
    frameStats: {
      samples: 50,
      lumaMean: 60,
      chromaDeviation: 1,
      clippedHighPct: 7,
      clippedLowPct: 0.5,
      monochrome: true,
    },
  },
  decodeFailed: false,
  suggestedPreset: "night_ir",
  proposedSteps: null,
  modeAvailable: true,
  modeUnavailableReason: null,
  unavailableSteps: [],
  unavailableFilters: [
    { stepId: "deblock", filter: "fspp", missing: ["fspp"], reasonKey: "cctv.filterUnavailable", reason: "missing fspp" },
  ],
  warnings: ["cctv.lite", "cctv.warning.liteAspect", "cctv.warning.monochrome", "cctv.filterUnavailable"],
};

export const ROI_SUMMARY: CctvRoiSummary = {
  kind: "plate",
  scale: 3,
  method: "median",
  motion: "homography",
  referenceFrame: 20,
  framesTotal: 30,
  framesUsed: 28,
  effectiveSamples: 2,
  rejectedFrames: [3, 17],
  nearCopies: true,
  density: { kind: "plate", axis: "height", storedPx: 14, displayPx: 14 },
  clippedFramesPct: 0,
  notices: [
    { key: "cctv.roi.densityPlate", params: { px: 14 } },
    { key: "cctv.roi.framesUsed", params: { used: 28, total: 30, effective: 2 } },
    { key: "cctv.roi.nearCopies", params: {} },
  ],
};
