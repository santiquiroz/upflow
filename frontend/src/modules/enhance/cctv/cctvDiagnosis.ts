import { formatFps } from "../../../lib/formatFps";
import type { CctvAnalysis, CctvAudioTrack } from "../../../services/cctv";
import { translateOr, type Translate } from "./cctvText";

export interface DiagnosisRow {
  labelKey: string;
  value: string;
}

const MISSING = "—";
const HZ_PER_KHZ = 1000;
const AUDIO_CODEC_LABELS: Readonly<Record<string, string>> = {
  pcm_alaw: "G.711 A-law",
  pcm_mulaw: "G.711 µ-law",
  aac: "AAC",
};
const FILTER_UNAVAILABLE = "cctv.filterUnavailable";
const MODE_UNAVAILABLE = "cctv.error.modeUnavailable";
// El aviso de ingesta ya explica el modo lite; el del diagnostico diria lo mismo.
const REDUNDANT_WITH: Readonly<Record<string, string>> = { "cctv.warning.liteAspect": "cctv.lite" };

function yesNo(t: Translate, value: boolean | null | undefined): string {
  if (value === null || value === undefined) {
    return MISSING;
  }
  return t(value ? "cctv.diag.yes" : "cctv.diag.no");
}

function size(width: number, height: number): string {
  return `${width} × ${height}`;
}

function decimal(value: number | null | undefined, digits: number): string {
  return value === null || value === undefined ? MISSING : value.toFixed(digits);
}

export function audioLabel(track: CctvAudioTrack): string {
  const codec = AUDIO_CODEC_LABELS[track.codec] ?? track.codec;
  return track.sampleRate ? `${codec}, ${track.sampleRate / HZ_PER_KHZ} kHz` : codec;
}

function interlaceValue(t: Translate, analysis: CctvAnalysis): string {
  const interlace = analysis.quality?.interlace;
  if (!interlace?.interlaced) {
    return yesNo(t, interlace?.interlaced);
  }
  return interlace.fieldOrder ? t(`cctv.diag.interlaced.${interlace.fieldOrder}`) : t("cctv.diag.yes");
}

function frameRateValue(analysis: CctvAnalysis): string {
  const declared = formatFps(analysis.video.headerRate) ?? MISSING;
  return `${declared} / ${decimal(analysis.frameIndex?.measuredFps, 2)}`;
}

function audioValue(t: Translate, analysis: CctvAnalysis): string {
  return analysis.audio.length > 0 ? analysis.audio.map(audioLabel).join(" · ") : t("cctv.diag.noAudio");
}

function liteRows(analysis: CctvAnalysis): DiagnosisRow[] {
  const lite = analysis.video.lite;
  return lite ? [{ labelKey: "cctv.diag.display", value: size(...lite.displaySize) }] : [];
}

export function diagnosisRows(t: Translate, analysis: CctvAnalysis): DiagnosisRow[] {
  const { video, frameIndex, quality } = analysis;
  const stats = quality?.frameStats;
  return [
    { labelKey: "cctv.diag.container", value: analysis.container.label },
    { labelKey: "cctv.diag.codec", value: video.codec },
    { labelKey: "cctv.diag.stored", value: size(video.width, video.height) },
    ...liteRows(analysis),
    { labelKey: "cctv.diag.fps", value: frameRateValue(analysis) },
    { labelKey: "cctv.diag.vfr", value: yesNo(t, frameIndex?.isVfr) },
    { labelKey: "cctv.diag.frames", value: String(frameIndex?.frameCount ?? MISSING) },
    { labelKey: "cctv.diag.duplicates", value: String(frameIndex?.probableDuplicates ?? MISSING) },
    { labelKey: "cctv.diag.gaps", value: String(frameIndex?.gaps.length ?? MISSING) },
    { labelKey: "cctv.diag.gop", value: String(analysis.gop?.medianLength ?? MISSING) },
    { labelKey: "cctv.diag.interlaced", value: interlaceValue(t, analysis) },
    { labelKey: "cctv.diag.blocking", value: decimal(quality?.blocking?.median, 1) },
    { labelKey: "cctv.diag.blur", value: decimal(quality?.blur?.median, 1) },
    { labelKey: "cctv.diag.nightIr", value: yesNo(t, stats?.monochrome) },
    { labelKey: "cctv.diag.clipped", value: stats ? `${stats.clippedHighPct.toFixed(1)}%` : MISSING },
    { labelKey: "cctv.diag.audio", value: audioValue(t, analysis) },
  ];
}

function isRedundant(key: string, warnings: readonly string[]): boolean {
  const covering = REDUNDANT_WITH[key];
  return covering !== undefined && warnings.includes(covering);
}

function warningTexts(t: Translate, key: string, analysis: CctvAnalysis): string[] {
  if (key === FILTER_UNAVAILABLE) {
    return analysis.unavailableFilters.map((item) => t(FILTER_UNAVAILABLE, { filter: item.filter }));
  }
  if (key === MODE_UNAVAILABLE) {
    return [analysis.modeUnavailableReason ?? t(key)];
  }
  return [translateOr(t, key, key)];
}

export function diagnosisWarnings(t: Translate, analysis: CctvAnalysis): string[] {
  return analysis.warnings
    .filter((key) => !isRedundant(key, analysis.warnings))
    .flatMap((key) => warningTexts(t, key, analysis));
}
