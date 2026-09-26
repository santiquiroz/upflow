import type { CctvArtifactLink } from "../../../lib/apiTypes";
import type { CctvVerifyResult } from "../../../services/cctv";

export interface ResultFile {
  name: string;
  url: string;
  labelKey: string | null;
  params: Record<string, string>;
}

export type VerifyOutcome =
  | { kind: "unchanged"; checked: number }
  | { kind: "changed"; checked: number; mismatches: string[]; missing: string[] };

export const REPORT_ARTIFACT = "report_html";
export const PACKAGE_ARTIFACT = "package";

// Primero lo que el usuario mira (videos y cuadros), despues los archivos tecnicos.
const MEDIA_ARTIFACTS = ["analysis", "viewing", "enhanced", "comparison"] as const;
const TECHNICAL_ARTIFACTS = ["report_json", "sha256sums", "reproduce", "frame_index"] as const;
const OWN_ACTION_ARTIFACTS: ReadonlySet<string> = new Set([REPORT_ARTIFACT, PACKAGE_ARTIFACT]);
const STILL = /^still:(\d+):(original|processed)$/;
const ROI = /^roi:(.+)$/;
const KNOWN_ORDER: readonly string[] = [...MEDIA_ARTIFACTS, ...TECHNICAL_ARTIFACTS];
const PATTERN_RANK = MEDIA_ARTIFACTS.length;
const TECHNICAL_RANK_OFFSET = 1;

export function artifactUrl(artifacts: readonly CctvArtifactLink[], name: string): string | null {
  return artifacts.find((artifact) => artifact.name === name)?.url ?? null;
}

function stillFile(artifact: CctvArtifactLink, match: RegExpMatchArray): ResultFile {
  return { ...artifact, labelKey: `cctv.result.file.still.${match[2]}`, params: { frame: match[1] } };
}

function roiFile(artifact: CctvArtifactLink, match: RegExpMatchArray): ResultFile {
  return { ...artifact, labelKey: "cctv.result.file.roi", params: { name: match[1] } };
}

function namedFile(artifact: CctvArtifactLink): ResultFile {
  const labelKey = KNOWN_ORDER.includes(artifact.name) ? `cctv.result.file.${artifact.name}` : null;
  return { ...artifact, labelKey, params: {} };
}

export function resultFileOf(artifact: CctvArtifactLink): ResultFile {
  const still = artifact.name.match(STILL);
  if (still) {
    return stillFile(artifact, still);
  }
  const roi = artifact.name.match(ROI);
  return roi ? roiFile(artifact, roi) : namedFile(artifact);
}

function rankOf(name: string): number {
  const known = KNOWN_ORDER.indexOf(name);
  if (known < 0) {
    return PATTERN_RANK;
  }
  return known < PATTERN_RANK ? known : known + TECHNICAL_RANK_OFFSET;
}

export function resultFiles(artifacts: readonly CctvArtifactLink[]): ResultFile[] {
  return artifacts
    .filter((artifact) => !OWN_ACTION_ARTIFACTS.has(artifact.name))
    .map((artifact, position) => ({ artifact, position, rank: rankOf(artifact.name) }))
    .sort((a, b) => a.rank - b.rank || a.position - b.position)
    .map(({ artifact }) => resultFileOf(artifact));
}

export function verifyOutcome(result: CctvVerifyResult): VerifyOutcome {
  if (result.ok) {
    return { kind: "unchanged", checked: result.checked };
  }
  return { kind: "changed", checked: result.checked, mismatches: result.mismatches, missing: result.missing };
}
