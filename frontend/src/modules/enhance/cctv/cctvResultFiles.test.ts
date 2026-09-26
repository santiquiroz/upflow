import { describe, expect, it } from "vitest";
import type { CctvArtifactLink } from "../../../lib/apiTypes";
import { artifactUrl, resultFileOf, resultFiles, verifyOutcome } from "./cctvResultFiles";

function link(name: string): CctvArtifactLink {
  return { name, url: `/api/v1/video/jobs/job-1/artifacts/${name}` };
}

// El orden en que el backend los lista (cctv_artifacts.listed_artifacts).
const LISTED = [
  "analysis",
  "viewing",
  "comparison",
  "package",
  "report_json",
  "report_html",
  "sha256sums",
  "reproduce",
  "frame_index",
  "still:120:original",
  "still:120:processed",
].map(link);

describe("resultFiles", () => {
  it("lists the videos and frames first and the technical files after them", () => {
    expect(resultFiles(LISTED).map((file) => file.name)).toEqual([
      "analysis",
      "viewing",
      "comparison",
      "still:120:original",
      "still:120:processed",
      "report_json",
      "sha256sums",
      "reproduce",
      "frame_index",
    ]);
  });

  it("leaves the report and the handover package to their own buttons", () => {
    const names = resultFiles(LISTED).map((file) => file.name);

    expect(names).not.toContain("report_html");
    expect(names).not.toContain("package");
  });

  it("keeps the backend link as is", () => {
    expect(resultFiles([link("analysis")])[0].url).toBe("/api/v1/video/jobs/job-1/artifacts/analysis");
  });
});

describe("resultFileOf", () => {
  it("labels an exported frame with its number and role", () => {
    expect(resultFileOf(link("still:7:processed"))).toMatchObject({
      labelKey: "cctv.result.file.still.processed",
      params: { frame: "7" },
    });
  });

  it("labels a multi-frame still by its name", () => {
    expect(resultFileOf(link("roi:fused"))).toMatchObject({ labelKey: "cctv.result.file.roi", params: { name: "fused" } });
  });

  it("has no label for an artifact this version doesn't know, so the raw name is shown", () => {
    expect(resultFileOf(link("mystery")).labelKey).toBeNull();
  });
});

describe("artifactUrl", () => {
  it("finds an artifact by name", () => {
    expect(artifactUrl(LISTED, "package")).toBe("/api/v1/video/jobs/job-1/artifacts/package");
  });

  it("is null when the job did not produce it", () => {
    expect(artifactUrl([link("analysis")], "package")).toBeNull();
  });
});

describe("verifyOutcome", () => {
  it("reports no changes when every hash still matches", () => {
    expect(verifyOutcome({ ok: true, checked: 9, mismatches: [], missing: [] })).toEqual({
      kind: "unchanged",
      checked: 9,
    });
  });

  it("lists the changed and the missing files", () => {
    expect(verifyOutcome({ ok: false, checked: 9, mismatches: ["analysis.mkv"], missing: ["report.html"] })).toEqual({
      kind: "changed",
      checked: 9,
      mismatches: ["analysis.mkv"],
      missing: ["report.html"],
    });
  });
});
