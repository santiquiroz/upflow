import { describe, expect, it } from "vitest";
import {
  editorSource,
  hasUncolored,
  readRestoreSummary,
  showsInfoCard,
  type RestoreResultSummary,
} from "./restoreResultModel";
import { makeCompletedRestoreJob, makeRestoreMetadata } from "./restoreTestFixtures";

function summaryOf(overrides: Record<string, unknown> = {}): RestoreResultSummary {
  const summary = readRestoreSummary(makeCompletedRestoreJob(makeRestoreMetadata(overrides)));
  if (summary === null) {
    throw new Error("fixture should parse");
  }
  return summary;
}

describe("readRestoreSummary", () => {
  it("reads the artifacts, download names and view resolution of a finished restoration", () => {
    expect(summaryOf()).toEqual({
      artifacts: ["preview", "view", "beforeafter", "sidecar"],
      downloadNames: {
        restored: "grandma_restored.png",
        uncolored: "grandma_uncolored.png",
        beforeAfter: "grandma_before-after.jpg",
        sidecar: "grandma_restore.json",
      },
      viewFullResolution: true,
      compositeReasons: [],
    });
  });

  it("ignores jobs that have not finished", () => {
    const job = { ...makeCompletedRestoreJob(makeRestoreMetadata()), status: "running" as const };
    expect(readRestoreSummary(job)).toBeNull();
  });

  it("ignores jobs without a restoration summary", () => {
    expect(readRestoreSummary(makeCompletedRestoreJob(undefined))).toBeNull();
  });

  it("ignores a preview-area run, which only produces the preview artifact", () => {
    const metadata = { artifacts: ["preview"], previewCrop: [0, 0, 512, 512] };
    expect(readRestoreSummary(makeCompletedRestoreJob(metadata))).toBeNull();
  });

  it("treats a missing view resolution flag as a reduced copy", () => {
    expect(summaryOf({ viewFullResolution: undefined }).viewFullResolution).toBe(false);
  });
});

describe("result facts", () => {
  it("knows when there is an uncolored version", () => {
    expect(hasUncolored(summaryOf())).toBe(false);
    expect(hasUncolored(summaryOf({ artifacts: ["preview", "view", "beforeafter", "sidecar", "uncolored"] }))).toBe(
      true,
    );
  });

  it("shows the info card for faces, colorize and large fills, not for a generative upscale alone", () => {
    expect(showsInfoCard(summaryOf())).toBe(false);
    expect(showsInfoCard(summaryOf({ compositeReasons: ["generativeUpscale"] }))).toBe(false);
    for (const reason of ["faces", "colorize", "largeFill", "fillOverFace"]) {
      expect(showsInfoCard(summaryOf({ compositeReasons: [reason] }))).toBe(true);
    }
  });
});

describe("editorSource", () => {
  it("opens the restored file itself when the browser can show it", () => {
    const job = makeCompletedRestoreJob(makeRestoreMetadata());
    expect(editorSource(job, summaryOf())).toEqual({
      url: "/api/v1/jobs/job-1/download",
      fileName: "grandma_restored.png",
    });
  });

  it("falls back to the viewing copy for a TIFF result", () => {
    const job = { ...makeCompletedRestoreJob(makeRestoreMetadata()), outputFormat: "tiff" };
    const summary = summaryOf({
      downloadNames: {
        restored: "grandma_restored.tif",
        uncolored: "grandma_uncolored.tif",
        before_after: "grandma_before-after.jpg",
        sidecar: "grandma_restore.json",
      },
    });
    expect(editorSource(job, summary)).toEqual({
      url: "/api/v1/jobs/job-1/artifacts/view",
      fileName: "grandma_restored.jpg",
    });
  });
});
