import { describe, expect, it } from "vitest";
import { isCancellableJobStatus, isTerminalJobStatus, jobKindLabel } from "./jobStatus";

describe("isTerminalJobStatus", () => {
  it.each([
    ["queued", false],
    ["running", false],
    ["completed", true],
    ["failed", true],
    ["cancelled", true],
  ] as const)("returns %s -> %s", (status, expected) => {
    expect(isTerminalJobStatus(status)).toBe(expected);
  });
});

describe("isCancellableJobStatus", () => {
  it.each([
    ["queued", true],
    ["running", true],
    ["completed", false],
    ["failed", false],
    ["cancelled", false],
  ] as const)("returns %s -> %s", (status, expected) => {
    expect(isCancellableJobStatus(status)).toBe(expected);
  });
});

describe("jobKindLabel", () => {
  it("labels an image job", () => {
    expect(jobKindLabel("image")).toBe("Image");
  });

  it("labels a video job", () => {
    expect(jobKindLabel("video")).toBe("Video");
  });

  it("labels an audio job", () => {
    expect(jobKindLabel("audio")).toBe("Audio");
  });
});
