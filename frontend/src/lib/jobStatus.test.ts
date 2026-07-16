import { describe, expect, it } from "vitest";
import { isTerminalJobStatus, jobKindLabel } from "./jobStatus";

describe("isTerminalJobStatus", () => {
  it.each([
    ["queued", false],
    ["running", false],
    ["completed", true],
    ["failed", true],
  ] as const)("returns %s -> %s", (status, expected) => {
    expect(isTerminalJobStatus(status)).toBe(expected);
  });
});

describe("jobKindLabel", () => {
  it("labels an image job", () => {
    expect(jobKindLabel("image")).toBe("Image");
  });

  it("labels a video job", () => {
    expect(jobKindLabel("video")).toBe("Video");
  });
});
