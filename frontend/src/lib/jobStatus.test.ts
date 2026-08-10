import { describe, expect, it } from "vitest";
import { isCancellableJobStatus, isTerminalJobStatus, jobKindLabelKey, jobStatusLabelKey } from "./jobStatus";
import { en } from "../i18n/en";

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

describe("jobKindLabelKey", () => {
  it.each([
    "image",
    "video",
    "audio",
    "generation",
    "transcribe",
    "download",
    "shape3d",
  ] as const)("has a catalog entry for the %s family", (kind) => {
    // El nombre de la familia se lee traducido: una clave sin entrada mostraria
    // "job.kind.download" en la pantalla.
    expect(en[jobKindLabelKey(kind) as keyof typeof en]).toBeTruthy();
  });
});

describe("jobStatusLabelKey", () => {
  it.each(["queued", "running", "completed", "failed", "cancelled"] as const)(
    "has a catalog entry for %s",
    (status) => {
      expect(en[jobStatusLabelKey(status) as keyof typeof en]).toBeTruthy();
    },
  );

  it("maps running to the processing copy, which is what the queue already said", () => {
    expect(jobStatusLabelKey("running")).toBe("job.status.processing");
  });
});
