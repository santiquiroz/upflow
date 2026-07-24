import { describe, expect, it } from "vitest";
import { formatDuration } from "./formatDuration";

describe("formatDuration", () => {
  it("returns an em dash when startedAt is missing", () => {
    expect(formatDuration(null, "2026-01-01T00:00:42Z")).toBe("—");
  });

  it("returns an em dash when finishedAt is missing", () => {
    expect(formatDuration("2026-01-01T00:00:00Z", null)).toBe("—");
  });

  it("returns an em dash when both timestamps are missing", () => {
    expect(formatDuration(null, null)).toBe("—");
  });

  it("formats a sub-minute duration as seconds", () => {
    expect(formatDuration("2026-01-01T00:00:00Z", "2026-01-01T00:00:42Z")).toBe("42s");
  });

  it("formats a multi-minute duration as minutes and seconds", () => {
    expect(formatDuration("2026-01-01T00:00:00Z", "2026-01-01T00:03:12Z")).toBe("3m 12s");
  });

  it("formats a multi-hour duration as hours, minutes, and seconds", () => {
    expect(formatDuration("2026-01-01T00:00:00Z", "2026-01-01T01:04:12Z")).toBe("1h 04m 12s");
  });
});
