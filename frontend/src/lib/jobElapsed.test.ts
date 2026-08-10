import { describe, expect, it } from "vitest";
import { formatElapsed } from "./jobElapsed";

const START = "2026-01-01T00:00:00Z";

describe("formatElapsed", () => {
  it("returns null while the job has not started", () => {
    expect(formatElapsed(null, Date.parse(START))).toBeNull();
  });

  it("formats seconds", () => {
    expect(formatElapsed(START, Date.parse("2026-01-01T00:00:42Z"))).toBe("42s");
  });

  it("formats minutes and seconds", () => {
    expect(formatElapsed(START, Date.parse("2026-01-01T00:03:12Z"))).toBe("3m 12s");
  });

  it("formats hours", () => {
    expect(formatElapsed(START, Date.parse("2026-01-01T02:05:07Z"))).toBe("2h 05m 07s");
  });

  it("returns null when the clock is behind the start (never a negative time)", () => {
    expect(formatElapsed(START, Date.parse("2025-12-31T23:59:00Z"))).toBeNull();
  });

  it("returns null for an unparseable timestamp", () => {
    expect(formatElapsed("not a date", Date.now())).toBeNull();
  });
});
