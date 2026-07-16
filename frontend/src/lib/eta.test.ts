import { describe, expect, it } from "vitest";
import { estimateEta, formatEta, type EtaSample } from "./eta";

function sample(progress: number, t: number): EtaSample {
  return { progress, t };
}

describe("estimateEta", () => {
  it("returns null with zero samples", () => {
    expect(estimateEta([])).toBeNull();
  });

  it("returns null with a single sample", () => {
    expect(estimateEta([sample(0.1, 0)])).toBeNull();
  });

  it("computes ETA seconds from a steady rate", () => {
    const samples = [sample(0.2, 0), sample(0.4, 10_000)];

    // rate = 0.02 progress/s, remaining = 0.6 -> eta = 30s
    expect(estimateEta(samples)).toBeCloseTo(30, 5);
  });

  it("returns null when progress has not advanced (stalled)", () => {
    const samples = [sample(0.5, 0), sample(0.5, 5_000)];

    expect(estimateEta(samples)).toBeNull();
  });

  it("returns null when progress regresses", () => {
    const samples = [sample(0.6, 0), sample(0.4, 5_000)];

    expect(estimateEta(samples)).toBeNull();
  });

  it("returns null when the latest progress is already complete", () => {
    const samples = [sample(0.5, 0), sample(1, 5_000)];

    expect(estimateEta(samples)).toBeNull();
  });

  it("returns null when the window's elapsed time is zero", () => {
    const samples = [sample(0.2, 1_000), sample(0.4, 1_000)];

    expect(estimateEta(samples)).toBeNull();
  });

  it("uses only the most recent window of samples, ignoring older history", () => {
    const samples = [
      sample(0.0, 0),
      sample(0.5, 1_000), // fast early rate: 0.5/s over the full history
      sample(0.52, 21_000),
      sample(0.54, 41_000),
      sample(0.56, 61_000),
      sample(0.58, 81_000), // recent rate: 0.01/s over the last 5 samples
    ];

    // recent window = indices 1..5 -> delta progress 0.08 over 80s -> rate 0.001/s
    const eta = estimateEta(samples);
    expect(eta).not.toBeNull();
    expect(eta).toBeCloseTo((1 - 0.58) / 0.001, 5);
  });
});

describe("formatEta", () => {
  it("formats minutes and seconds", () => {
    expect(formatEta(150)).toBe("~2 min 30 s");
  });

  it("formats a whole number of minutes without a seconds suffix", () => {
    expect(formatEta(180)).toBe("~3 min");
  });

  it("formats sub-minute durations as seconds", () => {
    expect(formatEta(45)).toBe("~45 s");
  });

  it("rounds up to the next minute instead of showing 60 s", () => {
    expect(formatEta(59.6)).toBe("~1 min");
  });

  it("shows a coarse label for near-instant completion", () => {
    expect(formatEta(0.4)).toBe("<1 min");
  });
});
