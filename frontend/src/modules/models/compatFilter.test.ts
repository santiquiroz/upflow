import { describe, expect, it } from "vitest";
import { matchesCompatFilter } from "./compatFilter";

// El filtro es el mismo en los tres buscadores (generación, escalado y
// transcripción): si cambia acá, cambia en todos.
describe("matchesCompatFilter", () => {
  it("shows everything under 'all', including what cannot be installed", () => {
    for (const compat of ["ready_onnx", "needs_conversion", "single_file", "gated", "incompatible"]) {
      expect(matchesCompatFilter({ compat } as never, "all")).toBe(true);
    }
  });

  it("'ready' keeps only what runs as-is", () => {
    expect(matchesCompatFilter({ compat: "ready_onnx" } as never, "ready")).toBe(true);
    expect(matchesCompatFilter({ compat: "needs_conversion" } as never, "ready")).toBe(false);
    expect(matchesCompatFilter({ compat: "single_file" } as never, "ready")).toBe(false);
  });

  it("'conversion' keeps both flavours that go through the converter", () => {
    expect(matchesCompatFilter({ compat: "needs_conversion" } as never, "conversion")).toBe(true);
    expect(matchesCompatFilter({ compat: "single_file" } as never, "conversion")).toBe(true);
    expect(matchesCompatFilter({ compat: "ready_onnx" } as never, "conversion")).toBe(false);
  });

  it("hides what cannot be installed from the narrowed filters", () => {
    for (const filter of ["ready", "conversion"] as const) {
      expect(matchesCompatFilter({ compat: "gated" } as never, filter)).toBe(false);
      expect(matchesCompatFilter({ compat: "incompatible" } as never, filter)).toBe(false);
    }
  });
});
