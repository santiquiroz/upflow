import { describe, expect, it } from "vitest";
import type { ModelResponse, SupportedModelResponse } from "../../lib/apiTypes";
import { correctScaleFor, scalesForModel } from "./scalesForModel";

// El backend RECHAZA una escala que el modelo no soporta
// (`job_manager._resolve_builtin_model`: "Model X supports only scales [...]").
// Ofrecerla en pantalla significa que el usuario sube el archivo entero y recien
// ahi se entera. El dato para evitarlo ya viaja en /engine.

const CATALOG: SupportedModelResponse[] = [
  { key: "realesrgan-x4plus", label: "x4plus", category: "general", description: "", scales: [2, 3, 4] },
  { key: "realesr-animevideov3-x2", label: "v3 x2", category: "anime", description: "", scales: [2] },
];

function model(id: string): ModelResponse {
  return {
    id,
    name: id,
    kind: "builtin-ncnn",
    source: "builtin",
    scale: null,
    arch: "esrgan",
    sizeBytes: 0,
    status: "installed",
    error: null,
  };
}

describe("scalesForModel", () => {
  it("offers only what the chosen model supports", () => {
    expect(scalesForModel(model("realesr-animevideov3-x2"), CATALOG, [2, 3, 4])).toEqual([2]);
  });

  it("offers the whole range for a model that takes it", () => {
    expect(scalesForModel(model("realesrgan-x4plus"), CATALOG, [2, 3, 4])).toEqual([2, 3, 4]);
  });

  it("falls back to the engine range when no model is chosen yet", () => {
    expect(scalesForModel(null, CATALOG, [2, 3, 4])).toEqual([2, 3, 4]);
  });

  it("falls back to the engine range for a model the catalog does not describe", () => {
    // Un ONNX instalado desde Hugging Face no esta en el catalogo builtin.
    expect(scalesForModel(model("alguien/su-modelo"), CATALOG, [2, 3, 4])).toEqual([2, 3, 4]);
  });

  it("never offers a scale the engine disallows, even if the model claims it", () => {
    expect(scalesForModel(model("realesrgan-x4plus"), CATALOG, [2, 4])).toEqual([2, 4]);
  });
});

describe("correctScaleFor", () => {
  it("keeps the current scale when the model still supports it", () => {
    expect(correctScaleFor(2, [2, 3, 4])).toBe(2);
  });

  it("moves to the closest supported scale instead of leaving an invalid one", () => {
    // Cambiar de x4plus (en 4x) a un modelo que solo hace 2x no puede dejar 4x
    // seleccionado: es exactamente la combinacion que el backend rechaza.
    expect(correctScaleFor(4, [2])).toBe(2);
  });

  it("returns null when there is nothing to choose", () => {
    expect(correctScaleFor(4, [])).toBeNull();
  });
});
