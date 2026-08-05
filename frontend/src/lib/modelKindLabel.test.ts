import { describe, expect, it } from "vitest";
import { modelKindLabel } from "./modelKindLabel";

// El identificador interno de un tipo de modelo no es un nombre: "builtin-ncnn"
// aparecia tal cual en la lista de modelos instalados, donde el usuario espera
// leer con que corre el modelo.
describe("modelKindLabel", () => {
  it("names the Vulkan runtime instead of showing the enum", () => {
    expect(modelKindLabel("builtin-ncnn")).toBe("NCNN (Vulkan)");
  });

  it("names ONNX", () => {
    expect(modelKindLabel("onnx")).toBe("ONNX");
  });

  it("falls back to the raw value for a kind it does not know", () => {
    // Inventar un nombre lindo para algo desconocido seria peor que mostrar el
    // identificador: al menos ese se puede buscar.
    expect(modelKindLabel("algo-nuevo")).toBe("algo-nuevo");
  });
});
