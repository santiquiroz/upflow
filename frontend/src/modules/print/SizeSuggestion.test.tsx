import { describe, expect, it } from "vitest";
import { objectHintFromFileName } from "./SizeSuggestion";

// En el carril de foto no hay descripción: el nombre del archivo es la única
// señal de qué es el objeto. Y la mitad de las veces no dice nada — una foto de
// celular se llama IMG_20260808_143255.jpg. Preguntarle al modelo cuánto mide
// "IMG 20260808" devuelve un número igual, y ese número es una invención con
// apariencia de dato.

describe("objectHintFromFileName", () => {
  it("reads the object out of a descriptive name", () => {
    expect(objectHintFromFileName("coffee-mug.jpg")).toBe("coffee mug");
  });

  it("takes underscores and spaces as word breaks", () => {
    expect(objectHintFromFileName("taza_de_cafe.png")).toBe("taza de cafe");
  });

  it("says nothing for a camera name", () => {
    expect(objectHintFromFileName("IMG_20260808_143255.jpg")).toBe("");
  });

  it("says nothing for a screenshot name", () => {
    expect(objectHintFromFileName("Screenshot 2026-08-08 143255.png")).toBe("");
  });

  it("says nothing for a name that is only digits", () => {
    expect(objectHintFromFileName("20260808143255.jpeg")).toBe("");
  });

  it("says nothing for an empty name", () => {
    expect(objectHintFromFileName("")).toBe("");
  });

  it("keeps the object and drops the camera noise around it", () => {
    expect(objectHintFromFileName("IMG_20260808_taza.jpg")).toBe("taza");
  });

  it("keeps a designation that mixes letters and digits", () => {
    // "m3" es exactamente el tipo de dato que decide la medida: perderlo por
    // tener un digito seria tirar la mejor señal del nombre.
    expect(objectHintFromFileName("tornillo-m3.jpg")).toBe("tornillo m3");
  });

  it("survives a name with no extension", () => {
    expect(objectHintFromFileName("coffee mug")).toBe("coffee mug");
  });

  it("keeps accented words whole", () => {
    expect(objectHintFromFileName("cuchara-pequeña.jpg")).toBe("cuchara pequeña");
  });
});
