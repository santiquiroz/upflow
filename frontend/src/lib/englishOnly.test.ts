import { describe, expect, it } from "vitest";
import { isEnglishOnly } from "./englishOnly";

describe("isEnglishOnly", () => {
  it.each([
    "openai/whisper-tiny.en",
    "onnx-community/whisper-tiny.en_timestamped",
    "asr--onnx-community--whisper-small.en",
    "WHISPER-BASE.EN",
  ])("reconoce %s como solo inglés", (id) => {
    expect(isEnglishOnly(id)).toBe(true);
  });

  it.each([
    "openai/whisper-large-v3",
    "onnx-community/whisper-tiny",
    "onnx-community/whisper-base_timestamped",
  ])("deja pasar %s como multilingüe", (id) => {
    expect(isEnglishOnly(id)).toBe(false);
  });

  it("no confunde una palabra que contiene 'en'", () => {
    // Marcar de más dejaría a un modelo multilingüe sin poder elegir idioma.
    expect(isEnglishOnly("empresa/whisper-entrenado")).toBe(false);
    expect(isEnglishOnly("openai/whisper-tiny.english-ish")).toBe(false);
  });
});
