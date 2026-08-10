import { describe, expect, it } from "vitest";
import { translate } from "../i18n";
import { translateStageLabel } from "./jobStageLabels";

function t(locale: "en" | "es") {
  return (key: string, params?: Record<string, string>) => translate(locale, key, params);
}

describe("translateStageLabel", () => {
  it("translates a known stage by its key, ignoring the English label", () => {
    const stage = { key: "encoding_video", label: "Encoding video" };

    expect(translateStageLabel(stage, t("es"))).toBe("Codificando el video");
  });

  it("keeps the same key readable in English too", () => {
    const stage = { key: "encoding_video", label: "Encoding video" };

    expect(translateStageLabel(stage, t("en"))).toBe("Encoding video");
  });

  it("falls back to the backend label for a stage the catalog does not know", () => {
    // Una etapa nueva del servidor no puede mostrar "job.stage.loquesea".
    const stage = { key: "quantizing_weights", label: "Quantizing weights" };

    expect(translateStageLabel(stage, t("es"))).toBe("Quantizing weights");
  });

  it("translates a cleanup pass and keeps the model's proper name", () => {
    const stage = { key: "cleanup_denoise", label: "Cleaning up: UVR DeNoise by FoxJoy" };

    expect(translateStageLabel(stage, t("es"))).toBe("Limpiando: UVR DeNoise by FoxJoy");
    expect(translateStageLabel(stage, t("en"))).toBe("Cleaning up: UVR DeNoise by FoxJoy");
  });

  it("falls back to the model id when the cleanup label has no name in it", () => {
    const stage = { key: "cleanup_dereverb", label: "Cleaning up" };

    expect(translateStageLabel(stage, t("es"))).toBe("Limpiando: dereverb");
  });

  it.each([
    ["probing", "Analizando el video"],
    ["upscaling_frames", "Agrandando los cuadros"],
    ["interpolating_frames", "Interpolando cuadros"],
    ["validating", "Revisando la imagen"],
    ["decoding", "Decodificando el audio"],
    ["separating", "Separando pistas"],
    ["restoring", "Restaurando"],
    ["mastering", "Dando el acabado"],
    ["finalizing", "Escribiendo el resultado"],
    ["generating", "Generando"],
  ])("translates the %s stage produced by the backend", (key, expected) => {
    expect(translateStageLabel({ key, label: "whatever the server said" }, t("es"))).toBe(expected);
  });
});
