import { describe, expect, it } from "vitest";
import { translate } from "../i18n";
import { activeStageCount, translateStageLabel } from "./jobStageLabels";

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

  it.each([
    ["restore_descreen", "Removing print pattern"],
    ["restore_repair_detect", "Detecting damage"],
    ["restore_repair_fill", "Filling damage"],
    ["restore_deblock", "Removing JPEG artifacts"],
    ["restore_denoise", "Reducing noise"],
    ["restore_tone", "Fixing colors and tone"],
    ["restore_faces", "Restoring faces"],
    ["restore_colorize", "Colorizing"],
    ["saving", "Saving"],
  ])("names the %s restoration stage in English", (key, expected) => {
    expect(translateStageLabel({ key, label: "whatever the server said" }, t("en"))).toBe(expected);
  });

  it("counts the damaged areas while they are being filled", () => {
    const stage = { key: "restore_repair_fill", label: "Filling damage" };

    expect(translateStageLabel(stage, t("en"), { done: 3, total: 7 })).toBe("Filling damage (3/7 areas)");
    expect(translateStageLabel(stage, t("es"), { done: 3, total: 7 })).toBe("Rellenando daños (3/7 áreas)");
  });

  it("counts the faces while they are being restored", () => {
    const stage = { key: "restore_faces", label: "Restoring faces" };

    expect(translateStageLabel(stage, t("en"), { done: 1, total: 2 })).toBe("Restoring faces (1/2)");
  });

  it("ignores a count on a stage whose text has no counter", () => {
    const stage = { key: "restore_denoise", label: "Reducing noise" };

    expect(translateStageLabel(stage, t("en"), { done: 4, total: 12 })).toBe("Reducing noise");
  });

  it("does not reuse the audio restoring stage for photos", () => {
    expect(translateStageLabel({ key: "restoring", label: "Restoring" }, t("en"))).toBe("Restoring");
    expect(translateStageLabel({ key: "restore_tone", label: "x" }, t("en"))).not.toBe("Restoring");
  });
});

describe("activeStageCount", () => {
  it("reads the counter of the stage that is running", () => {
    const metadata = { stage: "restore_faces", framesDone: 1, framesTotal: 3 };

    expect(activeStageCount("restore_faces", metadata)).toEqual({ done: 1, total: 3 });
  });

  it("does not lend the running stage's counter to another stage", () => {
    const metadata = { stage: "restore_denoise", framesDone: 5, framesTotal: 12 };

    expect(activeStageCount("restore_faces", metadata)).toBeNull();
  });

  it("has no counter before the stage reports a total", () => {
    const metadata = { stage: "restore_repair_fill", framesDone: 0, framesTotal: null };

    expect(activeStageCount("restore_repair_fill", metadata)).toBeNull();
    expect(activeStageCount("restore_repair_fill", undefined)).toBeNull();
  });

  it("rejects an impossible counter", () => {
    const metadata = { stage: "restore_faces", framesDone: 4, framesTotal: 3 };

    expect(activeStageCount("restore_faces", metadata)).toBeNull();
  });
});
