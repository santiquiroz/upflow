import { describe, expect, it } from "vitest";
import {
  audioSourceFormat,
  convertibleFileCount,
  isLossyFormat,
  joinAsChoices,
  losesQualityIrreversibly,
  selectableSectionKeys,
} from "./selectionHint";

describe("selectableSectionKeys", () => {
  it("lists the sections in the order they are rendered", () => {
    expect(selectableSectionKeys({ masteringAvailable: true, restoreAvailable: true })).toEqual([
      "audio.section.denoise",
      "audio.section.mastering",
      "audio.section.restore",
      "voice.sectionTitle",
    ]);
  });

  it("leaves out the sections the panel is not rendering", () => {
    expect(selectableSectionKeys({ masteringAvailable: false, restoreAvailable: false })).toEqual([
      "audio.section.denoise",
      "voice.sectionTitle",
    ]);
  });
});

describe("joinAsChoices", () => {
  it("joins with the disjunction of each language, not a hardcoded word", () => {
    expect(joinAsChoices(["Denoise", "Mastering", "Voice"], "en")).toBe(
      "Denoise, Mastering, or Voice",
    );
    expect(joinAsChoices(["Ruido", "Acabado", "Voz"], "es")).toBe("Ruido, Acabado o Voz");
  });
});

// ---------------------------------------------------------------------------
// Conversion de formato. Espeja UNAMBIGUOUS_SOURCE_FORMATS del backend: si los
// dos lados no coinciden, la UI habilita un envio que la API rechaza (o al
// reves, esconde uno valido).
// ---------------------------------------------------------------------------

describe("audioSourceFormat", () => {
  it("resuelve las extensiones que determinan el codec sin ambiguedad", () => {
    expect(audioSourceFormat("cancion.flac")).toBe("flac");
    expect(audioSourceFormat("CANCION.FLAC")).toBe("flac");
    expect(audioSourceFormat("cancion.wav")).toBe("wav");
    expect(audioSourceFormat("cancion.mp3")).toBe("mp3");
  });

  it("no afirma nada de un contenedor que admite mas de un codec", () => {
    // .m4a puede traer ALAC o AAC: convertirlo a m4a NO es un trabajo vacio.
    expect(audioSourceFormat("cancion.m4a")).toBeNull();
    expect(audioSourceFormat("cancion.opus")).toBeNull();
    expect(audioSourceFormat("cancion")).toBeNull();
  });
});

describe("convertibleFileCount", () => {
  it("cuenta solo los archivos que cambiarian de formato", () => {
    const files = [{ name: "a.flac" }, { name: "b.wav" }, { name: "c.flac" }];

    expect(convertibleFileCount(files, "flac")).toBe(1);
    expect(convertibleFileCount(files, "mp3")).toBe(3);
  });

  it("cuenta los ambiguos como convertibles", () => {
    expect(convertibleFileCount([{ name: "a.m4a" }], "m4a")).toBe(1);
  });
});

describe("losesQualityIrreversibly", () => {
  it("avisa cuando se puede afirmar que el origen no tiene perdida", () => {
    expect(losesQualityIrreversibly([{ name: "a.flac" }], "mp3")).toBe(true);
    expect(losesQualityIrreversibly([{ name: "a.wav" }], "m4a")).toBe(true);
  });

  it("no avisa cuando el destino conserva todo", () => {
    expect(losesQualityIrreversibly([{ name: "a.flac" }], "wav")).toBe(false);
    expect(losesQualityIrreversibly([{ name: "a.wav" }], "flac")).toBe(false);
  });

  it("no avisa de una perdida que no se puede probar", () => {
    expect(losesQualityIrreversibly([{ name: "a.mp3" }], "m4a")).toBe(false);
    expect(losesQualityIrreversibly([{ name: "a.m4a" }], "mp3")).toBe(false);
  });
});

describe("isLossyFormat", () => {
  it("separa los destinos con perdida de los que no la tienen", () => {
    expect(isLossyFormat("mp3")).toBe(true);
    expect(isLossyFormat("m4a")).toBe(true);
    expect(isLossyFormat("flac")).toBe(false);
    expect(isLossyFormat("wav")).toBe(false);
  });
});
