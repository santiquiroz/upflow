import { describe, expect, it } from "vitest";
import { joinAsChoices, selectableSectionKeys } from "./selectionHint";

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
