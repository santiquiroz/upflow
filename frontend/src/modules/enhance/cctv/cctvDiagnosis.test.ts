import { describe, expect, it } from "vitest";
import { translate } from "../../../i18n";
import { en } from "../../../i18n/en";
import { audioLabel, diagnosisRows, diagnosisWarnings } from "./cctvDiagnosis";
import { ANALYSIS } from "./cctvFixtures";

const t = (key: string, params?: Record<string, string | number>) => translate("en", key, params);

function valueOf(labelKey: string, analysis = ANALYSIS): string | undefined {
  return diagnosisRows(t, analysis).find((row) => row.labelKey === labelKey)?.value;
}

describe("diagnosisRows", () => {
  it("shows the stored resolution and the real aspect of a lite recording", () => {
    expect(valueOf("cctv.diag.stored")).toBe("960 × 1080");
    expect(valueOf("cctv.diag.display")).toBe("1920 × 1080");
  });

  it("compares the declared and the measured frame rate", () => {
    expect(valueOf("cctv.diag.fps")).toBe("25 / 12.50");
    expect(valueOf("cctv.diag.vfr")).toBe(en["cctv.diag.yes"]);
  });

  it("names the field order of interlaced video", () => {
    expect(valueOf("cctv.diag.interlaced")).toBe(en["cctv.diag.interlaced.tff"]);
  });

  it("reports night footage and clipped highlights", () => {
    expect(valueOf("cctv.diag.nightIr")).toBe(en["cctv.diag.yes"]);
    expect(valueOf("cctv.diag.clipped")).toBe("7.0%");
  });

  it("uses dashes for what the analysis could not measure", () => {
    const bare = { ...ANALYSIS, frameIndex: null, gop: null, quality: null, audio: [] };

    expect(valueOf("cctv.diag.frames", bare)).toBe("—");
    expect(valueOf("cctv.diag.blocking", bare)).toBe("—");
    expect(valueOf("cctv.diag.interlaced", bare)).toBe("—");
    expect(valueOf("cctv.diag.audio", bare)).toBe(en["cctv.diag.noAudio"]);
  });
});

describe("audioLabel", () => {
  it("names G.711 with its sample rate", () => {
    expect(audioLabel({ codec: "pcm_alaw", sampleRate: 8000, channels: 1, family: "g711" })).toBe("G.711 A-law, 8 kHz");
  });
});

describe("diagnosisWarnings", () => {
  it("drops the duplicated lite warning and names each missing filter", () => {
    expect(diagnosisWarnings(t, ANALYSIS)).toEqual([
      en["cctv.lite"],
      en["cctv.warning.monochrome"],
      "This ffmpeg build doesn't include fspp, so this step is off.",
    ]);
  });

  it("keeps an unknown key visible instead of hiding it", () => {
    expect(diagnosisWarnings(t, { ...ANALYSIS, warnings: ["cctv.warning.new"] })).toEqual(["cctv.warning.new"]);
  });
});
