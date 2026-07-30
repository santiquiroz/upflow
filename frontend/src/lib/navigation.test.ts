import { describe, expect, it } from "vitest";
import { NAV_ENTRIES } from "./navigation";

describe("NAV_ENTRIES", () => {
  it("exposes the transcription page in primary navigation", () => {
    expect(NAV_ENTRIES).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "Transcribe",
          path: "/transcribe",
        }),
      ]),
    );
  });
});
