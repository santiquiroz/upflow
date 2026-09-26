import { describe, expect, it } from "vitest";
import { translate } from "../../../i18n";
import { errorText, translateOr, type Translate } from "./cctvText";

const t: Translate = (key, params) => translate("en", key as Parameters<typeof translate>[1], params);

describe("translateOr", () => {
  it("translates a known key with its parameters", () => {
    expect(translateOr(t, "cctv.ai.cpuBlocked", "fallback", { eta: "2 h" })).toBe(
      "AI enhancement needs a GPU. On this computer's CPU it would take about 2 h.",
    );
  });

  it("falls back for an unknown key or none", () => {
    expect(translateOr(t, "cctv.error.nope", "The server said no.")).toBe("The server said no.");
    expect(translateOr(t, null, "The server said no.")).toBe("The server said no.");
  });
});

describe("errorText", () => {
  it("keeps the server's sentence when the key needs parameters the error doesn't carry", () => {
    const message = "AI enhancement needs a GPU; it is not available on the CPU. On the CPU this clip would take about 2.0 h.";

    expect(errorText(t, { key: "cctv.ai.cpuBlocked", message })).toBe(message);
  });
});
