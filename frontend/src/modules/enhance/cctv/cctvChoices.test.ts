import { describe, expect, it } from "vitest";
import {
  buildCctvJobRequest,
  initialChoices,
  withLane,
  withNoOsd,
  withOsdBoxes,
  withOsdConfirmed,
  withPreset,
  withCaseDetails,
  withTrim,
} from "./cctvChoices";
import { EMPTY_CASE_DETAILS } from "./cctvCase";
import { ANALYSIS, PRESETS_RESPONSE } from "./cctvFixtures";

describe("initialChoices", () => {
  it("starts in the classic lane with the preset the diagnosis suggested", () => {
    const choices = initialChoices(ANALYSIS, PRESETS_RESPONSE);

    expect(choices.lane).toBe("classic");
    expect(choices.task).toBe("clarify");
    expect(choices.presetId).toBe("night_ir");
    expect(Object.keys(choices.steps)).toEqual(["deblock", "gray"]);
  });

  it("starts with empty case details", () => {
    expect(initialChoices(ANALYSIS, PRESETS_RESPONSE).caseDetails).toEqual(EMPTY_CASE_DETAILS);
  });

  it("falls back to the first preset when the suggestion is unknown", () => {
    const choices = initialChoices({ ...ANALYSIS, suggestedPreset: "moon" }, PRESETS_RESPONSE);

    expect(choices.presetId).toBe("day");
  });

  it("asks for an on-screen text decision instead of assuming one", () => {
    const choices = initialChoices(ANALYSIS, PRESETS_RESPONSE);

    expect(choices.noOsd).toBe(false);
    expect(choices.osdBoxesConfirmed).toBe(false);
  });

  it("suggests the Hikvision on-screen text boxes on the stored frame, still unconfirmed", () => {
    const choices = initialChoices(ANALYSIS, PRESETS_RESPONSE);

    expect(choices.osdBoxes).toEqual([
      [19, 22, 384, 64],
      [701, 994, 240, 64],
    ]);
    expect(choices.trim).toBeNull();
  });
});

describe("on-screen text decision", () => {
  it("confirms the boxes and drops the confirmation when a box changes", () => {
    const confirmed = withOsdConfirmed(initialChoices(ANALYSIS, PRESETS_RESPONSE));

    expect(confirmed.osdBoxesConfirmed).toBe(true);
    expect(withOsdBoxes(confirmed, [[0, 0, 20, 20]]).osdBoxesConfirmed).toBe(false);
  });

  it("can't confirm an empty list of boxes", () => {
    const empty = withOsdBoxes(initialChoices(ANALYSIS, PRESETS_RESPONSE), []);

    expect(withOsdConfirmed(empty).osdBoxesConfirmed).toBe(false);
  });

  it("clears the confirmation when switching to no on-screen text", () => {
    const confirmed = withOsdConfirmed(initialChoices(ANALYSIS, PRESETS_RESPONSE));

    const none = withNoOsd(confirmed, true);

    expect(none.noOsd).toBe(true);
    expect(none.osdBoxesConfirmed).toBe(false);
    expect(withOsdConfirmed(none).noOsd).toBe(false);
  });
});

describe("changing lane and preset", () => {
  it("rebuilds the steps from the preset chain of the new lane", () => {
    const classic = initialChoices(ANALYSIS, PRESETS_RESPONSE);

    const ai = withLane(classic, "ai", ANALYSIS, PRESETS_RESPONSE);

    expect(ai.task).toBe("enhance");
    expect(Object.keys(ai.steps)).toEqual(["ai_deblock", "gray"]);
    expect(classic.lane).toBe("classic");
  });

  it("replaces the steps with the chosen preset", () => {
    const choices = withPreset(initialChoices(ANALYSIS, PRESETS_RESPONSE), "day", ANALYSIS, PRESETS_RESPONSE);

    expect(choices.presetId).toBe("day");
    expect(Object.keys(choices.steps)).toEqual(["aspect", "deinterlace", "deblock", "denoise"]);
  });
});

describe("buildCctvJobRequest", () => {
  it("sends the session token, the task, the preset and the ordered steps", () => {
    const choices = { ...initialChoices(ANALYSIS, PRESETS_RESPONSE), noOsd: true };

    expect(buildCctvJobRequest("tok-1", choices, PRESETS_RESPONSE)).toEqual({
      token: "tok-1",
      task: "clarify",
      preset: "night_ir",
      steps: [
        { id: "deblock", params: { filter: "deblock", filter_type: "strong", block: 8 } },
        { id: "gray", params: { filter: "gray" } },
      ],
      osdBoxes: [],
      osdBoxesConfirmed: false,
      noOsd: true,
      trim: null,
    });
  });

  it("sends the trim as first and last frame", () => {
    const choices = withTrim({ ...initialChoices(ANALYSIS, PRESETS_RESPONSE), noOsd: true }, [25, 99]);

    expect(buildCctvJobRequest("tok-1", choices, PRESETS_RESPONSE).trim).toEqual([25, 99]);
  });

  it("sends the case details the operator filled in", () => {
    const base = { ...initialChoices(ANALYSIS, PRESETS_RESPONSE), noOsd: true };
    const choices = withCaseDetails(base, {
      ...EMPTY_CASE_DETAILS,
      caseLabel: "2026-114",
      recorderMake: "Hikvision",
      clockOffset: "-12",
    });

    const request = buildCctvJobRequest("tok-1", choices, PRESETS_RESPONSE);

    expect(request.caseLabel).toBe("2026-114");
    expect(request.acquisition).toEqual({ recorderMake: "Hikvision", clockOffsetSeconds: -12 });
    expect(request).not.toHaveProperty("operatorName");
  });

  it("sends confirmed boxes only when there is on-screen text", () => {
    const choices = {
      ...initialChoices(ANALYSIS, PRESETS_RESPONSE),
      osdBoxes: [[0, 0, 320, 40] as const],
      osdBoxesConfirmed: true,
    };

    const request = buildCctvJobRequest("tok-1", choices, PRESETS_RESPONSE);

    expect(request.osdBoxes).toEqual([[0, 0, 320, 40]]);
    expect(request.osdBoxesConfirmed).toBe(true);
    expect(request.noOsd).toBe(false);
  });
});
