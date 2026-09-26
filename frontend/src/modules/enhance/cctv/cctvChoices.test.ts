import { describe, expect, it } from "vitest";
import {
  buildCctvJobRequest,
  initialChoices,
  withLane,
  withNoOsd,
  withOsdBoxes,
  withOsdConfirmed,
  withPreset,
  withAiUpscale,
  withCaseDetails,
  withRedaction,
  withRoi,
  withTask,
  withTrim,
} from "./cctvChoices";
import { EMPTY_REDACTION, withTrackAdded } from "./cctvRedaction";
import { EMPTY_ROI } from "./cctvRoi";
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
    const choices = withTrim({ ...initialChoices(ANALYSIS, PRESETS_RESPONSE), noOsd: true }, [25, 99], 750);

    expect(buildCctvJobRequest("tok-1", choices, PRESETS_RESPONSE).trim).toEqual([25, 99]);
  });

  it("widening the trim of a redacted copy keeps every new frame hidden", () => {
    const redact = withTask(initialChoices(ANALYSIS, PRESETS_RESPONSE), "redact", ANALYSIS, PRESETS_RESPONSE);
    const trimmed = withTrim(redact, [100, 200], 750);
    const boxed = withRedaction(trimmed, withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 150, [100, 200]));

    const widened = buildCctvJobRequest("tok-1", withTrim(boxed, [0, 300], 750), PRESETS_RESPONSE);
    const untrimmed = buildCctvJobRequest("tok-1", withTrim(boxed, null, 750), PRESETS_RESPONSE);

    expect(widened.trim).toEqual([0, 300]);
    expect(widened.redaction?.tracks[0]).toMatchObject({ firstFrame: 0, lastFrame: 300 });
    expect(untrimmed.redaction?.tracks[0]).toMatchObject({ firstFrame: 0, lastFrame: 749 });
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

describe("AI upscale", () => {
  const aiChoices = () => withLane(initialChoices(ANALYSIS, PRESETS_RESPONSE), "ai", ANALYSIS, PRESETS_RESPONSE);

  it("adds the AI upscale step in catalog order with its model and scale", () => {
    const choices = withAiUpscale({ ...aiChoices(), noOsd: true }, { modelId: "realesrgan-x4plus", scale: 4 });

    const request = buildCctvJobRequest("tok-1", choices, PRESETS_RESPONSE);

    expect(request.steps.map((step) => step.id)).toEqual(["ai_deblock", "gray", "ai_upscale"]);
    expect(request.steps[2]).toEqual({ id: "ai_upscale", params: { filter: "onnx_upscale" } });
    expect(request.modelId).toBe("realesrgan-x4plus");
    expect(request.scale).toBe(4);
  });

  it("sends no model or scale with None", () => {
    const request = buildCctvJobRequest("tok-1", { ...aiChoices(), noOsd: true }, PRESETS_RESPONSE);

    expect(request).not.toHaveProperty("modelId");
    expect(request).not.toHaveProperty("scale");
  });

  it("drops the model when going back to the classic lane", () => {
    const choices = withAiUpscale(aiChoices(), { modelId: "realesrgan-x4plus", scale: 2 });

    expect(withLane(choices, "classic", ANALYSIS, PRESETS_RESPONSE).aiUpscale).toBeNull();
  });
});

describe("multi-frame still", () => {
  const ROI = { ...EMPTY_ROI, first: 10, last: 39, reference: 20, box: [100, 200, 60, 14] as const };

  it("keeps only deinterlace and deblock from the classic preset chain", () => {
    const choices = withTask(initialChoices(ANALYSIS, PRESETS_RESPONSE), "roi_fusion", ANALYSIS, PRESETS_RESPONSE);

    expect(Object.keys(choices.steps)).toEqual(["deblock"]);
  });

  it("uses the classic deblock even from the AI lane", () => {
    const ai = withLane(initialChoices(ANALYSIS, PRESETS_RESPONSE), "ai", ANALYSIS, PRESETS_RESPONSE);

    const choices = withTask(ai, "roi_fusion", ANALYSIS, PRESETS_RESPONSE);

    expect(Object.keys(choices.steps)).toEqual(["deblock"]);
    expect(choices.lane).toBe("ai");
  });

  it("sends the region with no trim, no on-screen text and no AI model", () => {
    const base = withTask(initialChoices(ANALYSIS, PRESETS_RESPONSE), "roi_fusion", ANALYSIS, PRESETS_RESPONSE);
    const choices = withAiUpscale(withTrim(withRoi({ ...base, noOsd: true }, ROI), [0, 99], 750), {
      modelId: "realesrgan-x4plus",
      scale: 2,
    });

    expect(buildCctvJobRequest("tok-1", choices, PRESETS_RESPONSE)).toEqual({
      token: "tok-1",
      task: "roi_fusion",
      preset: "night_ir",
      steps: [{ id: "deblock", params: { filter: "deblock", filter_type: "strong", block: 8 } }],
      osdBoxes: [],
      osdBoxesConfirmed: false,
      noOsd: false,
      trim: null,
      roi: {
        firstFrame: 10,
        lastFrame: 39,
        referenceFrame: 20,
        box: [100, 200, 60, 14],
        kind: "plate",
        scale: 2,
        method: "median",
      },
    });
  });

  it("goes back to the full preset chain when leaving the multi-frame still", () => {
    const roi = withTask(initialChoices(ANALYSIS, PRESETS_RESPONSE), "roi_fusion", ANALYSIS, PRESETS_RESPONSE);

    expect(Object.keys(withTask(roi, "clarify", ANALYSIS, PRESETS_RESPONSE).steps)).toEqual(["deblock", "gray"]);
  });
});
