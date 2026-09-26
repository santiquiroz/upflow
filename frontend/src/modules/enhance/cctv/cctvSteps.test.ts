import { describe, expect, it } from "vitest";
import { AI_STEPS, ANALYSIS, CLASSIC_STEPS, DAY_PRESET, NIGHT_PRESET } from "./cctvFixtures";
import {
  choicesFromPreset,
  defaultChoice,
  hasChosenAiSteps,
  incompleteStepIds,
  missingParams,
  paramValue,
  presetContextOf,
  previewStepRequests,
  stepRequests,
  stepWarningKey,
  visibleSteps,
  withStepEnabled,
  withStepFilter,
  withStepParam,
  type StepChoices,
} from "./cctvSteps";

const PROGRESSIVE = { interlaced: false, sampleAspect: null };

function stepById(id: string) {
  const found = AI_STEPS.find((candidate) => candidate.id === id);
  if (!found) throw new Error(id);
  return found;
}

describe("visibleSteps", () => {
  it("never shows an AI step in the classic lane, even if the catalog brought one", () => {
    const ids = visibleSteps(AI_STEPS, "classic").map((step) => step.id);

    expect(ids).not.toContain("ai_deblock");
    expect(ids).not.toContain("ai_upscale");
    expect(ids).not.toContain("interpolate");
  });

  it("hides the steps other controls own: trim, on-screen text, the AI upscale model and the AI label", () => {
    const ids = visibleSteps(AI_STEPS, "ai").map((step) => step.id);

    expect(ids).toEqual(["aspect", "deinterlace", "deblock", "ai_deblock", "denoise", "crop", "gray", "scale", "sharpen"]);
  });

  it("keeps the catalog order, which is the fixed processing order", () => {
    const ids = visibleSteps(CLASSIC_STEPS, "classic").map((step) => step.id);

    expect(ids).toEqual(["aspect", "deinterlace", "deblock", "denoise", "crop", "gray", "scale", "sharpen"]);
  });
});

describe("presetContextOf", () => {
  it("reads interlacing and the lite sample aspect from the analysis", () => {
    expect(presetContextOf(ANALYSIS)).toEqual({ interlaced: true, sampleAspect: [2, 1] });
  });

  it("assumes progressive square pixels when the analysis can't tell", () => {
    const bare = { ...ANALYSIS, quality: null, video: { ...ANALYSIS.video, lite: null } };

    expect(presetContextOf(bare)).toEqual(PROGRESSIVE);
  });
});

describe("choicesFromPreset", () => {
  it("applies conditional steps only when the diagnosis found the defect", () => {
    const choices = choicesFromPreset(DAY_PRESET, "classic", PROGRESSIVE, CLASSIC_STEPS);

    expect(Object.keys(choices)).toEqual(["deblock", "denoise"]);
  });

  it("sets the aspect correction from the stored sample aspect", () => {
    const choices = choicesFromPreset(DAY_PRESET, "classic", { interlaced: true, sampleAspect: [2, 1] }, CLASSIC_STEPS);

    expect(choices.aspect).toEqual({ filter: "setsar", params: { num: 2, den: 1 } });
    expect(choices.deinterlace).toEqual({ filter: "bwdif", params: { mode: "send_frame" } });
  });

  it("splits the filter name out of the preset parameters", () => {
    const choices = choicesFromPreset(DAY_PRESET, "classic", PROGRESSIVE, CLASSIC_STEPS);

    expect(choices.deblock).toEqual({ filter: "deblock", params: { filter_type: "weak", block: 8 } });
  });

  it("leaves a step off when this ffmpeg build lacks its filter", () => {
    const choices = choicesFromPreset(NIGHT_PRESET, "classic", PROGRESSIVE, CLASSIC_STEPS);

    expect(choices.denoise).toBeUndefined();
    expect(choices.gray).toEqual({ filter: "gray", params: {} });
  });

  it("uses the AI lane chain of the preset in the AI lane", () => {
    const choices = choicesFromPreset(DAY_PRESET, "ai", PROGRESSIVE, AI_STEPS);

    expect(Object.keys(choices)).toEqual(["ai_deblock"]);
    expect(choices.ai_deblock.params).toEqual({ strength: 40 });
  });
});

describe("editing steps", () => {
  it("turns a step on with its first filter this build has", () => {
    expect(defaultChoice(stepById("deblock"))).toEqual({ filter: "deblock", params: {} });
    expect(defaultChoice(stepById("denoise"))).toEqual({ filter: "atadenoise", params: {} });
  });

  it("adds and removes steps without touching the previous selection", () => {
    const before: StepChoices = {};
    const on = withStepEnabled(before, stepById("gray"), true);
    const off = withStepEnabled(on, stepById("gray"), false);

    expect(before).toEqual({});
    expect(on).toEqual({ gray: { filter: "gray", params: {} } });
    expect(off).toEqual({});
  });

  it("resets the parameters when the filter changes", () => {
    const choices: StepChoices = { deinterlace: { filter: "bwdif", params: { mode: "send_frame" } } };

    expect(withStepFilter(choices, "deinterlace", "yadif").deinterlace).toEqual({ filter: "yadif", params: {} });
  });

  it("sets a parameter and clears it back to the default with null", () => {
    const set = withStepParam({ deblock: { filter: "deblock", params: {} } }, "deblock", "block", 16);
    const cleared = withStepParam(set, "deblock", "block", null);

    expect(set.deblock.params).toEqual({ block: 16 });
    expect(cleared.deblock.params).toEqual({});
  });

  it("shows the default until the user sets a value", () => {
    const param = stepById("deblock").filters[0].params[1];

    expect(paramValue(param, { filter: "deblock", params: {} })).toBe(8);
    expect(paramValue(param, { filter: "deblock", params: { block: 16 } })).toBe(16);
  });
});

describe("completeness", () => {
  it("names the required parameters without a default that are still empty", () => {
    const crop = stepById("crop");

    expect(missingParams(crop, { filter: "crop", params: { w: 640 } })).toEqual(["h"]);
  });

  it("lists the enabled steps that can't run yet", () => {
    const choices: StepChoices = {
      crop: { filter: "crop", params: {} },
      gray: { filter: "gray", params: {} },
    };

    expect(incompleteStepIds(choices, AI_STEPS)).toEqual(["crop"]);
  });
});

describe("stepRequests", () => {
  it("sends the steps in the fixed order with the filter inside the parameters", () => {
    const choices: StepChoices = {
      gray: { filter: "gray", params: {} },
      deblock: { filter: "deblock", params: { block: 8 } },
    };

    expect(stepRequests(choices, CLASSIC_STEPS)).toEqual([
      { id: "deblock", params: { filter: "deblock", block: 8 } },
      { id: "gray", params: { filter: "gray" } },
    ]);
  });

  it("drops choices for steps the lane doesn't have", () => {
    const choices: StepChoices = { ai_deblock: { filter: "drunet_deblock", params: {} } };

    expect(stepRequests(choices, CLASSIC_STEPS)).toEqual([]);
  });
});

describe("filter preview steps", () => {
  const AI_CHOICES: StepChoices = {
    ai_deblock: { filter: "drunet_deblock", params: { strength: 40 } },
    gray: { filter: "gray", params: {} },
  };

  it("previews only the classic steps of the AI lane", () => {
    expect(previewStepRequests(AI_CHOICES, AI_STEPS)).toEqual([{ id: "gray", params: { filter: "gray" } }]);
  });

  it("says when AI steps are left out of the preview", () => {
    expect(hasChosenAiSteps(AI_CHOICES, AI_STEPS)).toBe(true);
    expect(hasChosenAiSteps({ gray: AI_CHOICES.gray }, AI_STEPS)).toBe(false);
  });
});

describe("stepWarningKey", () => {
  it("warns about halos when sharpening", () => {
    expect(stepWarningKey("sharpen", { filter: "cas", params: {} })).toBe("cctv.sharpen.halos");
  });

  it("warns that interpolated enlargement computes new pixel values", () => {
    expect(stepWarningKey("scale", { filter: "scale", params: { flags: "lanczos" } })).toBe("cctv.scale.newPixels");
    expect(stepWarningKey("scale", { filter: "scale", params: {} })).toBeNull();
  });

  it("warns that stabilization moves every frame and is missing from the single-frame preview", () => {
    expect(stepWarningKey("stabilize", { filter: "vidstab", params: {} })).toBe("cctv.stabilize.moved");
  });
});
