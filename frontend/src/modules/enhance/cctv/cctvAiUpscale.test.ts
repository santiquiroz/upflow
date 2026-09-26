import { describe, expect, it } from "vitest";
import { aiUpscaleFields, aiUpscaleFor, generativeTagKey, scaleFor, withAiUpscaleStep } from "./cctvAiUpscale";
import { AI_STEPS, AI_UPSCALE_MODELS, CLASSIC_STEPS } from "./cctvFixtures";

const [GENERATIVE, NON_GENERATIVE] = AI_UPSCALE_MODELS;

describe("generativeTagKey", () => {
  it("labels each model by whether it invents texture", () => {
    expect(generativeTagKey(true)).toBe("model.tag.generative");
    expect(generativeTagKey(false)).toBe("model.tag.nonGenerative");
  });
});

describe("scaleFor", () => {
  it("keeps the preferred scale when the model has its export", () => {
    expect(scaleFor(GENERATIVE, 4)).toBe(4);
  });

  it("falls back to the model's first installed scale", () => {
    expect(scaleFor(NON_GENERATIVE, 4)).toBe(2);
    expect(scaleFor(GENERATIVE, null)).toBe(2);
  });
});

describe("aiUpscaleFor", () => {
  it("picks the model with a scale it can run", () => {
    expect(aiUpscaleFor(AI_UPSCALE_MODELS, NON_GENERATIVE.id, 3)).toEqual({ modelId: NON_GENERATIVE.id, scale: 2 });
  });

  it("is None for no model or one that is not offered", () => {
    expect(aiUpscaleFor(AI_UPSCALE_MODELS, null, 2)).toBeNull();
    expect(aiUpscaleFor(AI_UPSCALE_MODELS, "realesrgan-ncnn", 2)).toBeNull();
  });
});

describe("withAiUpscaleStep", () => {
  it("adds the AI upscale step when a model is chosen", () => {
    const steps = withAiUpscaleStep({}, AI_STEPS, { modelId: GENERATIVE.id, scale: 2 });

    expect(steps).toEqual({ ai_upscale: { filter: "onnx_upscale", params: {} } });
  });

  it("leaves the steps alone with None or outside the AI lane", () => {
    const steps = { gray: { filter: "gray", params: {} } };

    expect(withAiUpscaleStep(steps, AI_STEPS, null)).toBe(steps);
    expect(withAiUpscaleStep(steps, CLASSIC_STEPS, { modelId: GENERATIVE.id, scale: 2 })).toBe(steps);
  });
});

describe("aiUpscaleFields", () => {
  it("sends the model and the scale only with a model", () => {
    expect(aiUpscaleFields({ modelId: "realesrgan-x4plus", scale: 4 })).toEqual({ modelId: "realesrgan-x4plus", scale: 4 });
    expect(aiUpscaleFields(null)).toEqual({});
  });
});
