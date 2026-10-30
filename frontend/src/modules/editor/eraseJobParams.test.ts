import { describe, expect, it, vi } from "vitest";
import type { GenerationModelSummary } from "../../lib/apiTypes";
import { apiPostJson } from "../../lib/api";
import { createGenerationJob, type CreateGenerationJobParams } from "../../services/generation";
import { buildEditorJobParams, shouldUseAutoErase } from "./eraseJobParams";

vi.mock("../../lib/api", () => ({ apiPostJson: vi.fn() }));

const inpaint: GenerationModelSummary = {
  id: "sdxl-inpaint",
  name: "SDXL Inpaint",
  status: "installed",
  inpaintOnly: true,
};
const lama: GenerationModelSummary = {
  id: "eraser-lama",
  name: "LaMa",
  status: "installed",
  eraseOnly: true,
};
const params: CreateGenerationJobParams = {
  prompt: "",
  negativePrompt: null,
  modelId: inpaint.id,
  steps: 25,
  guidance: 7.5,
  width: 1024,
  height: 1024,
  seed: null,
  device: null,
  autoUpscale: false,
  upscaleModelName: null,
  upscaleScale: null,
  upscaleModelId: null,
  initImageToken: "base",
  maskImageToken: "edit-mask",
};

describe("automatic erase eligibility", () => {
  it("uses the installed inpaint model and LaMa only in erase mode", () => {
    expect(shouldUseAutoErase("erase", inpaint.id, [inpaint, lama])).toBe(true);
    expect(shouldUseAutoErase("replace", inpaint.id, [inpaint, lama])).toBe(false);
  });

  it.each([
    [inpaint],
    [inpaint, { ...lama, status: "converting" }],
    [{ ...inpaint, status: "converting" }, lama],
    [{ ...inpaint, inpaintOnly: false }, lama],
    [{ ...inpaint, eraseOnly: true }, lama],
    [lama],
  ])("rejects unavailable models or models without dedicated diffusion inpainting: %j", (...models) => {
    expect(shouldUseAutoErase("erase", inpaint.id, models)).toBe(false);
  });
});

describe("buildEditorJobParams", () => {
  it("adds auto erase and preserves protection without mutating the input", () => {
    const input = { ...params, protectMaskToken: "protect-mask" };
    const result = buildEditorJobParams(input, "erase", [inpaint, lama]);

    expect(result).toEqual({ ...input, eraseStrategy: "auto" });
    expect(input).not.toHaveProperty("eraseStrategy");
  });

  it("requires an edit mask", () => {
    const input = { ...params, maskImageToken: undefined };

    expect(buildEditorJobParams(input, "erase", [inpaint, lama])).not.toHaveProperty("eraseStrategy");
  });

  it("removes an obsolete strategy when changing modes or capabilities", () => {
    const input: CreateGenerationJobParams = { ...params, eraseStrategy: "auto", protectMaskToken: "keep" };

    expect(buildEditorJobParams(input, "replace", [inpaint, lama])).toEqual({ ...params, protectMaskToken: "keep" });
    expect(buildEditorJobParams(input, "erase", [inpaint])).toEqual({ ...params, protectMaskToken: "keep" });
  });

  it("transmits automatic erase and protection through the generation service", async () => {
    const input = { ...params, protectMaskToken: "protect-mask" };

    await createGenerationJob(buildEditorJobParams(input, "erase", [inpaint, lama]));

    expect(apiPostJson).toHaveBeenCalledWith("/generation/jobs", expect.objectContaining({
      eraseStrategy: "auto",
      protectMaskToken: "protect-mask",
      maskImageToken: "edit-mask",
    }));
  });
});
