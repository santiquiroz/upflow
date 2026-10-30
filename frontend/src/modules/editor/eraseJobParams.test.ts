import { describe, expect, it, vi } from "vitest";
import type { GenerationModelSummary } from "../../lib/apiTypes";
import { apiPostJson } from "../../lib/api";
import { createGenerationJob, type CreateGenerationJobParams } from "../../services/generation";
import {
  buildEditorJobParams,
  canReplace,
  defaultEraseModel,
  effectiveMode,
  isReplaceModel,
  modelForMode,
  shouldUseAutoErase,
} from "./eraseJobParams";

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
const autoEraser: GenerationModelSummary = {
  id: "eraser-auto",
  name: "Borrado automático (LaMa + OSOR)",
  status: "installed",
  eraseOnly: true,
  supportsInpaint: true,
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
  it("defaults to eraser-auto when installed and otherwise keeps the current empty selection", () => {
    expect(defaultEraseModel([autoEraser, inpaint])).toBe("eraser-auto");
    expect(defaultEraseModel([inpaint])).toBeNull();
  });

  it("allows replacement only with an installed non-turbo fill model", () => {
    expect(canReplace([autoEraser, inpaint])).toBe(true);
    expect(canReplace([autoEraser])).toBe(false);
    expect(canReplace([{ ...inpaint, status: "converting" }])).toBe(false);
    expect(canReplace([{ ...inpaint, speed: "turbo" }])).toBe(false);
    expect(canReplace([{ ...inpaint, supportsInpaint: false }])).toBe(false);
  });

  it.each([
    [inpaint, true],
    [lama, false],
    [{ ...inpaint, speed: "turbo" }, false],
    [{ ...inpaint, status: "not_installed" }, false],
  ] as const)("identifies replacement model eligibility for %j", (model, expected) => {
    expect(isReplaceModel(model)).toBe(expected);
  });

  it("falls back to erase only after capabilities load without a fill model", () => {
    expect(effectiveMode("replace", [autoEraser], false)).toBe("replace");
    expect(effectiveMode("replace", [autoEraser], true)).toBe("erase");
    expect(effectiveMode("replace", [inpaint], true)).toBe("replace");
    expect(effectiveMode("erase", [autoEraser], true)).toBe("erase");
  });

  it("selects a model compatible with the next mode", () => {
    expect(modelForMode("replace", autoEraser.id, [autoEraser, inpaint])).toBeNull();
    expect(modelForMode("replace", inpaint.id, [autoEraser, inpaint])).toBe(inpaint.id);
    expect(modelForMode("erase", null, [autoEraser, inpaint])).toBe(autoEraser.id);
    expect(modelForMode("erase", "missing", [autoEraser, inpaint])).toBe(autoEraser.id);
    expect(modelForMode("erase", inpaint.id, [autoEraser, inpaint])).toBe(inpaint.id);
  });

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

  it("keeps protection and omits the legacy strategy for eraser-auto", async () => {
    vi.clearAllMocks();
    const input = {
      ...params,
      modelId: "eraser-auto",
      protectMaskToken: "protect-mask",
      eraseStrategy: "auto" as const,
    };
    const built = buildEditorJobParams(input, "erase", [autoEraser]);

    expect(built).not.toHaveProperty("eraseStrategy");
    expect(built.protectMaskToken).toBe("protect-mask");
    await createGenerationJob(built);
    expect(apiPostJson).toHaveBeenCalledWith(
      "/generation/jobs",
      expect.objectContaining({
        modelId: "eraser-auto",
        maskImageToken: "edit-mask",
        protectMaskToken: "protect-mask",
      }),
    );
    expect(apiPostJson).not.toHaveBeenCalledWith(
      "/generation/jobs",
      expect.objectContaining({ eraseStrategy: "auto" }),
    );
  });
});
