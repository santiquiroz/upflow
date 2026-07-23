import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiGet, apiPostJson } from "../lib/api";
import { createGenerationJob, getGenerationJob } from "./generation";

vi.mock("../lib/api", () => ({
  apiGet: vi.fn(),
  apiPostJson: vi.fn(),
}));

describe("generation service", () => {
  beforeEach(() => vi.clearAllMocks());

  it("posts camelCase body omitting empty optionals", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ id: "j1" });

    await createGenerationJob({
      prompt: "a red apple",
      negativePrompt: null,
      modelId: "gen--amd--sd15",
      steps: 25,
      guidance: 7.5,
      width: 512,
      height: 512,
      seed: null,
      device: null,
      autoUpscale: false,
      upscaleModelName: null,
      upscaleScale: null,
      upscaleModelId: null,
    });

    expect(apiPostJson).toHaveBeenCalledWith("/generation/jobs", {
      prompt: "a red apple",
      modelId: "gen--amd--sd15",
      steps: 25,
      guidance: 7.5,
      width: 512,
      height: 512,
      autoUpscale: false,
    });
  });

  it("includes upscale params only when autoUpscale", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ id: "j1" });

    await createGenerationJob({
      prompt: "x",
      negativePrompt: "blurry",
      modelId: "m",
      steps: 25,
      guidance: 7.5,
      width: 512,
      height: 512,
      seed: 42,
      device: "dml:0",
      autoUpscale: true,
      upscaleModelName: "realesrgan-x4plus",
      upscaleScale: 4,
      upscaleModelId: null,
    });

    const body = vi.mocked(apiPostJson).mock.calls[0][1] as Record<string, unknown>;
    expect(body.upscaleModelName).toBe("realesrgan-x4plus");
    expect(body.upscaleScale).toBe(4);
    expect(body.seed).toBe(42);
    expect(body.negativePrompt).toBe("blurry");
    expect(body.device).toBe("dml:0");
  });

  it("includes seed=0 in the body (does not regress on falsy checks)", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ id: "j1" });

    await createGenerationJob({
      prompt: "x",
      negativePrompt: null,
      modelId: "m",
      steps: 25,
      guidance: 7.5,
      width: 512,
      height: 512,
      seed: 0,
      device: null,
      autoUpscale: false,
      upscaleModelName: null,
      upscaleScale: null,
      upscaleModelId: null,
    });

    const body = vi.mocked(apiPostJson).mock.calls[0][1] as Record<string, unknown>;
    expect(body.seed).toBe(0);
  });

  it("gets a job by id", async () => {
    vi.mocked(apiGet).mockResolvedValue({ id: "j1" });
    await getGenerationJob("j1");
    expect(apiGet).toHaveBeenCalledWith("/generation/jobs/j1");
  });
});
