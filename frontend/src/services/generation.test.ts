import { beforeEach, describe, expect, it, vi } from "vitest";

import * as api from "../lib/api";
import { apiGet, apiPostJson } from "../lib/api";
import {
  convertGenerationModel,
  createGenerationJob,
  getConversionStatus,
  getGenerationJob,
  installGenerationModel,
  preflightGenerationModel,
  searchGenerationModels,
} from "./generation";

vi.mock("../lib/api", () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
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

  it("starts a generation model conversion", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ conversionId: "c1", statusUrl: "/x" });

    await convertGenerationModel("amd/x");

    expect(apiPostJson).toHaveBeenCalledWith("/generation/models/convert", { repoId: "amd/x" });
  });

  it("gets a generation model conversion by id", async () => {
    vi.mocked(apiGet).mockResolvedValue({ conversionId: "c1" });

    await getConversionStatus("c1");

    expect(apiGet).toHaveBeenCalledWith("/generation/models/convert/c1");
  });

  it("searches with an empty query for the browse view", async () => {
    const spy = vi.spyOn(api, "apiGet").mockResolvedValue({ results: [] });
    await searchGenerationModels("");
    expect(spy).toHaveBeenCalledWith("/generation/models/search?q=");
  });

  it("requests preflight with the reference resolution", async () => {
    const spy = vi.spyOn(api, "apiGet").mockResolvedValue({});
    await preflightGenerationModel("owner/name", 512, 512);
    expect(spy).toHaveBeenCalledWith(
      "/generation/models/preflight?repoId=owner%2Fname&width=512&height=512",
    );
  });

  it("sends the chosen precision when installing", async () => {
    const spy = vi.spyOn(api, "apiPostJson").mockResolvedValue({ installId: "1", statusUrl: "/x" });
    await installGenerationModel("owner/name", "fp16");
    expect(spy).toHaveBeenCalledWith("/generation/models", {
      repoId: "owner/name",
      precision: "fp16",
    });
  });

  it("sends the chosen checkpoint path when installing", async () => {
    const spy = vi.spyOn(api, "apiPostJson").mockResolvedValue({ installId: "1", statusUrl: "/x" });
    await installGenerationModel(
      "owner/name",
      "fp16",
      "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    );
    expect(spy).toHaveBeenCalledWith("/generation/models", {
      repoId: "owner/name",
      precision: "fp16",
      checkpointPath: "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    });
  });

  it("omits precision when none is chosen", async () => {
    const spy = vi.spyOn(api, "apiPostJson").mockResolvedValue({ installId: "1", statusUrl: "/x" });
    await installGenerationModel("owner/name");
    expect(spy).toHaveBeenCalledWith("/generation/models", { repoId: "owner/name" });
  });
});
