import { describe, expect, it } from "vitest";
import {
  addVideoStep,
  deriveVideoSteps,
  getAddableVideoStepIds,
  removeVideoStep,
  type VideoStepConfig,
} from "./videoSteps";

const BASE_CONFIG: VideoStepConfig = {
  modelName: "RealESRGAN x4plus",
  scale: 4,
  fpsMultiplier: 1,
  targetFps: null,
  interpEngine: "rife",
  keepAudio: true,
  audioEnhance: null,
  audioRestore: null,
  keepSubtitles: false,
};

describe("deriveVideoSteps", () => {
  it("derives the upscale step from the selected model and scale", () => {
    expect(deriveVideoSteps(BASE_CONFIG)).toEqual([
      {
        id: "upscale",
        modelName: "RealESRGAN x4plus",
        scale: 4,
      },
    ]);
  });

  it("keeps the backend's fixed semantic order when every step is active", () => {
    const steps = deriveVideoSteps({
      ...BASE_CONFIG,
      fpsMultiplier: 3,
      audioEnhance: "deepfilter",
      audioRestore: "apollo",
      keepSubtitles: true,
    });

    expect(steps.map((step) => step.id)).toEqual([
      "upscale",
      "interpolate",
      "audio",
      "subtitles",
    ]);
    expect(steps[1]).toEqual({
      id: "interpolate",
      interpEngine: "rife",
      fpsMultiplier: 3,
      targetFps: null,
    });
    expect(steps[2]).toEqual({
      id: "audio",
      audioEnhance: "deepfilter",
      audioRestore: "apollo",
    });
  });

  it("includes interpolation when the existing target-FPS mode is active", () => {
    expect(
      deriveVideoSteps({
        ...BASE_CONFIG,
        targetFps: "60000/1001",
        interpEngine: "gmfss",
      })[1],
    ).toEqual({
      id: "interpolate",
      interpEngine: "gmfss",
      fpsMultiplier: 1,
      targetFps: "60000/1001",
    });
  });

  it("does not claim audio enhancement when original audio is disabled", () => {
    const steps = deriveVideoSteps({
      ...BASE_CONFIG,
      keepAudio: false,
      audioEnhance: "rnnoise",
      audioRestore: "audiosr",
    });

    expect(steps.map((step) => step.id)).toEqual(["upscale"]);
  });
});

describe("removeVideoStep", () => {
  it("turns off the fields that generate upscale and interpolation", () => {
    const activeConfig = {
      ...BASE_CONFIG,
      fpsMultiplier: 4,
      targetFps: "60/1",
    };

    expect(removeVideoStep(activeConfig, "upscale").scale).toBeNull();
    expect(removeVideoStep(activeConfig, "interpolate")).toEqual({
      ...activeConfig,
      fpsMultiplier: 1,
      targetFps: null,
    });
  });

  it("turns off both audio processors and subtitle preservation", () => {
    const activeConfig = {
      ...BASE_CONFIG,
      audioEnhance: "deepfilter",
      audioRestore: "apollo",
      keepSubtitles: true,
    };

    expect(removeVideoStep(activeConfig, "audio")).toEqual({
      ...activeConfig,
      audioEnhance: null,
      audioRestore: null,
    });
    expect(removeVideoStep(activeConfig, "subtitles").keepSubtitles).toBe(false);
  });
});

describe("addVideoStep", () => {
  it("uses sensible interpolation, audio and subtitle defaults", () => {
    const withoutUpscale = { ...BASE_CONFIG, scale: null };
    const withInterpolation = addVideoStep(withoutUpscale, "interpolate");
    const withAudio = addVideoStep(withInterpolation, "audio");
    const withSubtitles = addVideoStep(withAudio, "subtitles");

    expect(withInterpolation).toEqual({
      ...withoutUpscale,
      fpsMultiplier: 2,
      targetFps: null,
    });
    expect(withAudio.audioEnhance).toBe("deepfilter");
    expect(withAudio.keepAudio).toBe(true);
    expect(withSubtitles.keepSubtitles).toBe(true);
  });

  it("restores upscale with the caller's profile scale", () => {
    expect(
      addVideoStep({ ...BASE_CONFIG, scale: null }, "upscale", {
        upscaleScale: 3,
      }).scale,
    ).toBe(3);
  });

  it("leaves already-active step fields unchanged", () => {
    const activeConfig = {
      ...BASE_CONFIG,
      fpsMultiplier: 4,
      audioEnhance: "rnnoise",
      keepSubtitles: true,
    };

    expect(addVideoStep(activeConfig, "interpolate")).toBe(activeConfig);
    expect(addVideoStep(activeConfig, "audio")).toBe(activeConfig);
    expect(addVideoStep(activeConfig, "subtitles")).toBe(activeConfig);
  });
});

describe("getAddableVideoStepIds", () => {
  it("returns only steps that are not represented by the current configuration", () => {
    expect(
      getAddableVideoStepIds({
        ...BASE_CONFIG,
        fpsMultiplier: 2,
        keepSubtitles: true,
      }),
    ).toEqual(["audio"]);
  });
});
