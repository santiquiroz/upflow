import { describe, expect, it } from "vitest";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { batchJobParams, batchOptions, batchSkipsFaces, batchSteps, canBatch } from "./restoreBatch";

const BASE: CreateRestoreJobParams = {
  source: { token: "tok-1" },
  steps: ["repair", "denoise", "faces"],
  options: {
    preset: "portrait",
    geometry: { rotate90: 1, crop: [0, 0, 100, 100], angle: 2 },
    repair: { sensitivity: 0.6, use_user_mask: true },
    denoise: { strength: 0.3, keep_grain: 0.25 },
    tone: { strength: 0.7, gray_point: [10, 20] },
    faces: { blend: 0.6, selected: [0], per_face: { 0: 0.6 } },
    preview_crop: [1, 2, 30, 40],
  },
  scale: 2,
  modelId: "realesr",
  device: "gpu:0",
  outputFormat: "png",
};

function photo(name: string): File {
  return new File(["x"], name, { type: "image/jpeg" });
}

describe("batchSteps", () => {
  it("drops the faces step and keeps the chain order", () => {
    expect(batchSteps(["repair", "faces", "colorize"])).toEqual(["repair", "colorize"]);
  });
});

describe("batchOptions", () => {
  it("keeps the step settings and drops what belongs to the first photo", () => {
    expect(batchOptions(BASE.options)).toEqual({
      preset: "portrait",
      repair: { sensitivity: 0.6 },
      denoise: { strength: 0.3, keep_grain: 0.25 },
      tone: { strength: 0.7 },
    });
  });

  it("leaves the original options untouched", () => {
    batchOptions(BASE.options);

    expect(BASE.options.repair).toEqual({ sensitivity: 0.6, use_user_mask: true });
    expect(BASE.options.faces).toBeDefined();
  });

  it("does not invent step settings that were not there", () => {
    expect(batchOptions({ denoise: { strength: 0.3 } })).toEqual({ denoise: { strength: 0.3 } });
  });
});

describe("batchJobParams", () => {
  it("sends each photo as its own job with the same steps and settings", () => {
    const params = batchJobParams(BASE, [photo("a.jpg"), photo("b.jpg")]);

    expect(params.map((item) => ("file" in item.source ? item.source.file.name : null))).toEqual(["a.jpg", "b.jpg"]);
    expect(params[0]).toEqual({
      ...BASE,
      source: { file: expect.any(File) },
      steps: ["repair", "denoise"],
      options: batchOptions(BASE.options),
    });
  });
});

describe("canBatch", () => {
  it("needs a step left once faces are dropped", () => {
    expect(canBatch(BASE)).toBe(true);
    expect(canBatch({ ...BASE, steps: ["faces"] })).toBe(false);
  });
});

describe("batchSkipsFaces", () => {
  it("says so when the first photo restored faces", () => {
    expect(batchSkipsFaces(BASE)).toBe(true);
    expect(batchSkipsFaces({ ...BASE, steps: ["repair"] })).toBe(false);
  });
});
