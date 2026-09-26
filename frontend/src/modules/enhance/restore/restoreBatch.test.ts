import { describe, expect, it } from "vitest";
import type { CreateRestoreJobParams } from "../../../services/restore";
import {
  batchFaceOptions,
  batchJobParams,
  batchOffersFaces,
  batchOptions,
  batchSteps,
  canBatch,
  hasBatchSteps,
} from "./restoreBatch";

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
  it("drops the faces step unless faces were turned on for the batch", () => {
    expect(batchSteps(["repair", "faces", "colorize"], false)).toEqual(["repair", "colorize"]);
    expect(batchSteps(["repair", "faces", "colorize"], true)).toEqual(["repair", "faces", "colorize"]);
  });
});

describe("batchFaceOptions", () => {
  it("keeps the model and the blend and drops the faces chosen on the first photo", () => {
    expect(batchFaceOptions({ model: "gfpgan-v1.4", blend: 0.6, selected: [0], per_face: { 0: 0.6 } })).toEqual({
      model: "gfpgan-v1.4",
      blend: 0.6,
    });
    expect(batchFaceOptions(undefined)).toEqual({});
  });
});

describe("batchOptions", () => {
  it("keeps the step settings, drops what belongs to the first photo and marks the batch", () => {
    expect(batchOptions(BASE.options, false)).toEqual({
      preset: "portrait",
      repair: { sensitivity: 0.6 },
      denoise: { strength: 0.3, keep_grain: 0.25 },
      tone: { strength: 0.7 },
      batch: true,
    });
  });

  it("carries only the shared face settings when faces are on", () => {
    expect(batchOptions(BASE.options, true).faces).toEqual({ blend: 0.6 });
  });

  it("leaves the original options untouched", () => {
    batchOptions(BASE.options, true);

    expect(BASE.options.repair).toEqual({ sensitivity: 0.6, use_user_mask: true });
    expect(BASE.options.faces).toEqual({ blend: 0.6, selected: [0], per_face: { 0: 0.6 } });
  });

  it("does not invent step settings that were not there", () => {
    expect(batchOptions({ denoise: { strength: 0.3 } }, false)).toEqual({ denoise: { strength: 0.3 }, batch: true });
  });
});

describe("batchJobParams", () => {
  it("sends each photo as its own job with the same steps and settings", () => {
    const params = batchJobParams(BASE, [photo("a.jpg"), photo("b.jpg")], false);

    expect(params.map((item) => ("file" in item.source ? item.source.file.name : null))).toEqual(["a.jpg", "b.jpg"]);
    expect(params[0]).toEqual({
      ...BASE,
      source: { file: expect.any(File) },
      steps: ["repair", "denoise"],
      options: batchOptions(BASE.options, false),
    });
  });

  it("keeps the faces step when faces are on", () => {
    const [first] = batchJobParams(BASE, [photo("a.jpg")], true);

    expect(first.steps).toEqual(["repair", "denoise", "faces"]);
    expect(first.options.faces).toEqual({ blend: 0.6 });
  });
});

describe("canBatch and hasBatchSteps", () => {
  it("offers the batch whenever the photo ran a step", () => {
    expect(canBatch(BASE)).toBe(true);
    expect(canBatch({ ...BASE, steps: [] })).toBe(false);
  });

  it("has nothing to run when only faces were restored and faces stay off", () => {
    expect(hasBatchSteps({ ...BASE, steps: ["faces"] }, false)).toBe(false);
    expect(hasBatchSteps({ ...BASE, steps: ["faces"] }, true)).toBe(true);
  });
});

describe("batchOffersFaces", () => {
  it("offers faces only when the first photo restored faces", () => {
    expect(batchOffersFaces(BASE)).toBe(true);
    expect(batchOffersFaces({ ...BASE, steps: ["repair"] })).toBe(false);
  });
});
