import { describe, expect, it } from "vitest";
import {
  PREVIEW_MAX_SIDE,
  beforeAtFullResolution,
  centeredPreviewCrop,
  cropFraction,
  previewCropFromPoints,
  previewSourceRect,
  readPreviewResult,
  withPreviewCrop,
} from "./previewCrop";
import { makeCompletedRestoreJob, makeRestoreMetadata } from "./restoreTestFixtures";

const BOX = { left: 100, top: 50, width: 600, height: 400 };
const WORKING = { width: 1200, height: 800 };
const PREVIEW_ONLY = { artifacts: ["preview"], previewCrop: [100, 200, 300, 250] };

describe("previewCropFromPoints", () => {
  it("maps a drag on the shown photo to working-copy pixels", () => {
    const crop = previewCropFromPoints({ x: 200, y: 100 }, { x: 300, y: 200 }, BOX, WORKING);

    expect(crop).toEqual([200, 100, 200, 200]);
  });

  it("keeps the drag's anchor and stops at 512 px on each side", () => {
    const crop = previewCropFromPoints({ x: 600, y: 400 }, { x: 110, y: 60 }, BOX, WORKING);

    expect(crop).toEqual([1000 - PREVIEW_MAX_SIDE, 700 - PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE]);
  });

  it("centers the largest allowed area on a click", () => {
    const crop = previewCropFromPoints({ x: 400, y: 250 }, { x: 401, y: 251 }, BOX, WORKING);

    expect(crop).toEqual([344, 144, PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE]);
  });

  it("keeps a click near the edge inside the photo", () => {
    const crop = previewCropFromPoints({ x: 102, y: 52 }, { x: 102, y: 52 }, BOX, WORKING);

    expect(crop).toEqual([0, 0, PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE]);
  });

  it("clamps a drag that leaves the shown photo", () => {
    const crop = previewCropFromPoints({ x: 650, y: 400 }, { x: 900, y: 600 }, BOX, WORKING);

    expect(crop).toEqual([1100, 700, 100, 100]);
  });

  it("rejects a sliver too thin to preview", () => {
    expect(previewCropFromPoints({ x: 200, y: 100 }, { x: 400, y: 104 }, BOX, WORKING)).toBeNull();
  });

  it("never exceeds a photo smaller than the limit", () => {
    const small = { width: 300, height: 200 };

    expect(previewCropFromPoints({ x: 400, y: 250 }, { x: 400, y: 250 }, BOX, small)).toEqual([0, 0, 300, 200]);
  });

  it("gives nothing for a pointer without coordinates", () => {
    expect(previewCropFromPoints({ x: Number.NaN, y: 1 }, { x: 5, y: 5 }, BOX, WORKING)).toBeNull();
  });

  it("gives nothing while the photo has no size on screen", () => {
    expect(previewCropFromPoints({ x: 1, y: 1 }, { x: 5, y: 5 }, { ...BOX, width: 0 }, WORKING)).toBeNull();
  });
});

describe("centeredPreviewCrop", () => {
  it("starts on the center of the photo", () => {
    expect(centeredPreviewCrop(WORKING)).toEqual([344, 144, PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE]);
  });

  it("takes the whole photo when it is smaller than the limit", () => {
    expect(centeredPreviewCrop({ width: 400, height: 300 })).toEqual([0, 0, 400, 300]);
  });
});

describe("cropFraction", () => {
  it("places the area over the shown photo", () => {
    expect(cropFraction([300, 200, 600, 400], WORKING)).toEqual({ x: 0.25, y: 0.25, width: 0.5, height: 0.5 });
  });
});

describe("previewSourceRect", () => {
  it("scales the area to the session preview, which can be smaller than the working copy", () => {
    const rect = previewSourceRect([400, 200, 300, 250], WORKING, { width: 600, height: 400 });

    expect(rect).toEqual({ x: 200, y: 100, width: 150, height: 125 });
  });
});

describe("beforeAtFullResolution", () => {
  it("is only true while the session preview keeps every pixel", () => {
    expect(beforeAtFullResolution({ width: 2048, height: 1536 })).toBe(true);
    expect(beforeAtFullResolution({ width: 1536, height: 3000 })).toBe(false);
  });
});

describe("withPreviewCrop", () => {
  it("adds the area without touching the other options", () => {
    const options = { preset: "gentle", repair: { sensitivity: 0.5 } };

    expect(withPreviewCrop(options, [1, 2, 30, 40])).toEqual({ ...options, preview_crop: [1, 2, 30, 40] });
    expect(options).not.toHaveProperty("preview_crop");
  });
});

describe("readPreviewResult", () => {
  it("reads the area of a finished preview run", () => {
    const job = makeCompletedRestoreJob(makeRestoreMetadata(PREVIEW_ONLY));

    expect(readPreviewResult(job)).toEqual({ crop: [100, 200, 300, 250] });
  });

  it("ignores a full restoration", () => {
    const job = makeCompletedRestoreJob(makeRestoreMetadata({ previewCrop: null }));

    expect(readPreviewResult(job)).toBeNull();
  });

  it("ignores a run that has not finished", () => {
    const job = { ...makeCompletedRestoreJob(makeRestoreMetadata(PREVIEW_ONLY)), status: "running" as const };

    expect(readPreviewResult(job)).toBeNull();
  });

  it("ignores an area that is not four numbers", () => {
    const job = makeCompletedRestoreJob(makeRestoreMetadata({ ...PREVIEW_ONLY, previewCrop: [1, 2, "3"] }));

    expect(readPreviewResult(job)).toBeNull();
  });
});
