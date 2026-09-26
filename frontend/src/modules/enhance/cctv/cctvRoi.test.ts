import { describe, expect, it } from "vitest";
import { AI_STEPS, ROI_SUMMARY } from "./cctvFixtures";
import {
  EMPTY_ROI,
  MAX_ROI_FRAMES,
  roiBlockerKey,
  roiCatalog,
  roiDensityNotice,
  roiFrameCount,
  roiRequest,
  roiResultNotices,
  withRoiBox,
  type RoiChoice,
} from "./cctvRoi";

const READY: RoiChoice = { ...EMPTY_ROI, first: 10, last: 39, reference: 20, box: [100, 200, 60, 14] };

describe("roiCatalog", () => {
  it("keeps only the steps the multi-frame still runs before aligning", () => {
    expect(roiCatalog(AI_STEPS).map((step) => step.id)).toEqual(["deinterlace", "deblock"]);
  });
});

describe("withRoiBox", () => {
  it("makes the frame the box was drawn on the reference frame", () => {
    expect(withRoiBox(EMPTY_ROI, [0, 0, 20, 10], 42)).toMatchObject({ box: [0, 0, 20, 10], reference: 42 });
  });

  it("forgets the reference when the box is removed", () => {
    expect(withRoiBox(READY, null, 42)).toMatchObject({ box: null, reference: null });
  });
});

describe("roiFrameCount", () => {
  it("counts both ends", () => {
    expect(roiFrameCount(READY)).toBe(30);
  });

  it("has no count until both ends are set in order", () => {
    expect(roiFrameCount({ ...READY, last: null })).toBeNull();
    expect(roiFrameCount({ ...READY, first: 40 })).toBeNull();
  });
});

describe("roiBlockerKey", () => {
  it("lets a complete region through", () => {
    expect(roiBlockerKey(READY, 750)).toBeNull();
  });

  it("asks for the box first", () => {
    expect(roiBlockerKey(EMPTY_ROI, 750)).toBe("cctv.roi.blocked.box");
  });

  it("refuses a range over the frame limit or past the end", () => {
    expect(roiBlockerKey({ ...READY, first: 0, last: MAX_ROI_FRAMES }, 750)).toBe("cctv.roi.blocked.range");
    expect(roiBlockerKey({ ...READY, last: 750 }, 750)).toBe("cctv.roi.blocked.range");
    expect(roiBlockerKey({ ...READY, first: null }, 750)).toBe("cctv.roi.blocked.range");
  });

  it("accepts exactly the frame limit", () => {
    expect(roiBlockerKey({ ...READY, first: 0, last: MAX_ROI_FRAMES - 1 }, 750)).toBeNull();
  });

  it("needs the reference frame inside the range", () => {
    expect(roiBlockerKey({ ...READY, reference: 40 }, 750)).toBe("cctv.roi.blocked.reference");
  });
});

describe("roiRequest", () => {
  it("maps the choice to the job's roi field", () => {
    expect(roiRequest({ ...READY, kind: "face_or_object", scale: 3, method: "trimmed_mean" })).toEqual({
      firstFrame: 10,
      lastFrame: 39,
      referenceFrame: 20,
      box: [100, 200, 60, 14],
      kind: "face_or_object",
      scale: 3,
      method: "trimmed_mean",
    });
  });

  it("has nothing to send without a box", () => {
    expect(roiRequest(EMPTY_ROI)).toBeNull();
  });
});

describe("roiDensityNotice", () => {
  it("always states a plate's height in recorded pixels", () => {
    expect(roiDensityNotice("plate", [0, 0, 60, 14])).toEqual({ key: "cctv.roi.densityPlate", params: { px: 14 } });
  });

  it("warns about a face narrower than about 40 px", () => {
    expect(roiDensityNotice("face_or_object", [0, 0, 32, 40])).toEqual({ key: "cctv.roi.densityFace", params: { px: 32 } });
  });

  it("says nothing about a face that is wide enough or a missing box", () => {
    expect(roiDensityNotice("face_or_object", [0, 0, 40, 40])).toBeNull();
    expect(roiDensityNotice("plate", null)).toBeNull();
  });
});

describe("roiResultNotices", () => {
  it("lists the fusion's warnings but not the frames-used line", () => {
    expect(roiResultNotices(ROI_SUMMARY).map((notice) => notice.key)).toEqual([
      "cctv.roi.densityPlate",
      "cctv.roi.nearCopies",
      "cctv.roi.result.rejected",
    ]);
  });

  it("names the frames that could not be aligned", () => {
    const rejected = roiResultNotices(ROI_SUMMARY).at(-1);

    expect(rejected).toEqual({ key: "cctv.roi.result.rejected", params: { frames: "3, 17" } });
  });

  it("adds nothing when every frame aligned", () => {
    expect(roiResultNotices({ ...ROI_SUMMARY, notices: [], rejectedFrames: [] })).toEqual([]);
  });
});
