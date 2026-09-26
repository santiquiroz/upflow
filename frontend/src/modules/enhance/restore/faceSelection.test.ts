import { describe, expect, it } from "vitest";
import type { RestoreOptions } from "../../../lib/restoreApiTypes";
import {
  bySize,
  effectiveFaceBlend,
  faceTier,
  isSelectable,
  needsConfirmation,
  selectedFaceIndices,
  withFaceChoices,
} from "./faceSelection";
import { makeFace } from "./restoreTestFixtures";

describe("faceTier", () => {
  it.each([
    [40, 0.02, "restore"],
    [32, 0.02, "restore"],
    [40, 0.06, "alreadyClear"],
    [31.9, 0.02, "small"],
    [16, 0.5, "small"],
    [15.9, 0.02, "tooSmallFaithful"],
    [8, 0.02, "tooSmallFaithful"],
    [7.9, 0.02, "tooSmall"],
  ])("puts a face of %s px between the eyes and sharpness %s in %s", (eyePx, sharpness, tier) => {
    expect(faceTier(makeFace({ eyePx, sharpness }))).toBe(tier);
  });

  it("treats an unmeasured face as too small", () => {
    expect(faceTier(makeFace({ eyePx: null }))).toBe("tooSmall");
  });

  it("reads a large face without a sharpness measure as one to restore", () => {
    expect(faceTier(makeFace({ eyePx: 50, sharpness: null }))).toBe("restore");
  });

  it("only blocks faces under 8 px and only asks to confirm between 8 and 16 px", () => {
    expect(isSelectable("tooSmall")).toBe(false);
    expect(isSelectable("tooSmallFaithful")).toBe(true);
    expect(needsConfirmation("tooSmallFaithful")).toBe(true);
    expect(needsConfirmation("small")).toBe(false);
  });
});

describe("bySize", () => {
  it("lists the largest face first without touching the input", () => {
    const faces = [makeFace({ index: 0, eyePx: 20 }), makeFace({ index: 1, eyePx: 60 }), makeFace({ index: 2, eyePx: null })];
    expect(bySize(faces).map((face) => face.index)).toEqual([1, 0, 2]);
    expect(faces.map((face) => face.index)).toEqual([0, 1, 2]);
  });
});

describe("selectedFaceIndices", () => {
  it("keeps the checked faces that can be restored", () => {
    const faces = [
      makeFace({ index: 0, enabled: true }),
      makeFace({ index: 1, enabled: false }),
      makeFace({ index: 2, enabled: true, eyePx: 5 }),
    ];
    expect(selectedFaceIndices(faces)).toEqual([0]);
  });
});

describe("effectiveFaceBlend", () => {
  const proposedOn = makeFace({ enabled: true, blend: 0.6 });
  const proposedOff = makeFace({ enabled: false, blend: 0.5 });

  it("follows the step blend for a face the policy turned on and the user left alone", () => {
    expect(effectiveFaceBlend(proposedOn, proposedOn, 0.8)).toBe(0.8);
  });

  it("keeps the blend the user set on the face", () => {
    expect(effectiveFaceBlend({ ...proposedOn, blend: 0.3 }, proposedOn, 0.8)).toBe(0.3);
  });

  it("keeps the policy blend of an opt-in face", () => {
    expect(effectiveFaceBlend({ ...proposedOff, enabled: true }, proposedOff, 0.8)).toBe(0.5);
  });

  it("keeps the face blend when there is no proposal to compare with", () => {
    expect(effectiveFaceBlend(proposedOn, undefined, 0.8)).toBe(0.6);
  });
});

describe("withFaceChoices", () => {
  const proposed = [
    makeFace({ index: 0, enabled: true, blend: 0.6 }),
    makeFace({ index: 1, enabled: false, blend: 0.5, eyePx: 20 }),
    makeFace({ index: 2, enabled: false, blend: 0.4, eyePx: 4 }),
  ];

  it("leaves the options alone when the faces step is off", () => {
    const options: RestoreOptions = { preset: "gentle", repair: { sensitivity: 0.5 } };
    expect(withFaceChoices(options, proposed, proposed)).toBe(options);
  });

  it("names every face to restore and its blend, so only the faces in the grid are restored", () => {
    const current = [proposed[0], { ...proposed[1], enabled: true }, { ...proposed[2], enabled: true }];
    const options: RestoreOptions = { faces: { model: "gfpgan-v1.4", blend: 0.7 } };

    expect(withFaceChoices(options, current, proposed)).toEqual({
      faces: { model: "gfpgan-v1.4", blend: 0.7, selected: [0, 1], per_face: { "0": 0.7, "1": 0.5 } },
    });
    expect(options).toEqual({ faces: { model: "gfpgan-v1.4", blend: 0.7 } });
  });

  it("sends an empty selection when every face was unchecked", () => {
    const current = proposed.map((face) => ({ ...face, enabled: false }));
    expect(withFaceChoices({ faces: { blend: 0.6 } }, current, proposed)).toEqual({
      faces: { blend: 0.6, selected: [], per_face: {} },
    });
  });
});
