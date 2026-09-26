import { describe, expect, it } from "vitest";
import { draftOf, isDraftChanged, readRestoredFaces, withFaceChoice } from "./faceResultModel";

describe("readRestoredFaces", () => {
  it("keeps only the faces the job restored, in order", () => {
    const raw = [
      { index: 1, enabled: true, restored: true, blend: 0.6, eyePx: 40 },
      { index: 0, enabled: false, restored: false, blend: 0.4 },
      { index: 2, enabled: false, restored: true, blend: 0.5, recomposedAt: "2026-09-25T10:00:00Z" },
    ];
    expect(readRestoredFaces(raw)).toEqual([
      { index: 1, enabled: true, blend: 0.6 },
      { index: 2, enabled: false, blend: 0.5 },
    ]);
  });

  it("ignores anything that is not a valid face", () => {
    expect(readRestoredFaces(undefined)).toEqual([]);
    expect(readRestoredFaces({ index: 0 })).toEqual([]);
    expect(
      readRestoredFaces([
        null,
        { index: "0", restored: true, enabled: true, blend: 0.6 },
        { index: 1, restored: true, enabled: true, blend: 2 },
        { index: 2, restored: true, enabled: "yes", blend: 0.5 },
      ]),
    ).toEqual([]);
  });
});

describe("face drafts", () => {
  const faces = [
    { index: 0, enabled: true, blend: 0.6 },
    { index: 3, enabled: false, blend: 0.4 },
  ];

  it("starts from what the result has", () => {
    const draft = draftOf(faces);
    expect(draft).toEqual({ 0: { enabled: true, blend: 0.6 }, 3: { enabled: false, blend: 0.4 } });
    expect(isDraftChanged(faces, draft)).toBe(false);
  });

  it("changes one face without touching the others or the previous draft", () => {
    const draft = draftOf(faces);
    const changed = withFaceChoice(draft, 3, { enabled: true });
    expect(changed).toEqual({ 0: { enabled: true, blend: 0.6 }, 3: { enabled: true, blend: 0.4 } });
    expect(draft[3]).toEqual({ enabled: false, blend: 0.4 });
    expect(isDraftChanged(faces, changed)).toBe(true);
  });

  it("is unchanged again once the face goes back to what the result has", () => {
    const back = withFaceChoice(withFaceChoice(draftOf(faces), 0, { blend: 0.3 }), 0, { blend: 0.6 });
    expect(isDraftChanged(faces, back)).toBe(false);
  });
});
