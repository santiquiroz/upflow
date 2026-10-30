import { afterEach, describe, expect, it, vi } from "vitest";
import * as editorService from "../../services/editor";
import type { EditorPerson } from "../../services/editor";
import { lastPeopleBatch, silhouettesFor } from "./silhouettes";

vi.mock("../../services/editor", () => ({ segmentEditorBox: vi.fn() }));

function person(id: number, role: EditorPerson["role"] = "background"): EditorPerson {
  return { id, box: [id, 0, id + 10, 50], score: 0.9, role, face: null, anchor: [id + 5, 25] };
}

function fakeMask(blob: Blob): Promise<HTMLCanvasElement> {
  const canvas = document.createElement("canvas");
  canvas.dataset.source = String(blob.size);
  return Promise.resolve(canvas);
}

afterEach(() => vi.mocked(editorService.segmentEditorBox).mockReset());

describe("silhouettesFor", () => {
  it("segments one person at a time, in order, and batches the removed ones", async () => {
    const pending: Array<() => void> = [];
    vi.mocked(editorService.segmentEditorBox).mockImplementation(() =>
      new Promise((resolve) => { pending.push(() => resolve(new Blob(["png"]))); }),
    );
    const people = [person(1, "subject"), person(2), person(3)];

    const running = silhouettesFor("tok", people, { 1: "keep", 2: "remove", 3: "remove" }, true, { toMask: fakeMask });
    await vi.waitFor(() => expect(pending).toHaveLength(1));
    pending[0]();
    await vi.waitFor(() => expect(pending).toHaveLength(2));
    pending[1]();
    await vi.waitFor(() => expect(pending).toHaveLength(3));
    pending[2]();
    const { entry, failed } = await running;

    expect(vi.mocked(editorService.segmentEditorBox).mock.calls.map(([, box]) => box[0])).toEqual([1, 2, 3]);
    expect(failed).toBe(0);
    expect(entry.kind).toBe("batch");
    expect(entry.personIds).toEqual([2, 3]);
    expect(entry.entries?.map((item) => item.kind)).toEqual(["protect", "segment", "segment"]);
    expect(entry.entries?.[1].point).toEqual({ x: 7, y: 25 });
    expect(entry.entries?.[1].mask).toBeInstanceOf(HTMLCanvasElement);
  });

  it("skips kept people when they are not protected", async () => {
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"]));

    const { entry } = await silhouettesFor(
      "tok", [person(1, "subject"), person(2)], { 1: "keep", 2: "remove" }, false, { toMask: fakeMask },
    );

    expect(editorService.segmentEditorBox).toHaveBeenCalledOnce();
    expect(editorService.segmentEditorBox).toHaveBeenCalledWith("tok", [2, 0, 12, 50]);
    expect(entry.entries?.map((item) => item.kind)).toEqual(["segment"]);
  });

  it("reports progress after each person", async () => {
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"]));
    const onProgress = vi.fn();

    await silhouettesFor("tok", [person(1), person(2)], {}, true, { toMask: fakeMask, onProgress });

    expect(onProgress.mock.calls).toEqual([[1, 2], [2, 2]]);
  });

  it("counts a failed person and keeps the others", async () => {
    vi.mocked(editorService.segmentEditorBox)
      .mockResolvedValueOnce(new Blob(["png"]))
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce(new Blob(["png"]));

    const { entry, failed } = await silhouettesFor(
      "tok", [person(1), person(2), person(3)], {}, true, { toMask: fakeMask },
    );

    expect(failed).toBe(1);
    expect(entry.personIds).toEqual([1, 3]);
  });

  it("returns an empty batch when every request fails", async () => {
    vi.mocked(editorService.segmentEditorBox).mockRejectedValue(new Error("down"));

    const { entry, failed } = await silhouettesFor("tok", [person(1), person(2)], {}, true, { toMask: fakeMask });

    expect(failed).toBe(2);
    expect(entry.entries).toEqual([]);
    expect(entry.personIds).toEqual([]);
  });
});

describe("lastPeopleBatch", () => {
  it("finds the last batch that removes people", () => {
    const older = { kind: "batch" as const, personIds: [1], entries: [] };
    const newer = { kind: "batch" as const, personIds: [2], entries: [] };

    expect(lastPeopleBatch([older, newer, { kind: "batch", personIds: [], entries: [] }, { kind: "segment" }])).toBe(newer);
    expect(lastPeopleBatch([{ kind: "segment" }])).toBeNull();
  });
});
