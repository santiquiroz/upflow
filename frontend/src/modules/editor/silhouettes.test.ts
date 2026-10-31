import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as editorService from "../../services/editor";
import type { EditorPerson } from "../../services/editor";
import type { CropMask } from "./editorMaskCanvas";
import { lastPeopleBatch, silhouettesFor } from "./silhouettes";

vi.mock("../../services/editor", () => ({ segmentEditorBox: vi.fn() }));

function person(id: number, role: EditorPerson["role"] = "background"): EditorPerson {
  return { id, box: [id, 0, id + 10, 50], score: 0.9, role, face: null, anchor: [id + 5, 25] };
}

// Como el recorte real: un lienzo del tamaño de la caja, ubicado en su origen dentro de la foto.
function fakeMask(_blob: Blob, box: [number, number, number, number]): Promise<CropMask> {
  const mask = document.createElement("canvas");
  mask.width = box[2] - box[0];
  mask.height = box[3] - box[1];
  return Promise.resolve({ mask, origin: { x: box[0], y: box[1] } });
}

beforeEach(() => {
  // jsdom no trae canvas 2D: la unión de recortes solo necesita dibujar, así que basta con un contexto que lo acepte.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => ({ drawImage: vi.fn() }) as never);
});

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
    expect(entry.entries?.map((item) => item.kind)).toEqual(["segment", "protect"]);
    const [segment, protect] = entry.entries ?? [];
    expect(segment.mask).toBeInstanceOf(HTMLCanvasElement);
    expect(protect.points).toEqual([{ x: 6, y: 25 }]);
  });

  it("merges the removed people into one canvas sized to their boxes", async () => {
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"]));

    const { entry } = await silhouettesFor(
      "tok", [person(1), person(2), person(3)], { 1: "remove", 2: "remove", 3: "remove" }, false, { toMask: fakeMask },
    );

    const segments = entry.entries?.filter((item) => item.kind === "segment") ?? [];
    expect(segments).toHaveLength(1);
    expect(segments[0].origin).toEqual({ x: 1, y: 0 });
    expect([segments[0].mask?.width, segments[0].mask?.height]).toEqual([12, 50]);
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
