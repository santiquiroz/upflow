import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../lib/api";
import { detectPeople, segmentEditorBox, segmentEditorObject } from "./editor";

afterEach(() => {
  vi.unstubAllGlobals();
});

function mockFetch(response: Response) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

describe("detectPeople", () => {
  it("posts the image token and returns the people array", async () => {
    const people = [{ id: 1, role: "subject" }];
    mockFetch(new Response(JSON.stringify({ people }), { status: 200 }));

    await expect(detectPeople("image-token")).resolves.toEqual(people);

    expect(fetch).toHaveBeenCalledWith("/api/v1/editor/people", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageToken: "image-token" }),
    });
  });

  it("throws an ApiError with the backend detail on failure", async () => {
    mockFetch(new Response(JSON.stringify({ detail: "Detector pack missing" }), { status: 503 }));

    await expect(detectPeople("image-token")).rejects.toEqual(
      expect.objectContaining({
        name: "ApiError",
        status: 503,
        message: "Detector pack missing",
      }),
    );
  });
});

describe("editor segmentation", () => {
  it("segments an object by point", async () => {
    const mask = new Blob(["point-mask"], { type: "image/png" });
    mockFetch(new Response(mask, { status: 200 }));

    await expect(segmentEditorObject("image-token", 12, 34)).resolves.toBeInstanceOf(Blob);

    expect(fetch).toHaveBeenCalledWith("/api/v1/editor/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageToken: "image-token", x: 12, y: 34 }),
    });
  });

  it("segments an object by box", async () => {
    const box: [number, number, number, number] = [1, 2, 30, 40];
    mockFetch(new Response(new Blob(["box-mask"], { type: "image/png" }), { status: 200 }));

    await segmentEditorBox("image-token", box);

    expect(fetch).toHaveBeenCalledWith("/api/v1/editor/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageToken: "image-token", box }),
    });
  });

  it("uses the backend error detail for box segmentation failures", async () => {
    mockFetch(new Response(JSON.stringify({ detail: "Invalid image token" }), { status: 400 }));

    await expect(segmentEditorBox("bad-token", [0, 0, 1, 1])).rejects.toBeInstanceOf(ApiError);

    mockFetch(new Response(JSON.stringify({ detail: "Invalid image token" }), { status: 400 }));
    await expect(segmentEditorObject("bad-token", 0, 0)).rejects.toMatchObject({
      status: 400,
      message: "Invalid image token",
    });
  });
});
