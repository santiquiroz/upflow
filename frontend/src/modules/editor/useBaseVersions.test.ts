import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { InitImageResponse } from "../../lib/apiTypes";
import { uploadGenerationInitImage } from "../../services/generation";
import { MAX_BASE_VERSIONS, useBaseVersions } from "./useBaseVersions";

vi.mock("../../services/generation", () => ({ uploadGenerationInitImage: vi.fn() }));

let uploads = 0;

beforeEach(() => {
  uploads = 0;
  vi.mocked(uploadGenerationInitImage).mockImplementation(async () => {
    uploads += 1;
    return { initImageToken: `tok-${uploads}`, width: 4, height: 4, originalFilename: "foto.png" } as InitImageResponse;
  });
});

function photo(name: string): File {
  return new File([name], name, { type: "image/png" });
}

async function setup() {
  const hook = renderHook(() => useBaseVersions());
  await act(async () => {
    await hook.result.current.reset(photo("foto.png"));
  });
  return hook;
}

describe("useBaseVersions", () => {
  it("push adds a version and undo goes back one, never past the original", async () => {
    const { result } = await setup();
    await act(async () => {
      await result.current.push(photo("edit-1.png"));
    });
    expect(result.current.current?.initImageToken).toBe("tok-2");

    let undone = false;
    act(() => {
      undone = result.current.undo();
    });
    expect(undone).toBe(true);
    expect(result.current.current?.initImageToken).toBe("tok-1");

    act(() => {
      undone = result.current.undo();
    });
    expect(undone).toBe(false);
    expect(result.current.current?.initImageToken).toBe("tok-1");
  });

  it("caps the versions at ten and keeps the original", async () => {
    const { result } = await setup();
    for (let index = 0; index < MAX_BASE_VERSIONS + 2; index += 1) {
      await act(async () => {
        await result.current.push(photo(`edit-${index}.png`));
      });
    }
    expect(result.current.original?.info.initImageToken).toBe("tok-1");
    expect(result.current.current?.initImageToken).toBe(`tok-${MAX_BASE_VERSIONS + 3}`);

    let undos = 0;
    act(() => {
      while (result.current.undo()) undos += 1;
    });
    expect(undos).toBe(MAX_BASE_VERSIONS - 1);
    expect(result.current.current?.initImageToken).toBe("tok-1");
  });

  it("push reports when the cap dropped an old version", async () => {
    const { result } = await setup();
    const evictions: boolean[] = [];
    for (let index = 0; index < MAX_BASE_VERSIONS; index += 1) {
      await act(async () => {
        evictions.push(await result.current.push(photo(`edit-${index}.png`)));
      });
    }
    expect(evictions).toEqual([...Array(MAX_BASE_VERSIONS - 1).fill(false), true]);
  });

  it("reset starts a new session with the new photo as the original", async () => {
    const { result } = await setup();
    await act(async () => {
      await result.current.push(photo("edit.png"));
    });
    await act(async () => {
      await result.current.reset(photo("otra.png"));
    });
    expect(result.current.original?.info.initImageToken).toBe("tok-3");
    expect(result.current.current?.initImageToken).toBe("tok-3");
    expect(result.current.undo()).toBe(false);
  });

  it("keeps the original unchanged when a result is pushed as the new base", async () => {
    const { result } = await setup();
    const originalBlob = result.current.original?.blob;
    await act(async () => {
      await result.current.push(photo("result.png"));
    });
    expect(result.current.original?.blob).toBe(originalBlob);
    expect(result.current.original?.info.initImageToken).toBe("tok-1");
  });

  it("drops a push that finishes after a new photo was opened", async () => {
    const { result } = await setup();
    let finishUpload: (value: InitImageResponse) => void = () => {};
    vi.mocked(uploadGenerationInitImage).mockImplementationOnce(
      () => new Promise<InitImageResponse>((resolve) => { finishUpload = resolve; }),
    );
    let pending: Promise<boolean> = Promise.resolve(false);
    act(() => {
      pending = result.current.push(photo("late.png"));
    });
    await act(async () => {
      await result.current.reset(photo("nueva.png"));
    });
    await act(async () => {
      finishUpload({ initImageToken: "tok-late", width: 4, height: 4, originalFilename: "late.png" } as InitImageResponse);
      await pending;
    });
    expect(result.current.current?.initImageToken).toBe("tok-2");
    expect(result.current.undo()).toBe(false);
  });
});
