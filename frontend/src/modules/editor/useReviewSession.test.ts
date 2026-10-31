import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { EraseZone, GenerationJob } from "../../lib/apiTypes";
import { useReviewSession } from "./useReviewSession";

function zone(index: number, engine = "lama+osor"): EraseZone {
  return { index, box: [0, 0, 10, 10], engine, guard: null, residualPerson: false };
}

function completed(downloadUrl: string, zones: EraseZone[]): GenerationJob {
  return { status: "completed", downloadUrl, eraseZones: zones } as GenerationJob;
}

function mask(): ImageData {
  return { data: new Uint8ClampedArray(4), width: 1, height: 1, colorSpace: "srgb" } as ImageData;
}

function startedSession() {
  const firstMask = mask();
  const hook = renderHook(({ job }: { job: GenerationJob | undefined }) => useReviewSession(job), {
    initialProps: { job: undefined as GenerationJob | undefined },
  });
  act(() => hook.result.current.start(firstMask, "protect"));
  hook.rerender({ job: completed("/erase.png", [zone(0), zone(1)]) });
  return { ...hook, firstMask };
}

describe("useReviewSession", () => {
  it("adopts the zones and result of the completed erase", () => {
    const { result } = startedSession();

    expect(result.current.zones).toEqual([zone(0), zone(1)]);
    expect(result.current.session?.resultUrl).toBe("/erase.png");
  });

  it("a redo updates one row and the result without overwriting the session", () => {
    const { result, rerender, firstMask } = startedSession();

    act(() => result.current.updateZone(zone(1, "lama"), "/redo.png"));
    rerender({ job: completed("/other.png", [zone(0, "osor")]) });

    expect(result.current.zones).toEqual([zone(0), zone(1, "lama")]);
    expect(result.current.session).toEqual(expect.objectContaining({
      eraseMask: firstMask, protectMaskToken: "protect", resultUrl: "/redo.png",
    }));
  });

  it("stores the grow and feather used on submit so a redo reuses them", () => {
    const hook = renderHook(() => useReviewSession(undefined));

    act(() => hook.result.current.start(mask(), "protect", { growPx: 12, featherPx: 5 }));

    expect(hook.result.current.session).toEqual(expect.objectContaining({ maskGrowPx: 12, maskFeatherPx: 5 }));
  });

  it("a new erase submit restarts the session", () => {
    const { result, rerender } = startedSession();
    const nextMask = mask();

    act(() => result.current.updateZone(zone(1, "lama"), "/redo.png"));
    act(() => result.current.start(nextMask));

    expect(result.current.zones).toBeNull();
    expect(result.current.session).toEqual(expect.objectContaining({ eraseMask: nextMask, resultUrl: null }));
    rerender({ job: completed("/erase-2.png", [zone(0, "osor")]) });
    expect(result.current.zones).toEqual([zone(0, "osor")]);
    expect(result.current.session?.resultUrl).toBe("/erase-2.png");
  });
});
