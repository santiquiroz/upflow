import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useUndoLog } from "./useUndoLog";

describe("useUndoLog", () => {
  it("undoes the last action first: a base edit made after a mask entry comes off before it", () => {
    const { result } = renderHook(() => useUndoLog());
    let maskId = 0;
    act(() => {
      maskId = result.current.record("mask");
      result.current.record("base");
    });
    expect(result.current.last?.kind).toBe("base");

    act(() => result.current.remove(result.current.last!.id));
    expect(result.current.last).toEqual({ id: maskId, kind: "mask" });

    act(() => result.current.remove(maskId));
    expect(result.current.last).toBeNull();
  });

  it("removes an action by id, not by position", () => {
    const { result } = renderHook(() => useUndoLog());
    let first = 0;
    let second = 0;
    act(() => {
      first = result.current.record("mask");
      second = result.current.record("mask");
    });
    act(() => result.current.remove(first));
    expect(result.current.last?.id).toBe(second);
  });

  it("keeps base actions and only the mask actions whose entries survive", () => {
    const { result } = renderHook(() => useUndoLog());
    let kept = 0;
    act(() => {
      kept = result.current.record("mask");
      result.current.record("mask");
      result.current.record("base");
    });
    act(() => result.current.keepMasks(new Set([kept])));
    act(() => result.current.remove(kept));
    expect(result.current.last?.kind).toBe("base");
  });

  it("removeOldest drops the oldest action of a kind and keeps the rest in order", () => {
    const { result } = renderHook(() => useUndoLog());
    let mask = 0;
    let newestBase = 0;
    act(() => {
      result.current.record("base");
      mask = result.current.record("mask");
      newestBase = result.current.record("base");
    });
    act(() => result.current.removeOldest("base"));
    expect(result.current.actions).toEqual([
      { id: mask, kind: "mask" },
      { id: newestBase, kind: "base" },
    ]);
  });

  it("clear empties the log", () => {
    const { result } = renderHook(() => useUndoLog());
    act(() => {
      result.current.record("mask");
      result.current.record("base");
    });
    act(() => result.current.clear());
    expect(result.current.last).toBeNull();
  });
});
