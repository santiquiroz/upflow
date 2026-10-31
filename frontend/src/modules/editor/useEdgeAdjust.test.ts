import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NO_EDGE, type EdgeAdjust } from "./editorMaskCanvas";
import { EDGE_PREVIEW_DEBOUNCE_MS, useEdgeAdjust } from "./useEdgeAdjust";

const PHOTO = {};

function renderEdgeAdjust() {
  const previews: (EdgeAdjust | null)[] = [];
  const hook = renderHook(() => {
    const state = useEdgeAdjust({ mode: "erase", modelId: "eraser-auto", models: [], photo: PHOTO });
    if (previews[previews.length - 1] !== state.preview) previews.push(state.preview);
    return state;
  });
  return { ...hook, previews };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("useEdgeAdjust", () => {
  it("recomputes the preview at most once per debounce window while the slider moves fast", () => {
    const { result, previews } = renderEdgeAdjust();

    for (const growPx of [1, 2, 3, 4, 5]) {
      act(() => result.current.setEdge({ growPx, featherPx: 0 }));
      act(() => {
        vi.advanceTimersByTime(EDGE_PREVIEW_DEBOUNCE_MS / 5);
      });
    }
    expect(previews).toEqual([NO_EDGE]);

    act(() => {
      vi.advanceTimersByTime(EDGE_PREVIEW_DEBOUNCE_MS);
    });

    expect(previews).toEqual([NO_EDGE, { growPx: 5, featherPx: 0 }]);
  });

  it("keeps the value sent with the job exact and immediate", () => {
    const { result } = renderEdgeAdjust();

    act(() => result.current.setEdge({ growPx: 12, featherPx: 5 }));

    expect(result.current.edge).toEqual({ growPx: 12, featherPx: 5 });
    expect(result.current.preview).toBe(NO_EDGE);
  });

  it("shows no preview outside erase mode", () => {
    const { result } = renderHook(() =>
      useEdgeAdjust({ mode: "replace", modelId: "eraser-auto", models: [], photo: PHOTO }),
    );

    expect(result.current.edgeActive).toBe(false);
    expect(result.current.preview).toBeNull();
  });
});
