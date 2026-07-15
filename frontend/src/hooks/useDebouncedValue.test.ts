import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDebouncedValue } from "./useDebouncedValue";

const DEBOUNCE_MS = 300;

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("useDebouncedValue", () => {
  it("returns the initial value immediately", () => {
    const { result } = renderHook(() => useDebouncedValue("a", DEBOUNCE_MS));
    expect(result.current).toBe("a");
  });

  it("does not update before the delay elapses", () => {
    const { result, rerender } = renderHook(({ value }) => useDebouncedValue(value, DEBOUNCE_MS), {
      initialProps: { value: "a" },
    });

    rerender({ value: "ab" });
    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_MS - 1);
    });

    expect(result.current).toBe("a");
  });

  it("updates to the latest value once the delay elapses", () => {
    const { result, rerender } = renderHook(({ value }) => useDebouncedValue(value, DEBOUNCE_MS), {
      initialProps: { value: "a" },
    });

    rerender({ value: "ab" });
    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_MS);
    });

    expect(result.current).toBe("ab");
  });

  it("resets the timer on every rapid change and only settles on the final value", () => {
    const { result, rerender } = renderHook(({ value }) => useDebouncedValue(value, DEBOUNCE_MS), {
      initialProps: { value: "a" },
    });

    rerender({ value: "ab" });
    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_MS - 50);
    });
    rerender({ value: "abc" });
    act(() => {
      vi.advanceTimersByTime(DEBOUNCE_MS - 50);
    });

    expect(result.current).toBe("a");

    act(() => {
      vi.advanceTimersByTime(50);
    });

    expect(result.current).toBe("abc");
  });
});
