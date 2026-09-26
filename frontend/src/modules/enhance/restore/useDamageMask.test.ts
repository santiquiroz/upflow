import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { RestoreAnalysis } from "../../../lib/restoreApiTypes";
import type { BrushStroke } from "../../editor/maskCanvas";
import type { ProbabilityMap } from "./damageMask";
import { makeAnalysis } from "./restoreTestFixtures";
import { useDamageMask, type DamageMaskInput } from "./useDamageMask";

const PROB_URL = "/api/v1/restore/analysis/tok-1/damage_prob.png";
const EMPTY_MAP: ProbabilityMap = { width: 10, height: 10, data: new Uint8Array(100) };
const DEFAULTS = { sensitivity: 0.5, growPx: 0 };
const DOT: BrushStroke = { mode: "paint", radius: 1, points: [{ x: 5, y: 5 }] };

function analysisOf(overrides: Partial<RestoreAnalysis> = {}): RestoreAnalysis {
  const damage = { coverage: 0.2, probUrl: PROB_URL, largeHoles: 0 };
  return makeAnalysis({ width: 10, height: 10, damage, ...overrides });
}

function renderDamage(initial: DamageMaskInput) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client: queryClient }, children);
  return renderHook((input: DamageMaskInput) => useDamageMask(input), { initialProps: initial, wrapper });
}

describe("useDamageMask", () => {
  it("uses the analysis numbers while the map loads and the measured ones after", async () => {
    let resolve: (map: ProbabilityMap) => void = () => undefined;
    const loadMap = vi.fn(() => new Promise<ProbabilityMap>((done) => (resolve = done)));
    const { result } = renderDamage({ analysis: analysisOf(), revision: 1, settings: DEFAULTS, active: true, loadMap });

    expect(result.current.mapStatus).toBe("loading");
    expect(result.current.coverage).toBe(0.2);
    expect(result.current.needsReview).toBe(true);

    act(() => resolve(EMPTY_MAP));

    await waitFor(() => expect(result.current.mapStatus).toBe("ready"));
    expect(result.current.coverage).toBe(0);
    expect(result.current.needsReview).toBe(false);
  });

  it("does nothing while the repair step is off", () => {
    const loadMap = vi.fn();
    const { result } = renderDamage({ analysis: analysisOf(), revision: 1, settings: DEFAULTS, active: false, loadMap });

    expect(loadMap).not.toHaveBeenCalled();
    expect(result.current.mask).toBeNull();
    expect(result.current.needsReview).toBe(false);
  });

  it("drops the strokes and the review when the geometry changes", async () => {
    const loadMap = vi.fn().mockResolvedValue(EMPTY_MAP);
    const input: DamageMaskInput = { analysis: analysisOf(), revision: 1, settings: DEFAULTS, active: true, loadMap };
    const { result, rerender } = renderDamage(input);
    await waitFor(() => expect(result.current.mapStatus).toBe("ready"));

    act(() => result.current.addStroke(DOT));
    act(() => result.current.markReviewed());
    expect(result.current.edited).toBe(true);
    expect(result.current.reviewed).toBe(true);

    rerender({ ...input, revision: 2 });

    expect(result.current.edited).toBe(false);
    expect(result.current.reviewed).toBe(false);
  });

  it("needs a new review after another stroke", async () => {
    const loadMap = vi.fn().mockResolvedValue(EMPTY_MAP);
    const { result } = renderDamage({ analysis: analysisOf(), revision: 1, settings: DEFAULTS, active: true, loadMap });
    await waitFor(() => expect(result.current.mapStatus).toBe("ready"));

    act(() => result.current.markReviewed());
    act(() => result.current.addStroke(DOT));

    expect(result.current.reviewed).toBe(false);
  });

  it("encodes the current mask as a PNG", async () => {
    const loadMap = vi.fn().mockResolvedValue(EMPTY_MAP);
    const { result } = renderDamage({ analysis: analysisOf(), revision: 1, settings: DEFAULTS, active: true, loadMap });
    await waitFor(() => expect(result.current.mapStatus).toBe("ready"));

    const blob = await result.current.maskBlob();

    expect(blob.type).toBe("image/png");
  });
});
