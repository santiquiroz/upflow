import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UseGenerationJobResult } from "../../hooks/useGenerationJob";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type { EraseZone, GenerationJob } from "../../lib/apiTypes";
import * as generationService from "../../services/generation";
import type { ReviewSession } from "./useReviewSession";
import { useZoneRedo } from "./useZoneRedo";

const generationJob: UseGenerationJobResult = vi.hoisted(() => ({
  phase: "idle" as const,
  job: undefined,
  errorMessage: null,
  submit: vi.fn(),
  cancel: vi.fn(),
  reset: vi.fn(),
}));

vi.mock("../../hooks/useGenerationJob", () => ({ useGenerationJob: () => generationJob }));
vi.mock("../../services/generation", () => ({ uploadGenerationInitImage: vi.fn() }));
vi.mock("./editorMaskCanvas", () => ({ exportImageDataMaskBlob: vi.fn(async () => new Blob(["mask"])) }));

const ZONE: EraseZone = { index: 0, box: [0, 0, 2, 2], engine: "lama+osor", guard: null, residualPerson: false };

const SESSION: ReviewSession = {
  eraseMask: { data: new Uint8ClampedArray(16).fill(255), width: 2, height: 2, colorSpace: "srgb" } as ImageData,
  maskGrowPx: 0,
  maskFeatherPx: 0,
  zones: [ZONE],
  resultUrl: "/erase.png",
};

function wrapper({ children }: { children: ReactNode }) {
  return <LocaleProvider initialLocale="en">{children}</LocaleProvider>;
}

function renderRedo(onDone: (zone: EraseZone, resultUrl: string | null) => void) {
  return renderHook(
    (props: { onDone: typeof onDone }) => useZoneRedo({
      session: SESSION,
      baseInfo: { initImageToken: "base", originalFilename: "foto.png", width: 2, height: 2 },
      models: [{ id: "eraser-auto", name: "Auto", status: "installed", eraseOnly: true, supportsInpaint: true }],
      defaults: { prompt: "background", negativePrompt: null, steps: 30, guidance: 7, device: null },
      onDone: props.onDone,
    }),
    { wrapper, initialProps: { onDone } },
  );
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async () => ({ blob: async () => new Blob(["result"]) })));
  vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValue({
    initImageToken: "token", originalFilename: "x.png", width: 2, height: 2,
  });
});

afterEach(() => {
  generationJob.job = undefined;
  vi.mocked(generationJob.submit).mockClear();
  vi.unstubAllGlobals();
});

describe("useZoneRedo", () => {
  it("calls the latest onDone when the redo finishes", async () => {
    const first = vi.fn();
    const latest = vi.fn();
    const { result, rerender } = renderRedo(first);

    await act(() => result.current.redo(ZONE, "lama"));
    await waitFor(() => expect(generationJob.submit).toHaveBeenCalled());
    rerender({ onDone: latest });
    generationJob.job = { status: "completed", downloadUrl: "/redo.png", eraseZones: [{ ...ZONE, engine: "lama" }] } as GenerationJob;
    rerender({ onDone: latest });

    expect(first).not.toHaveBeenCalled();
    expect(latest).toHaveBeenCalledWith(expect.objectContaining({ index: 0, engine: "lama" }), "/redo.png");
    expect(result.current.redoingIndex).toBeNull();
  });

  it("reports the running job progress while a zone is redone", async () => {
    const { result, rerender } = renderRedo(vi.fn());

    await act(() => result.current.redo(ZONE, "lama"));
    generationJob.job = { status: "running", progressPct: 40, downloadUrl: null } as GenerationJob;
    rerender({ onDone: vi.fn() });

    expect(result.current.redoingIndex).toBe(0);
    expect(result.current.progressPct).toBe(40);
  });
});
