import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RestoreAnalysis } from "../../../lib/restoreApiTypes";
import { makeAnalysis } from "./restoreTestFixtures";
import { useRestoreSession, type RestoreSessionServices } from "./useRestoreSession";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function makeServices(overrides: Partial<RestoreSessionServices> = {}): RestoreSessionServices {
  return {
    analyzePhoto: vi.fn().mockResolvedValue(makeAnalysis()),
    setPhotoGeometry: vi.fn().mockResolvedValue(makeAnalysis({ width: 800, height: 1200 })),
    uploadDamageMask: vi.fn().mockResolvedValue({ coverage: 0.03, width: 1200, height: 800 }),
    ...overrides,
  };
}

function photo(name = "grandma.jpg"): File {
  return new File(["x"], name, { type: "image/jpeg" });
}

describe("useRestoreSession", () => {
  it("starts idle with nothing analyzed", () => {
    const { result } = renderHook(() => useRestoreSession(makeServices()));

    expect(result.current.phase).toBe("idle");
    expect(result.current.analysis).toBeNull();
  });

  it("analyzes a photo and keeps the token, the faces and a new preview revision", async () => {
    const services = makeServices();
    const { result } = renderHook(() => useRestoreSession(services));

    await act(() => result.current.analyze(photo()));

    expect(result.current.phase).toBe("ready");
    expect(result.current.fileName).toBe("grandma.jpg");
    expect(result.current.analysis?.token).toBe("tok-1");
    expect(result.current.faces).toHaveLength(1);
    expect(result.current.revision).toBe(1);
  });

  it("reports the analysis failure with the server message", async () => {
    const services = makeServices({ analyzePhoto: vi.fn().mockRejectedValue(new Error("Unsupported image")) });
    const { result } = renderHook(() => useRestoreSession(services));

    await act(() => result.current.analyze(photo()));

    expect(result.current.phase).toBe("failed");
    expect(result.current.errorMessage).toBe("Unsupported image");
  });

  it("ignores the answer for a photo that was replaced while it was being analyzed", async () => {
    const first = deferred<RestoreAnalysis>();
    const analyzePhoto = vi
      .fn()
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce(makeAnalysis({ token: "tok-2", originalName: "second.jpg" }));
    const { result } = renderHook(() => useRestoreSession(makeServices({ analyzePhoto })));

    act(() => {
      void result.current.analyze(photo("first.jpg"));
    });
    await act(() => result.current.analyze(photo("second.jpg")));
    await act(async () => first.resolve(makeAnalysis({ token: "tok-1" })));

    expect(result.current.analysis?.token).toBe("tok-2");
    expect(result.current.fileName).toBe("second.jpg");
  });

  it("aborts the upload of the photo it replaces", async () => {
    const analyzePhoto = vi.fn().mockReturnValueOnce(new Promise(() => undefined));
    const { result } = renderHook(() => useRestoreSession(makeServices({ analyzePhoto })));

    act(() => {
      void result.current.analyze(photo("first.jpg"));
    });
    const firstSignal = analyzePhoto.mock.calls[0][1].signal as AbortSignal;
    act(() => result.current.reset());

    expect(firstSignal.aborted).toBe(true);
    expect(result.current.phase).toBe("idle");
  });

  it("sends the geometry for the current token and shows the new working copy", async () => {
    const services = makeServices();
    const { result } = renderHook(() => useRestoreSession(services));
    await act(() => result.current.analyze(photo()));

    await act(() => result.current.applyGeometry({ rotate90: 1, crop: null, angle: 0 }));

    expect(services.setPhotoGeometry).toHaveBeenCalledWith("tok-1", { rotate90: 1, crop: null, angle: 0 });
    expect(result.current.analysis?.width).toBe(800);
    expect(result.current.revision).toBe(2);
  });

  it("is busy while the geometry is being applied", async () => {
    const pending = deferred<RestoreAnalysis>();
    const services = makeServices({ setPhotoGeometry: vi.fn().mockReturnValue(pending.promise) });
    const { result } = renderHook(() => useRestoreSession(services));
    await act(() => result.current.analyze(photo()));

    act(() => {
      void result.current.applyGeometry({ rotate90: 1, crop: null, angle: 0 });
    });

    expect(result.current.phase).toBe("updating");
    await act(async () => pending.resolve(makeAnalysis()));
    expect(result.current.phase).toBe("ready");
  });

  it("keeps the previous analysis when the geometry fails", async () => {
    const services = makeServices({ setPhotoGeometry: vi.fn().mockRejectedValue(new Error("Crop outside")) });
    const { result } = renderHook(() => useRestoreSession(services));
    await act(() => result.current.analyze(photo()));

    await act(() => result.current.applyGeometry({ rotate90: 0, crop: [0, 0, 9999, 1], angle: 0 }));

    expect(result.current.phase).toBe("ready");
    expect(result.current.analysis?.token).toBe("tok-1");
    expect(result.current.errorMessage).toBe("Crop outside");
  });

  it("does not send geometry before a photo was analyzed", async () => {
    const services = makeServices();
    const { result } = renderHook(() => useRestoreSession(services));

    await act(() => result.current.applyGeometry({ rotate90: 1, crop: null, angle: 0 }));

    expect(services.setPhotoGeometry).not.toHaveBeenCalled();
  });

  it("uploads the painted mask and keeps its coverage until the geometry changes", async () => {
    const services = makeServices();
    const { result } = renderHook(() => useRestoreSession(services));
    await act(() => result.current.analyze(photo()));
    const mask = new Blob(["png"], { type: "image/png" });

    let saved = false;
    await act(async () => {
      saved = await result.current.saveMask(mask);
    });

    expect(saved).toBe(true);
    expect(services.uploadDamageMask).toHaveBeenCalledWith("tok-1", mask);
    expect(result.current.maskCoverage).toBe(0.03);
    await act(() => result.current.applyGeometry({ rotate90: 1, crop: null, angle: 0 }));
    await waitFor(() => expect(result.current.maskCoverage).toBeNull());
  });

  it("reports a rejected mask without losing the analysis", async () => {
    const services = makeServices({ uploadDamageMask: vi.fn().mockRejectedValue(new Error("Mask size")) });
    const { result } = renderHook(() => useRestoreSession(services));
    await act(() => result.current.analyze(photo()));

    let saved = true;
    await act(async () => {
      saved = await result.current.saveMask(new Blob(["png"]));
    });

    expect(saved).toBe(false);
    expect(result.current.errorMessage).toBe("Mask size");
    expect(result.current.analysis).not.toBeNull();
  });

  it("changes one face without touching the others", async () => {
    const analysis = makeAnalysis({
      faces: [makeAnalysis().faces[0], { ...makeAnalysis().faces[0], index: 1, blend: 0.4 }],
    });
    const services = makeServices({ analyzePhoto: vi.fn().mockResolvedValue(analysis) });
    const { result } = renderHook(() => useRestoreSession(services));
    await act(() => result.current.analyze(photo()));

    act(() => result.current.updateFace(1, { enabled: false }));

    expect(result.current.faces[0].enabled).toBe(true);
    expect(result.current.faces[1]).toMatchObject({ enabled: false, blend: 0.4 });
    expect(analysis.faces[1].enabled).toBe(true);
  });
});
