import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createEditorHandoffStore } from "../../lib/editorHandoffStore";
import { useEditorHandoff } from "./useEditorHandoff";

const HANDOFF = { url: "/api/v1/jobs/job-1/download", fileName: "grandma_restored.png" };

function setup(fetchBlob = vi.fn().mockResolvedValue(new Blob(["png"], { type: "image/png" }))) {
  const store = createEditorHandoffStore();
  const onFile = vi.fn();
  const onError = vi.fn();
  renderHook(() => useEditorHandoff(onFile, onError, { store, fetchBlob }));
  return { store, onFile, onError, fetchBlob };
}

describe("useEditorHandoff", () => {
  it("opens an image offered after the Editor is mounted, once", async () => {
    const { store, onFile, fetchBlob } = setup();
    act(() => store.offer(HANDOFF));
    await waitFor(() => expect(onFile).toHaveBeenCalledTimes(1));
    const file = onFile.mock.calls[0][0] as File;
    expect(file.name).toBe("grandma_restored.png");
    expect(file.type).toBe("image/png");
    expect(fetchBlob).toHaveBeenCalledWith(HANDOFF.url);
    expect(store.getSnapshot()).toBeNull();
  });

  it("opens an image offered before the Editor was mounted", async () => {
    const store = createEditorHandoffStore();
    store.offer(HANDOFF);
    const onFile = vi.fn();
    const fetchBlob = vi.fn().mockResolvedValue(new Blob(["jpg"], { type: "image/jpeg" }));
    renderHook(() => useEditorHandoff(onFile, vi.fn(), { store, fetchBlob }));
    await waitFor(() => expect(onFile).toHaveBeenCalledTimes(1));
    expect(fetchBlob).toHaveBeenCalledTimes(1);
  });

  it("reports a readable error when the image cannot be fetched", async () => {
    const { store, onFile, onError } = setup(vi.fn().mockRejectedValue(new Error("HTTP 404")));
    act(() => store.offer(HANDOFF));
    await waitFor(() => expect(onError).toHaveBeenCalledWith("Could not open grandma_restored.png in the Editor."));
    expect(onFile).not.toHaveBeenCalled();
  });
});
