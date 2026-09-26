import { describe, expect, it, vi } from "vitest";
import { createEditorHandoffStore } from "./editorHandoffStore";

const HANDOFF = { url: "/api/v1/jobs/job-1/download", fileName: "grandma_restored.png" };

describe("editorHandoffStore", () => {
  it("starts empty", () => {
    expect(createEditorHandoffStore().getSnapshot()).toBeNull();
  });

  it("hands the offered image over exactly once", () => {
    const store = createEditorHandoffStore();
    store.offer(HANDOFF);
    expect(store.getSnapshot()).toEqual(HANDOFF);
    expect(store.take()).toEqual(HANDOFF);
    expect(store.take()).toBeNull();
    expect(store.getSnapshot()).toBeNull();
  });

  it("tells subscribers when an image is offered and taken", () => {
    const store = createEditorHandoffStore();
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    store.offer(HANDOFF);
    store.take();
    store.take();
    expect(listener).toHaveBeenCalledTimes(2);
    unsubscribe();
    store.offer(HANDOFF);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("keeps only the latest offer", () => {
    const store = createEditorHandoffStore();
    store.offer(HANDOFF);
    store.offer({ url: "/other.png", fileName: "other.png" });
    expect(store.take()).toEqual({ url: "/other.png", fileName: "other.png" });
  });
});
