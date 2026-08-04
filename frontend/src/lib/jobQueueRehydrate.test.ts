import { describe, expect, it, vi } from "vitest";
import { createJobQueueStore } from "./jobQueueStore";
import { rehydrateJobQueue } from "./jobQueueRehydrate";

// La cola vivia solo en memoria: recargar el navegador la borraba aunque los
// trabajos siguieran corriendo en el servidor. Se rehidrata desde el servidor y
// NO desde localStorage, porque el servidor es la fuente real: asi no pueden
// aparecer fantasmas de trabajos que ya no existen.

function respuestas(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    image: { jobs: [] },
    video: { jobs: [] },
    audio: { jobs: [] },
    generation: { jobs: [] },
    ...overrides,
  };
}

function fetchers(data: Record<string, unknown>) {
  return {
    fetchImageJobs: vi.fn().mockResolvedValue(data.image),
    fetchVideoJobs: vi.fn().mockResolvedValue(data.video),
    fetchAudioJobs: vi.fn().mockResolvedValue(data.audio),
    fetchGenerationJobs: vi.fn().mockResolvedValue(data.generation),
  };
}

describe("rehydrateJobQueue", () => {
  it("brings back the jobs that are still running", async () => {
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          video: {
            jobs: [
              { jobId: "v1", status: "running", originalFilename: "capitulo.mkv" },
              { jobId: "v2", status: "queued", originalFilename: "otro.mkv" },
            ],
          },
        }),
      ),
    );

    expect(store.getSnapshot().map((j) => j.id)).toEqual(["v1", "v2"]);
    expect(store.getSnapshot()[0].fileName).toBe("capitulo.mkv");
    expect(store.getSnapshot()[0].kind).toBe("video");
  });

  it("leaves out the ones that already finished", async () => {
    // Traer un trabajo terminado seria peor que no traer nada: la cola diria
    // que hay algo corriendo cuando no lo hay.
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          image: {
            jobs: [
              { jobId: "i1", status: "completed", originalFilename: "a.png" },
              { jobId: "i2", status: "failed", originalFilename: "b.png" },
              { jobId: "i3", status: "cancelled", originalFilename: "c.png" },
              { jobId: "i4", status: "running", originalFilename: "d.png" },
            ],
          },
        }),
      ),
    );

    expect(store.getSnapshot().map((j) => j.id)).toEqual(["i4"]);
  });

  it("mixes every kind of job", async () => {
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          image: { jobs: [{ jobId: "i1", status: "running", originalFilename: "a.png" }] },
          video: { jobs: [{ jobId: "v1", status: "running", originalFilename: "b.mkv" }] },
          audio: { jobs: [{ id: "a1", status: "running", originalFilename: "c.flac" }] },
          generation: { jobs: [{ id: "g1", status: "running", prompt: "un zorro" }] },
        }),
      ),
    );

    const kinds = store.getSnapshot().map((j) => j.kind).sort();
    expect(kinds).toEqual(["audio", "generation", "image", "video"]);
  });

  it("uses the prompt as the name of a generation job, which has no file", async () => {
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          generation: {
            jobs: [{ id: "g1", status: "queued", prompt: "un gato con sombrero" }],
          },
        }),
      ),
    );

    expect(store.getSnapshot()[0].fileName).toBe("un gato con sombrero");
  });

  it("does not duplicate what the store is already tracking", async () => {
    const store = createJobQueueStore();
    store.addTrackedJob({ id: "v1", kind: "video", fileName: "capitulo.mkv", createdAt: 1 });

    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          video: { jobs: [{ jobId: "v1", status: "running", originalFilename: "capitulo.mkv" }] },
        }),
      ),
    );

    expect(store.getSnapshot()).toHaveLength(1);
  });

  it("one broken endpoint does not lose the rest of the queue", async () => {
    const store = createJobQueueStore();
    const calls = fetchers(
      respuestas({
        video: { jobs: [{ jobId: "v1", status: "running", originalFilename: "b.mkv" }] },
      }),
    );
    calls.fetchAudioJobs = vi.fn().mockRejectedValue(new Error("500"));

    await rehydrateJobQueue(store, calls);

    expect(store.getSnapshot().map((j) => j.id)).toEqual(["v1"]);
  });

  it("never throws, so a failure cannot break the app on startup", async () => {
    const store = createJobQueueStore();
    const calls = fetchers(respuestas());
    calls.fetchImageJobs = vi.fn().mockRejectedValue(new Error("sin red"));
    calls.fetchVideoJobs = vi.fn().mockRejectedValue(new Error("sin red"));
    calls.fetchAudioJobs = vi.fn().mockRejectedValue(new Error("sin red"));
    calls.fetchGenerationJobs = vi.fn().mockRejectedValue(new Error("sin red"));

    await expect(rehydrateJobQueue(store, calls)).resolves.toBeUndefined();
    expect(store.getSnapshot()).toEqual([]);
  });
});
