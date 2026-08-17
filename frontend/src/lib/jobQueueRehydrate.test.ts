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
    transcribe: { jobs: [] },
    download: { jobs: [] },
    shape3d: { jobs: [] },
    ...overrides,
  };
}

function fetchers(data: Record<string, unknown>) {
  return {
    fetchImageJobs: vi.fn().mockResolvedValue(data.image),
    fetchVideoJobs: vi.fn().mockResolvedValue(data.video),
    fetchAudioJobs: vi.fn().mockResolvedValue(data.audio),
    fetchGenerationJobs: vi.fn().mockResolvedValue(data.generation),
    fetchTranscribeJobs: vi.fn().mockResolvedValue(data.transcribe),
    fetchDownloadJobs: vi.fn().mockResolvedValue(data.download),
    fetchShape3dJobs: vi.fn().mockResolvedValue(data.shape3d),
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
          transcribe: { jobs: [{ id: "t1", status: "running", originalFilename: "d.wav" }] },
          download: { jobs: [{ id: "d1", status: "running", mediaTitle: "Un clip" }] },
          shape3d: { jobs: [{ id: "s1", status: "running", prompt: "una traba" }] },
        }),
      ),
    );

    const kinds = store.getSnapshot().map((j) => j.kind).sort();
    expect(kinds).toEqual([
      "audio",
      "download",
      "generation",
      "image",
      "shape3d",
      "transcribe",
      "video",
    ]);
  });

  it("brings back a transcription that survived the reload", async () => {
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          transcribe: {
            jobs: [
              { id: "t1", status: "running", originalFilename: "charla.wav" },
              { id: "t2", status: "completed", originalFilename: "vieja.wav" },
            ],
          },
        }),
      ),
    );

    expect(store.getSnapshot().map((j) => j.id)).toEqual(["t1"]);
    expect(store.getSnapshot()[0].fileName).toBe("charla.wav");
    expect(store.getSnapshot()[0].kind).toBe("transcribe");
  });

  it("names a download by its title, and falls back to the URL", async () => {
    // Una descarga no sube ningun archivo: su nombre es lo que el probe encontro,
    // y si no encontro nada, la URL que el usuario pego.
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          download: {
            jobs: [
              { id: "d1", status: "running", mediaTitle: "Un clip", url: "https://x/1" },
              { id: "d2", status: "queued", mediaTitle: null, url: "https://x/2" },
            ],
          },
        }),
      ),
    );

    expect(store.getSnapshot().map((j) => j.fileName)).toEqual(["Un clip", "https://x/2"]);
    expect(store.getSnapshot().map((j) => j.kind)).toEqual(["download", "download"]);
  });

  it("keeps a 3D job identifiable even in photo mode, which has no prompt", async () => {
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          shape3d: {
            jobs: [
              { id: "s1", status: "running", prompt: "una traba" },
              { id: "s2", status: "queued", prompt: "" },
            ],
          },
        }),
      ),
    );

    expect(store.getSnapshot().map((j) => j.fileName)).toEqual(["una traba", "s2"]);
    expect(store.getSnapshot().map((j) => j.kind)).toEqual(["shape3d", "shape3d"]);
  });

  it("leaves out finished jobs of the three families that had no listing", async () => {
    const store = createJobQueueStore();
    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          transcribe: { jobs: [{ id: "t1", status: "completed", originalFilename: "a.wav" }] },
          download: { jobs: [{ id: "d1", status: "failed", mediaTitle: "Un clip" }] },
          shape3d: { jobs: [{ id: "s1", status: "cancelled", prompt: "una traba" }] },
        }),
      ),
    );

    expect(store.getSnapshot()).toEqual([]);
  });

  it("does not duplicate a job that its own page is already tracking", async () => {
    // Las tres familias nuevas reusan la query key de su pagina: si la pagina ya
    // lo seguia, la rehidratacion no puede agregar una segunda entrada.
    const store = createJobQueueStore();
    store.addTrackedJob({ id: "t1", kind: "transcribe", fileName: "charla.wav", createdAt: 1 });
    store.addTrackedJob({ id: "d1", kind: "download", fileName: "Un clip", createdAt: 2 });
    store.addTrackedJob({ id: "s1", kind: "shape3d", fileName: "una traba", createdAt: 3 });

    await rehydrateJobQueue(
      store,
      fetchers(
        respuestas({
          transcribe: { jobs: [{ id: "t1", status: "running", originalFilename: "charla.wav" }] },
          download: { jobs: [{ id: "d1", status: "running", mediaTitle: "Un clip" }] },
          shape3d: { jobs: [{ id: "s1", status: "running", prompt: "una traba" }] },
        }),
      ),
    );

    expect(store.getSnapshot().map((j) => j.id)).toEqual(["s1", "d1", "t1"]);
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
    calls.fetchTranscribeJobs = vi.fn().mockRejectedValue(new Error("sin red"));
    calls.fetchDownloadJobs = vi.fn().mockRejectedValue(new Error("sin red"));
    calls.fetchShape3dJobs = vi.fn().mockRejectedValue(new Error("sin red"));

    await expect(rehydrateJobQueue(store, calls)).resolves.toBeUndefined();
    expect(store.getSnapshot()).toEqual([]);
  });
});
