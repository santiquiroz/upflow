import { describe, expect, it, vi } from "vitest";
import { createJobQueueStore, type TrackedJob } from "./jobQueueStore";

function makeJob(overrides: Partial<TrackedJob> = {}): TrackedJob {
  return { id: "job-1", kind: "image", fileName: "photo.png", createdAt: 1, ...overrides };
}

describe("createJobQueueStore", () => {
  it("starts with an empty snapshot", () => {
    const store = createJobQueueStore();

    expect(store.getSnapshot()).toEqual([]);
  });

  it("prepends newly added jobs so the newest job is first", () => {
    const store = createJobQueueStore();

    store.addTrackedJob(makeJob({ id: "job-1", createdAt: 1 }));
    store.addTrackedJob(makeJob({ id: "job-2", createdAt: 2 }));

    expect(store.getSnapshot().map((job) => job.id)).toEqual(["job-2", "job-1"]);
  });

  it("removes a tracked job by id", () => {
    const store = createJobQueueStore();
    store.addTrackedJob(makeJob({ id: "job-1" }));
    store.addTrackedJob(makeJob({ id: "job-2" }));

    store.removeTrackedJob("job-1");

    expect(store.getSnapshot().map((job) => job.id)).toEqual(["job-2"]);
  });

  it("notifies subscribers when a job is added", () => {
    const store = createJobQueueStore();
    const listener = vi.fn();
    store.subscribe(listener);

    store.addTrackedJob(makeJob());

    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("notifies subscribers when a job is removed", () => {
    const store = createJobQueueStore();
    store.addTrackedJob(makeJob({ id: "job-1" }));
    const listener = vi.fn();
    store.subscribe(listener);

    store.removeTrackedJob("job-1");

    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("stops notifying a listener once it unsubscribes", () => {
    const store = createJobQueueStore();
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    unsubscribe();

    store.addTrackedJob(makeJob());

    expect(listener).not.toHaveBeenCalled();
  });

  it("ignora un id que ya esta en la cola en vez de duplicarlo", () => {
    const store = createJobQueueStore();
    store.addTrackedJob(makeJob({ id: "job-1", fileName: "primero.png" }));

    store.addTrackedJob(makeJob({ id: "job-1", fileName: "otro-nombre.png" }));

    // Dos entradas para un solo trabajo siempre es un error. Importa desde que
    // hay trabajos que la pantalla descubre sondeando (los que encadena una
    // descarga): ahi el mismo id vuelve a llegar en cada respuesta.
    expect(store.getSnapshot()).toHaveLength(1);
    expect(store.getSnapshot()[0].fileName).toBe("primero.png");
  });

  it("no avisa a los suscriptores cuando el id repetido se descarta", () => {
    const store = createJobQueueStore();
    store.addTrackedJob(makeJob({ id: "job-1" }));
    const listener = vi.fn();
    store.subscribe(listener);

    store.addTrackedJob(makeJob({ id: "job-1" }));

    // Sin esto, un sondeo por segundo dispara un re-render por segundo que no
    // cambia nada de lo que se ve.
    expect(listener).not.toHaveBeenCalled();
  });

  it("keeps two independent store instances isolated from each other", () => {
    const storeA = createJobQueueStore();
    const storeB = createJobQueueStore();

    storeA.addTrackedJob(makeJob({ id: "job-1" }));

    expect(storeA.getSnapshot()).toHaveLength(1);
    expect(storeB.getSnapshot()).toHaveLength(0);
  });
});
