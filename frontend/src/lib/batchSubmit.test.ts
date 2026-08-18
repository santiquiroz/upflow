import { describe, expect, it, vi } from "vitest";
import { submitBatch } from "./batchSubmit";
import type { JobQueueStore } from "./jobQueueStore";

function colaFalsa() {
  const agregados: { id: string; fileName: string }[] = [];
  const store = {
    addTrackedJob: (job: { id: string; fileName: string }) => agregados.push(job),
  } as unknown as JobQueueStore;
  return { store, agregados };
}

const base = {
  kind: "audio" as const,
  fileNameOf: (p: { nombre: string }) => p.nombre,
  onPendingChange: () => {},
  onFailedChange: () => {},
};

describe("submitBatch", () => {
  it("no hace nada con una lista vacía", async () => {
    const { store } = colaFalsa();
    const uploadFirst = vi.fn();

    await submitBatch({
      ...base,
      paramsList: [],
      queue: store,
      uploadFirst,
      createRest: async () => "x",
    });

    expect(uploadFirst).not.toHaveBeenCalled();
  });

  it("manda el primero por la mutación observada y el resto a la cola", async () => {
    const { store, agregados } = colaFalsa();
    const uploadFirst = vi.fn().mockResolvedValue(undefined);

    await submitBatch({
      ...base,
      paramsList: [{ nombre: "a.wav" }, { nombre: "b.wav" }, { nombre: "c.wav" }],
      queue: store,
      uploadFirst,
      createRest: async (p) => `id-${p.nombre}`,
    });

    // El primero NO se agrega acá: lo sigue la pantalla y lo registra su hook.
    expect(uploadFirst).toHaveBeenCalledTimes(1);
    expect(agregados.map((j) => j.fileName)).toEqual(["b.wav", "c.wav"]);
    expect(agregados.map((j) => j.id)).toEqual(["id-b.wav", "id-c.wav"]);
  });

  it("espera al primero antes de arrancar el resto", async () => {
    // Sin esa espera los archivos llegan al servidor en un orden distinto del
    // que eligió el usuario, y la cola queda desordenada respecto de la pantalla.
    const { store } = colaFalsa();
    const orden: string[] = [];
    let liberar: (() => void) | null = null;
    const primeroTermina = new Promise<void>((resolve) => {
      liberar = resolve;
    });

    const promesa = submitBatch({
      ...base,
      paramsList: [{ nombre: "primero" }, { nombre: "segundo" }],
      queue: store,
      uploadFirst: async () => {
        orden.push("arranca primero");
        await primeroTermina;
        orden.push("termina primero");
      },
      createRest: async (p) => {
        orden.push(`arranca ${p.nombre}`);
        return "id";
      },
    });

    await Promise.resolve();
    expect(orden).toEqual(["arranca primero"]);
    liberar?.();
    await promesa;
    expect(orden).toEqual(["arranca primero", "termina primero", "arranca segundo"]);
  });

  it("un fallo del primero no cancela el lote", async () => {
    const { store, agregados } = colaFalsa();

    await submitBatch({
      ...base,
      paramsList: [{ nombre: "roto.wav" }, { nombre: "sano.wav" }],
      queue: store,
      uploadFirst: async () => {
        throw new Error("413");
      },
      createRest: async (p) => `id-${p.nombre}`,
    });

    expect(agregados.map((j) => j.fileName)).toEqual(["sano.wav"]);
  });

  it("cuenta los que fallaron sin cortar los que siguen", async () => {
    const { store, agregados } = colaFalsa();
    const fallidos: number[] = [];

    await submitBatch({
      ...base,
      paramsList: [{ nombre: "1" }, { nombre: "2" }, { nombre: "3" }],
      queue: store,
      uploadFirst: async () => {},
      createRest: async (p) => {
        if (p.nombre === "2") {
          throw new Error("429");
        }
        return `id-${p.nombre}`;
      },
      onFailedChange: (n) => fallidos.push(n),
    });

    expect(agregados.map((j) => j.fileName)).toEqual(["3"]);
    expect(fallidos.at(-1)).toBe(1);
  });
});
