import { describe, expect, it, vi } from "vitest";
import { uploadSequentially } from "./sequentialUploads";

// De a uno y no todos juntos: en paralelo compiten por el ancho de banda, el
// porcentaje de subida deja de significar algo y la cola acotada del servidor
// responde 429 a los ultimos.
describe("uploadSequentially", () => {
  it("sends them in the order they were given", async () => {
    const enviados: string[] = [];
    await uploadSequentially(["a", "b", "c"], async (item) => {
      enviados.push(item);
    });

    expect(enviados).toEqual(["a", "b", "c"]);
  });

  it("never has two in flight at the same time", async () => {
    let enVuelo = 0;
    let maximo = 0;
    await uploadSequentially(["a", "b", "c"], async () => {
      enVuelo += 1;
      maximo = Math.max(maximo, enVuelo);
      await Promise.resolve();
      enVuelo -= 1;
    });

    expect(maximo).toBe(1);
  });

  it("keeps going after one fails and reports how many failed", async () => {
    // Que un archivo no entre no es razon para no intentar los que siguen.
    const enviados: string[] = [];
    const resultado = await uploadSequentially(["a", "b", "c"], async (item) => {
      if (item === "b") {
        throw new Error("cola llena");
      }
      enviados.push(item);
    });

    expect(enviados).toEqual(["a", "c"]);
    expect(resultado.failed).toBe(1);
  });

  it("reports what is left after each one", async () => {
    const restantes: number[] = [];
    await uploadSequentially(
      ["a", "b", "c"],
      async () => undefined,
      (quedan) => restantes.push(quedan),
    );

    expect(restantes).toEqual([2, 1, 0]);
  });

  it("an empty list does nothing", async () => {
    const upload = vi.fn();
    const resultado = await uploadSequentially([], upload);

    expect(upload).not.toHaveBeenCalled();
    expect(resultado.failed).toBe(0);
  });
});
