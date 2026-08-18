import { uploadSequentially } from "./sequentialUploads";
import type { JobQueueStore, TrackedJobKind } from "./jobQueueStore";

/** Manda N archivos como N trabajos: el primero mirado, el resto a la cola.
 *
 * El primero pasa por la mutación normal —es el que la pantalla sigue en vivo,
 * con su barra de subida— y se ESPERA antes de arrancar los demás. Sin esa
 * espera los archivos llegan al servidor en un orden distinto del que eligió el
 * usuario, y la cola queda desordenada respecto de lo que ve.
 *
 * Los que siguen se suben de a uno (ver `uploadSequentially`) y se registran en
 * la cola global apenas el servidor devuelve su id, así aparecen en la barra
 * lateral mientras todavía están subiendo los que faltan.
 *
 * Vive acá y no en cada hook porque el mismo bloque estaba copiado en imagen y
 * audio, y cada módulo que sumaba lotes lo copiaba otra vez.
 */
export interface BatchSubmitOptions<TParams> {
  paramsList: TParams[];
  kind: TrackedJobKind;
  queue: JobQueueStore;
  /** Nombre a mostrar mientras el primero sube y todavía no hay id. */
  fileNameOf: (params: TParams) => string;
  /** El primero: la mutación que la pantalla ya observa. */
  uploadFirst: (params: TParams) => Promise<unknown>;
  /** Los demás: crear el trabajo y devolver su id para la cola. */
  createRest: (params: TParams) => Promise<string>;
  onPendingChange: (pending: number) => void;
  onFailedChange: (failed: number) => void;
  onFirstStarted?: (fileName: string) => void;
}

export async function submitBatch<TParams>({
  paramsList,
  kind,
  queue,
  fileNameOf,
  uploadFirst,
  createRest,
  onPendingChange,
  onFailedChange,
  onFirstStarted,
}: BatchSubmitOptions<TParams>): Promise<void> {
  if (paramsList.length === 0) {
    return;
  }
  const [primero, ...resto] = paramsList;
  onPendingChange(resto.length);
  onFailedChange(0);
  onFirstStarted?.(fileNameOf(primero));

  // Un fallo del primero no cancela el lote: que un archivo no entre no es
  // razón para no intentar los que siguen.
  await uploadFirst(primero).catch(() => undefined);

  const { failed } = await uploadSequentially(
    resto,
    async (params) => {
      const id = await createRest(params);
      queue.addTrackedJob({
        id,
        kind,
        fileName: fileNameOf(params),
        createdAt: Date.now(),
      });
    },
    onPendingChange,
  );
  onFailedChange(failed);
}
