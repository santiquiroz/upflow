export interface SequentialUploadResult {
  failed: number;
}

/** Sube los items DE A UNO, sin cortar cuando alguno falla.
 *
 * De a uno y no todos juntos: en paralelo compiten por el ancho de banda, el
 * porcentaje de subida deja de significar algo y la cola acotada del servidor
 * responde 429 a los ultimos. Y no se corta ante un fallo: que un archivo no
 * entre no es razon para no intentar los que siguen.
 */
export async function uploadSequentially<T>(
  items: T[],
  upload: (item: T) => Promise<unknown>,
  onRemaining?: (remaining: number) => void,
): Promise<SequentialUploadResult> {
  let failed = 0;
  for (let indice = 0; indice < items.length; indice += 1) {
    try {
      await upload(items[indice]);
    } catch {
      failed += 1;
    }
    onRemaining?.(items.length - indice - 1);
  }
  return { failed };
}
