import type { Shape3dJob } from "../../lib/apiTypes";
import { jobQueueStore, type JobQueueStore } from "../../lib/jobQueueStore";

// Generar una pieza son minutos de CPU: sin seguirla en la cola, irse de la
// pantalla equivalia a perderla de vista aunque siguiera corriendo. El nombre
// visible es el prompt; en el modo foto no hay prompt y queda el archivo.
export function trackShape3dJob(
  job: Shape3dJob,
  fallbackName: string,
  queue: JobQueueStore = jobQueueStore,
): void {
  queue.addTrackedJob({
    id: job.id,
    kind: "shape3d",
    fileName: job.prompt.trim() || fallbackName,
    createdAt: Date.now(),
  });
}
