import type { CapabilityTreeResponse } from "../lib/apiTypes";
import { useCapabilityTree } from "../hooks/useCapabilityTree";

export const RESTORE_CAPABILITY_ID = "image.restore";

// El backend decide: con RESTORE_PHOTO_ENABLED apagado la capacidad cae al mapa de ruta.
// Mientras el arbol no llega la pestaña queda oculta: mostrarla y sacarla seria peor.
export function isRestoreReleased(tree: CapabilityTreeResponse | undefined): boolean {
  return (
    tree?.domains.some((domain) =>
      domain.capabilities.some((capability) => capability.id === RESTORE_CAPABILITY_ID),
    ) ?? false
  );
}

export function useRestoreReleased(): boolean {
  return isRestoreReleased(useCapabilityTree().data);
}
