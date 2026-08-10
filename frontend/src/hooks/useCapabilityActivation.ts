import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import type { CapabilityResponse } from "../lib/apiTypes";
import { patchSetting } from "../services/settings";

export interface UseCapabilityActivationResult {
  capabilityId: string | null;
  errorMessage: string | null;
  isBusy: boolean;
  isDone: boolean;
  activate: (capability: CapabilityResponse) => void;
}

// Prender los ajustes que le faltan a UNA capacidad, desde su propia tarjeta.
// El backend solo pone en `activatableSettings` flags que aplican en caliente,
// asi que al volver el arbol re-resuelto la tarjeta queda disponible de verdad
// y no "guardado, ahora reinicia".
export function useCapabilityActivation(): UseCapabilityActivationResult {
  const queryClient = useQueryClient();
  const [capabilityId, setCapabilityId] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: async (capability: CapabilityResponse) => {
      for (const key of capability.activatableSettings) {
        await patchSetting(key, "true");
      }
    },
    onSettled: () => {
      // Tambien en el fallo: si el primero de dos ajustes entro y el segundo
      // no, la tarjeta tiene que mostrar el estado real, no el viejo.
      void queryClient.invalidateQueries({ queryKey: ["capability-tree"] });
      void queryClient.invalidateQueries({ queryKey: ["editable-settings"] });
    },
  });

  return {
    capabilityId,
    errorMessage: mutation.error instanceof Error ? mutation.error.message : null,
    isBusy: mutation.isPending,
    isDone: mutation.isSuccess,
    activate: (capability: CapabilityResponse) => {
      setCapabilityId(capability.id);
      mutation.mutate(capability);
    },
  };
}
