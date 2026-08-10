import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchEditableSettings, patchSetting } from "../services/settings";

const EDITABLE_SETTINGS_QUERY_KEY = ["editable-settings"] as const;

export function useEditableSettings() {
  const queryClient = useQueryClient();
  const settingsQuery = useQuery({
    queryKey: EDITABLE_SETTINGS_QUERY_KEY,
    queryFn: fetchEditableSettings,
  });
  const patchMutation = useMutation({
    mutationFn: ({ key, value }: { key: string; value: string }) => patchSetting(key, value),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: EDITABLE_SETTINGS_QUERY_KEY });
      // Un ajuste puede ser el unico requisito que le falta a una capacidad
      // (ENABLE_AUDIOSR con el pack ya en disco): sin re-resolver el arbol, la
      // tarjeta sigue diciendo que falta configurarla.
      void queryClient.invalidateQueries({ queryKey: ["capability-tree"] });
    },
  });

  return { settingsQuery, patchMutation };
}
