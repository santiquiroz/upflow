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
    onSuccess: () => queryClient.invalidateQueries({ queryKey: EDITABLE_SETTINGS_QUERY_KEY }),
  });

  return { settingsQuery, patchMutation };
}
