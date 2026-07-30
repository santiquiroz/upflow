import { useQuery } from "@tanstack/react-query";
import { preflightUpscalerModel } from "../lib/api";
import type { UpscalerPreflightResponse } from "../lib/apiTypes";

export function useUpscalerModelPreflight(repoId: string, enabled: boolean) {
  return useQuery<UpscalerPreflightResponse>({
    queryKey: ["upscaler-model-preflight", repoId],
    queryFn: () => preflightUpscalerModel(repoId),
    enabled,
    staleTime: 5 * 60 * 1000,
  });
}
