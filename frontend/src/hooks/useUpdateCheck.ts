import { useQuery } from "@tanstack/react-query";
import type { UpdateCheck } from "../lib/apiTypes";
import { fetchUpdateCheck } from "../services/update";

const UPDATE_CHECK_QUERY_KEY = ["update-check"] as const;

// The release list barely moves; a passive, silent check is enough. One hour of
// freshness, no retries and no focus/reconnect refetch keep this off the hot path.
const UPDATE_CHECK_STALE_TIME_MS = 60 * 60 * 1000;

export function useUpdateCheck() {
  return useQuery<UpdateCheck>({
    queryKey: UPDATE_CHECK_QUERY_KEY,
    queryFn: fetchUpdateCheck,
    staleTime: UPDATE_CHECK_STALE_TIME_MS,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
}
