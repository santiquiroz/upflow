import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { fixLever, getCapabilities, rescanCapabilities } from "../lib/api";
import type { LeverResponse } from "../lib/apiTypes";

const CAPABILITIES_QUERY_KEY = ["capabilities"] as const;

export function useCapabilities() {
  const queryClient = useQueryClient();
  const [fixingLeverId, setFixingLeverId] = useState<string | null>(null);

  const capabilitiesQuery = useQuery({ queryKey: CAPABILITIES_QUERY_KEY, queryFn: getCapabilities });

  const rescanMutation = useMutation({
    mutationFn: rescanCapabilities,
    onSuccess: (data) => queryClient.setQueryData(CAPABILITIES_QUERY_KEY, data),
  });

  const fixMutation = useMutation({
    mutationFn: (leverId: string) => fixLever(leverId),
    onMutate: (leverId: string) => setFixingLeverId(leverId),
    onSettled: () => setFixingLeverId(null),
    onSuccess: (data, leverId) => {
      queryClient.setQueryData<{ levers: LeverResponse[] } | undefined>(CAPABILITIES_QUERY_KEY, (prev) => {
        if (!prev) return prev;
        return { levers: prev.levers.map((lever) => (lever.id === leverId ? data.lever : lever)) };
      });
    },
  });

  return {
    levers: capabilitiesQuery.data?.levers ?? [],
    isLoading: capabilitiesQuery.isLoading,
    isError: capabilitiesQuery.isError,
    rescan: rescanMutation.mutate,
    isRescanning: rescanMutation.isPending,
    fix: fixMutation.mutate,
    fixingLeverId,
  };
}
