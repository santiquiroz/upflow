import { useQuery } from "@tanstack/react-query";
import type { CapabilityTreeResponse } from "../lib/apiTypes";
import { fetchCapabilityTree } from "../services/capabilities";

export function useCapabilityTree() {
  return useQuery<CapabilityTreeResponse>({
    queryKey: ["capability-tree"],
    queryFn: fetchCapabilityTree,
  });
}
