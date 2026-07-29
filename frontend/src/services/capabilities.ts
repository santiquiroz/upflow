import { apiGet } from "../lib/api";
import type { CapabilityTreeResponse } from "../lib/apiTypes";

export function fetchCapabilityTree(): Promise<CapabilityTreeResponse> {
  return apiGet<CapabilityTreeResponse>("/capabilities/tree");
}
