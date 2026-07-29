import { apiGet, apiPost } from "../lib/api";
import type { CapabilityTreeResponse, ProvisionJob } from "../lib/apiTypes";

export function fetchCapabilityTree(): Promise<CapabilityTreeResponse> {
  return apiGet<CapabilityTreeResponse>("/capabilities/tree");
}

export function provisionCapability(capabilityId: string): Promise<ProvisionJob> {
  return apiPost<ProvisionJob>(`/capabilities/${capabilityId}/provision`);
}

export function getProvisionStatus(jobId: string): Promise<ProvisionJob> {
  return apiGet<ProvisionJob>(`/capabilities/provision/${jobId}`);
}
