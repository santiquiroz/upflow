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

/**
 * Baja un paquete por su nombre, sin pasar por una capacidad.
 *
 * Casi toda pantalla sabe QUÉ le falta pero no a qué capacidad pertenece, y
 * varios paquetes existen sin figurar en el catálogo.
 */
export function provisionPack(pack: string, variant?: string): Promise<ProvisionJob> {
  const query = variant ? `?variant=${encodeURIComponent(variant)}` : "";
  return apiPost<ProvisionJob>(`/packs/${pack}/provision${query}`);
}
