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
export function provisionPack(
  pack: string,
  variant?: string,
  acceptLicense = false,
): Promise<ProvisionJob> {
  return apiPost<ProvisionJob>(`/packs/${pack}/provision${provisionQuery(variant, acceptLicense)}`);
}

export function provisionQuery(variant: string | undefined, acceptLicense: boolean): string {
  const params = new URLSearchParams();
  if (variant) params.set("variant", variant);
  if (acceptLicense) params.set("acceptLicense", "true");
  const query = params.toString();
  return query ? `?${query}` : "";
}
