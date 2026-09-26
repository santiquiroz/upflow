import { apiGet } from "../lib/api";

// Espejo de app/schemas_restore.py (LicensesResponse y compania).
export interface LicenseFile {
  name: string;
  text: string;
}

export interface LicensedModel {
  id: string;
  name: string;
  licenseSpdx: string;
  licenseUrl: string;
  copyright: string;
  attribution: string;
  dataLineage: string;
  // "yes" | "no" | "unclear" hoy; un valor nuevo del backend se muestra tal cual.
  commercialUse: string;
  sourceUrl: string;
  sourceRevision: string;
  modifications: string[];
  files: LicenseFile[];
}

export interface LicensedPack {
  pack: string;
  models: LicensedModel[];
}

export interface ThirdPartyNotice {
  title: string;
  section: string;
  fields: Record<string, string[]>;
  licenseText: string | null;
}

export interface LicensesResponse {
  packs: LicensedPack[];
  thirdParty: ThirdPartyNotice[];
}

export function fetchLicenses(): Promise<LicensesResponse> {
  return apiGet<LicensesResponse>("/licenses");
}

// Espejo de app/schemas_restore.py::LicenseGateResponse.
export interface PackLicense {
  pack: string;
  gated: boolean;
  licenseText: string | null;
}

export function fetchPackLicense(pack: string): Promise<PackLicense> {
  return apiGet<PackLicense>(`/packs/${encodeURIComponent(pack)}/license`);
}
