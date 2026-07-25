import { apiGet, apiPatchJson } from "../lib/api";
import type { EditableSettingsResponse } from "../lib/apiTypes";

export function fetchEditableSettings(): Promise<EditableSettingsResponse> {
  return apiGet<EditableSettingsResponse>("/settings");
}

export function patchSetting(key: string, value: string): Promise<{ key: string }> {
  return apiPatchJson<{ key: string }>("/settings", { key, value });
}
