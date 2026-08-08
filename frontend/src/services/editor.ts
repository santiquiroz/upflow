import { ApiError, apiPostJson } from "../lib/api";
import type { InsertObjectResponse } from "../lib/apiTypes";

const API_BASE = "/api/v1";

export interface EditorCapabilities {
  tapSelect: boolean;
}

export async function fetchEditorCapabilities(): Promise<EditorCapabilities> {
  const response = await fetch(`${API_BASE}/editor/capabilities`);
  if (!response.ok) {
    throw new ApiError(response.status, "Could not load editor capabilities");
  }
  return (await response.json()) as EditorCapabilities;
}

// Devuelve la máscara del objeto tocado como Blob PNG (blanco=objeto). Va con
// fetch crudo porque la respuesta es binaria, no JSON.
export async function segmentEditorObject(
  imageToken: string,
  x: number,
  y: number,
): Promise<Blob> {
  const response = await fetch(`${API_BASE}/editor/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageToken, x, y }),
  });
  if (!response.ok) {
    let detail = "Segmentation failed";
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // cuerpo no-JSON: se queda el mensaje genérico
    }
    throw new ApiError(response.status, detail);
  }
  return response.blob();
}

export interface InsertObjectParams {
  targetToken: string;
  sourceToken: string;
  sourceMaskToken: string;
  // Modo reemplazo: máscara de un objeto del destino. Con este token el backend
  // ignora x/y/width/height (posición y tamaño salen del bbox de la máscara).
  targetMaskToken?: string;
  x: number;
  y: number;
  width: number;
  height: number;
  featherPx?: number;
  matchColor?: boolean;
  harmonize?: boolean;
  modelId?: string;
  prompt?: string;
  device?: string;
}

// Pega el objeto recortado de la imagen origen sobre la destino. El composite
// vuelve inline en base64; con harmonize el jobId entra a la cola normal.
export function insertObject(params: InsertObjectParams): Promise<InsertObjectResponse> {
  return apiPostJson<InsertObjectResponse>("/editor/insert-object", params);
}
