import { ApiError, apiPostJson } from "../lib/api";
import type { InitImageResponse, InsertObjectResponse } from "../lib/apiTypes";

const API_BASE = "/api/v1";

export interface EditorCapabilities {
  tapSelect: boolean;
  people?: boolean;
  osor?: boolean;
}

export interface EditorPerson {
  id: number;
  box: [number, number, number, number];
  score: number;
  role: "subject" | "background";
  face: [number, number, number, number] | null;
  anchor: [number, number];
}

export async function fetchEditorCapabilities(): Promise<EditorCapabilities> {
  const response = await fetch(`${API_BASE}/editor/capabilities`);
  if (!response.ok) {
    throw new ApiError(response.status, "Could not load editor capabilities");
  }
  return (await response.json()) as EditorCapabilities;
}

export async function detectPeople(imageToken: string): Promise<EditorPerson[]> {
  const response = await apiPostJson<{ people: EditorPerson[] }>("/editor/people", { imageToken });
  return response.people;
}

async function failureOf(response: Response, fallback: string): Promise<ApiError> {
  let detail = fallback;
  try {
    detail = ((await response.json()) as { detail?: string }).detail ?? fallback;
  } catch {
    // cuerpo no-JSON: se queda el mensaje genérico
  }
  return new ApiError(response.status, detail);
}

async function fetchEditorSegment(body: {
  imageToken: string;
  x?: number;
  y?: number;
  box?: [number, number, number, number];
}): Promise<Blob> {
  const response = await fetch(`${API_BASE}/editor/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw await failureOf(response, "Segmentation failed");
  }
  return response.blob();
}

// Devuelve la máscara del objeto tocado como Blob PNG (blanco=objeto). Va con
// fetch crudo porque la respuesta es binaria, no JSON.
export async function segmentEditorObject(
  imageToken: string,
  x: number,
  y: number,
): Promise<Blob> {
  return fetchEditorSegment({ imageToken, x, y });
}

export async function segmentEditorBox(
  imageToken: string,
  box: [number, number, number, number],
): Promise<Blob> {
  return fetchEditorSegment({ imageToken, box });
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
  // 0..1: qué fracción del objeto re-genera la armonización, desde el borde
  // hacia adentro. 1 = el objeto entero (comportamiento clásico).
  harmonizeBlend?: number;
  modelId?: string;
  prompt?: string;
  device?: string;
}

// Pega el objeto recortado de la imagen origen sobre la destino. El composite
// vuelve inline en base64; con harmonize el jobId entra a la cola normal.
export function insertObject(params: InsertObjectParams): Promise<InsertObjectResponse> {
  return apiPostJson<InsertObjectResponse>("/editor/insert-object", params);
}

// Corrige la zona de la máscara con LaMa y devuelve la imagen nueva como staging
// propio (no pasa por la cola de trabajos).
export function healEditorImage(imageToken: string, maskToken: string): Promise<InitImageResponse> {
  return apiPostJson<InitImageResponse>("/editor/heal", { imageToken, maskToken });
}

// Baja una imagen de partida en staging. Va con fetch crudo porque la respuesta es binaria.
export async function fetchInitImage(token: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/generation/init-image/${token}`);
  if (!response.ok) {
    throw await failureOf(response, "Could not load the image");
  }
  return response.blob();
}
