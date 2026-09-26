import { ApiError, readErrorDetail } from "./api";

const API_BASE = "/api/v1";

export interface UploadOptions {
  // `null` significa "no se sabe cuanto falta". El servidor no siempre publica
  // el total, y dibujar un porcentaje inventado seria mentir.
  onProgress?: (percent: number | null) => void;
  signal?: AbortSignal;
}

type XhrFactory = () => XMLHttpRequest;

function parseBody(body: string): unknown {
  try {
    return JSON.parse(body);
  } catch {
    // No era JSON: el status es lo unico cierto que queda.
    return null;
  }
}

function errorFromBody(status: number, body: string): ApiError {
  const { message, key } = readErrorDetail(parseBody(body));
  return new ApiError(status, message ?? `Request failed with status ${status}`, key);
}

// `fetch` no expone el progreso de subida — no es una limitacion de la app sino
// de la API del navegador. XMLHttpRequest es la unica forma de saber cuanto se
// lleva enviado y de cortar la subida a mitad de camino.
export function postFormWithProgress<T>(
  path: string,
  formData: FormData,
  options: UploadOptions = {},
  createXhr: XhrFactory = () => new XMLHttpRequest(),
): Promise<T> {
  const { onProgress, signal } = options;

  return new Promise<T>((resolve, reject) => {
    const cancelled = () => reject(new ApiError(0, "Upload cancelled"));

    if (signal?.aborted) {
      cancelled();
      return;
    }

    const xhr = createXhr();
    xhr.open("POST", `${API_BASE}${path}`);

    if (onProgress) {
      xhr.upload.onprogress = (event: ProgressEvent) => {
        onProgress(event.lengthComputable ? Math.round((event.loaded / event.total) * 100) : null);
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T);
        } catch {
          reject(new ApiError(xhr.status, "The server returned a malformed response"));
        }
        return;
      }
      reject(errorFromBody(xhr.status, xhr.responseText));
    };

    xhr.onerror = () => reject(new ApiError(0, "The connection failed during the upload"));
    xhr.onabort = cancelled;

    signal?.addEventListener("abort", () => xhr.abort(), { once: true });

    xhr.send(formData);
  });
}
