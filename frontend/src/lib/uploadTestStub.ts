import { vi } from "vitest";

// Las subidas usan XMLHttpRequest y no `fetch`, porque fetch no puede reportar
// progreso de subida. Los tests que verifican el multipart necesitan stubear
// XHR; este helper existe para no repetir el stub en cada suite.

export interface CapturedUpload {
  url: string;
  body: FormData;
}

export const capturedUploads: CapturedUpload[] = [];

export function mockUploadOnce(body: unknown, status = 200): void {
  capturedUploads.length = 0;
  class StubXhr {
    status = status;
    responseText = JSON.stringify(body);
    readonly upload = { onprogress: null as ((event: ProgressEvent) => void) | null };
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    onabort: (() => void) | null = null;
    private url = "";

    open(_method: string, url: string): void {
      this.url = url;
    }

    send(form: FormData): void {
      capturedUploads.push({ url: this.url, body: form });
      queueMicrotask(() => this.onload?.());
    }

    abort(): void {
      this.onabort?.();
    }
  }
  vi.stubGlobal("XMLHttpRequest", StubXhr);
}
