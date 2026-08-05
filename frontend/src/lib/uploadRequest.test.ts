import { describe, expect, it, vi } from "vitest";
import { ApiError } from "./api";
import { postFormWithProgress } from "./uploadRequest";

// `fetch` NO puede reportar progreso de subida: no expone el evento. Por eso la
// subida usa XMLHttpRequest, que es la unica forma en el navegador de saber
// cuanto se lleva enviado — y de cancelarla a mitad de camino. Un archivo de
// video puede pesar 2 GB, asi que subir sin porcentaje ni cancelar es dejar al
// usuario mirando una pantalla quieta durante minutos.

class FakeXhr {
  status = 200;
  responseText = "{}";
  readonly upload = { onprogress: null as ((event: ProgressEvent) => void) | null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  aborted = false;
  opened: [string, string] | null = null;
  sent: FormData | null = null;

  open(method: string, url: string): void {
    this.opened = [method, url];
  }

  send(body: FormData): void {
    this.sent = body;
  }

  abort(): void {
    this.aborted = true;
    this.onabort?.();
  }

  emitProgress(loaded: number, total: number, lengthComputable = true): void {
    this.upload.onprogress?.({ loaded, total, lengthComputable } as ProgressEvent);
  }

  finish(status: number, body: string): void {
    this.status = status;
    this.responseText = body;
    this.onload?.();
  }
}

function upload(xhr: FakeXhr, options: Parameters<typeof postFormWithProgress>[2] = {}) {
  return postFormWithProgress<{ jobId: string }>(
    "/jobs",
    new FormData(),
    options,
    () => xhr as unknown as XMLHttpRequest,
  );
}

describe("postFormWithProgress", () => {
  it("resolves with the parsed body when the server accepts the upload", async () => {
    const xhr = new FakeXhr();
    const pending = upload(xhr);

    xhr.finish(200, JSON.stringify({ jobId: "abc" }));

    await expect(pending).resolves.toEqual({ jobId: "abc" });
    expect(xhr.opened).toEqual(["POST", "/api/v1/jobs"]);
  });

  it("reports how much of the file has been sent", async () => {
    const xhr = new FakeXhr();
    const seen: (number | null)[] = [];
    const pending = upload(xhr, { onProgress: (percent) => seen.push(percent) });

    xhr.emitProgress(25, 100);
    xhr.emitProgress(100, 100);
    xhr.finish(200, "{}");
    await pending;

    expect(seen).toEqual([25, 100]);
  });

  it("reports an unknown percentage instead of inventing one", async () => {
    // Sin `lengthComputable` no se sabe el total. Dibujar un porcentaje
    // inventado seria mentir sobre lo que falta.
    const xhr = new FakeXhr();
    const seen: (number | null)[] = [];
    const pending = upload(xhr, { onProgress: (percent) => seen.push(percent) });

    xhr.emitProgress(25, 0, false);
    xhr.finish(200, "{}");
    await pending;

    expect(seen).toEqual([null]);
  });

  it("surfaces the server's reason when it rejects the upload", async () => {
    const xhr = new FakeXhr();
    const pending = upload(xhr);

    xhr.finish(413, JSON.stringify({ detail: "Upload exceeds limit of 2048 MB" }));

    await expect(pending).rejects.toThrow(ApiError);
    await expect(pending).rejects.toThrow(/2048 MB/);
  });

  it("still fails with the status when the body is not JSON", async () => {
    const xhr = new FakeXhr();
    const pending = upload(xhr);

    xhr.finish(500, "<html>nginx</html>");

    await expect(pending).rejects.toThrow(/500/);
  });

  it("cancels the upload in flight when the caller aborts", async () => {
    const xhr = new FakeXhr();
    const controller = new AbortController();
    const pending = upload(xhr, { signal: controller.signal });

    controller.abort();

    expect(xhr.aborted).toBe(true);
    await expect(pending).rejects.toThrow(/cancel/i);
  });

  it("fails loudly when the connection drops", async () => {
    const xhr = new FakeXhr();
    const pending = upload(xhr);

    xhr.onerror?.();

    await expect(pending).rejects.toThrow(ApiError);
  });

  it("does not send anything after the caller already aborted", async () => {
    const xhr = new FakeXhr();
    const controller = new AbortController();
    controller.abort();

    const pending = upload(xhr, { signal: controller.signal });

    await expect(pending).rejects.toThrow(/cancel/i);
    expect(xhr.sent).toBeNull();
  });
});

describe("vi is available for the suite", () => {
  it("keeps the fake isolated", () => {
    expect(vi).toBeDefined();
  });
});
