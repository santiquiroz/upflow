import { afterEach, describe, expect, it, vi, type Mock } from "vitest";
import { capturedUploads as uploads, mockUploadOnce } from "../lib/uploadTestStub";
import {
  createKaraokeJob,
  renderKaraokeJob,
  updateKaraokeLyrics,
  type KaraokeJob,
} from "./karaoke";

// El detector de cantantes es opt-in: un campo enviado de mas activa el
// clustering en un tema de un solo cantante, y uno omitido de mas lo apaga en
// silencio. Estos tests fijan exactamente que viaja y que no.

function mockFetchOnce(body: unknown, init: ResponseInit = { status: 200 }) {
  const response = new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

const file = new File(["binary"], "cancion.wav", { type: "audio/wav" });

function baseParams() {
  return { file, asrModelId: "asr-1" };
}

function sentBody(): FormData {
  return uploads[0].body;
}

function mockAccepted() {
  mockUploadOnce(
    { jobId: "kj-1", status: "queued", statusUrl: "/x", downloadUrl: null },
    202,
  );
}

function reviewJob(): KaraokeJob {
  return {
    id: "kj-1",
    status: "completed",
    phase: "review",
    originalFilename: "cancion.wav",
    asrModelId: "asr-1",
    separationModelId: null,
    cleanupSteps: [],
    restoreMode: null,
    language: null,
    romanize: false,
    translateTo: null,
    device: null,
    backgroundKind: "generated",
    progressPct: 100,
    error: null,
    lines: [],
    singers: [],
    instrumentalUrl: null,
    sourceHasPicture: false,
    downloadUrl: null,
    practiceMixUrl: null,
  };
}

describe("createKaraokeJob singer detection fields", () => {
  it("sends detect_singers with the chosen singer count", async () => {
    mockAccepted();
    await createKaraokeJob({ ...baseParams(), detectSingers: true, singerCount: 3 });
    const body = sentBody();
    expect(body.get("detect_singers")).toBe("true");
    expect(body.get("singer_count")).toBe("3");
  });

  it("omits singer_count when it was not chosen so the backend default wins", async () => {
    mockAccepted();
    await createKaraokeJob({ ...baseParams(), detectSingers: true });
    const body = sentBody();
    expect(body.get("detect_singers")).toBe("true");
    expect(body.has("singer_count")).toBe(false);
  });

  it("omits both fields when detection is off", async () => {
    // singer_count sin detect_singers es invalido para el backend: mandarlo
    // suelto convertiria un create sano en un 400.
    mockAccepted();
    await createKaraokeJob({ ...baseParams(), detectSingers: false, singerCount: 3 });
    const body = sentBody();
    expect(body.has("detect_singers")).toBe(false);
    expect(body.has("singer_count")).toBe(false);
  });

  it("omits both fields when the caller does not pass them", async () => {
    mockAccepted();
    await createKaraokeJob(baseParams());
    const body = sentBody();
    expect(body.has("detect_singers")).toBe(false);
    expect(body.has("singer_count")).toBe(false);
  });
});

describe("updateKaraokeLyrics singer fields", () => {
  it("sends the per-line singer reassignment inside the line edit", async () => {
    mockFetchOnce(reviewJob());
    await updateKaraokeLyrics("kj-1", [{ index: 0, singer: "s2" }]);
    const [, init] = (fetch as Mock).mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({ lines: [{ index: 0, singer: "s2" }] });
  });

  it("sends the full singer list when renames are staged", async () => {
    mockFetchOnce(reviewJob());
    await updateKaraokeLyrics(
      "kj-1",
      [],
      [
        { id: "s1", label: "Ana" },
        { id: "s2", label: "Beto" },
      ],
    );
    const [, init] = (fetch as Mock).mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      lines: [],
      singers: [
        { id: "s1", label: "Ana" },
        { id: "s2", label: "Beto" },
      ],
    });
  });

  it("omits the singers key entirely when no rename was staged", async () => {
    // Mandar singers vacio significaria "borra los cantantes"; el campo
    // ausente significa "no los toques". No son lo mismo para el backend.
    mockFetchOnce(reviewJob());
    await updateKaraokeLyrics("kj-1", [{ index: 0, text: "hola" }]);
    const [, init] = (fetch as Mock).mock.calls[0];
    expect("singers" in JSON.parse(init.body)).toBe(false);
  });
});

describe("renderKaraokeJob singer fields", () => {
  function renderParams() {
    return {
      backgroundKind: "generated" as const,
      subtitleSize: "medium",
      subtitlePosition: "bottom",
      subtitleColor: "#FFFF00",
      subtitleHighlightColor: "#FFFFFF",
    };
  }

  it("sends one singer_colors entry per singer as id:hex plus the muted singer", async () => {
    mockUploadOnce(reviewJob());
    await renderKaraokeJob("kj-1", {
      ...renderParams(),
      singerColors: { s1: "#FF5252", s2: "#40C4FF" },
      muteSinger: "s1",
    });
    const body = sentBody();
    expect(body.getAll("singer_colors")).toEqual(["s1:#FF5252", "s2:#40C4FF"]);
    expect(body.get("mute_singer")).toBe("s1");
  });

  it("omits the singer fields when the job has no singers", async () => {
    mockUploadOnce(reviewJob());
    await renderKaraokeJob("kj-1", renderParams());
    const body = sentBody();
    expect(body.has("singer_colors")).toBe(false);
    expect(body.has("mute_singer")).toBe(false);
  });

  it("omits mute_singer when nobody practices so every voice stays", async () => {
    mockUploadOnce(reviewJob());
    await renderKaraokeJob("kj-1", {
      ...renderParams(),
      singerColors: { s1: "#FF5252" },
      muteSinger: null,
    });
    const body = sentBody();
    expect(body.getAll("singer_colors")).toEqual(["s1:#FF5252"]);
    expect(body.has("mute_singer")).toBe(false);
  });
});
