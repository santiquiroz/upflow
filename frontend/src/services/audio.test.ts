import { afterEach, describe, expect, it, vi } from "vitest";
import { capturedUploads as uploads, mockUploadOnce } from "../lib/uploadTestStub";
import type {
  AudioCapabilities,
  AudioJob,
  CreateJobResponse,
  VoiceCatalog,
} from "../lib/apiTypes";
import {
  cancelAudioJob,
  createAudioJob,
  fetchAudioCapabilities,
  fetchVoiceCatalog,
  getAudioJob,
} from "./audio";

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

describe("createAudioJob", () => {
  it("issues a multipart POST to /api/v1/audio/jobs with the selected modes and device", async () => {
    const payload: CreateJobResponse = {
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    };
    mockUploadOnce(payload, 202);
    const file = new File(["binary"], "voice.wav", { type: "audio/wav" });

    const result = await createAudioJob({
      file,
      denoise: "deepfilter",
      restore: "apollo",
      outputFormat: "flac",
      device: "dml:0",
    });

    expect(uploads[0].url).toBe("/api/v1/audio/jobs");
    const body = uploads[0].body;
    expect(body.get("file")).toBe(file);
    expect(body.get("denoise")).toBe("deepfilter");
    expect(body.get("restore")).toBe("apollo");
    expect(body.get("output_format")).toBe("flac");
    expect(body.get("device")).toBe("dml:0");
    expect(result).toEqual(payload);
  });

  it("omits denoise, restore and device fields when they are not selected", async () => {
    mockUploadOnce(
      { jobId: "aud-2", status: "queued", statusUrl: "/api/v1/audio/jobs/aud-2", downloadUrl: null },
      202,
    );
    const file = new File(["binary"], "voice.wav", { type: "audio/wav" });

    await createAudioJob({ file, denoise: "rnnoise", restore: null, outputFormat: "flac", device: null });

    const body = uploads[0].body;
    expect(body.get("denoise")).toBe("rnnoise");
    expect(body.has("restore")).toBe(false);
    expect(body.has("device")).toBe(false);
  });

  it("always sends the chosen output_format", async () => {
    mockUploadOnce(
      { jobId: "aud-3", status: "queued", statusUrl: "/api/v1/audio/jobs/aud-3", downloadUrl: null },
      202,
    );
    const file = new File(["binary"], "voice.wav", { type: "audio/wav" });

    await createAudioJob({ file, denoise: null, restore: null, outputFormat: "mp3", device: null });

    const body = uploads[0].body;
    expect(body.get("output_format")).toBe("mp3");
  });

  it("sends the lossy quality tier when one was chosen", async () => {
    mockUploadOnce(
      { jobId: "aud-4", status: "queued", statusUrl: "/api/v1/audio/jobs/aud-4", downloadUrl: null },
      202,
    );
    const file = new File(["binary"], "voice.flac", { type: "audio/flac" });

    await createAudioJob({
      file,
      denoise: null,
      restore: null,
      outputFormat: "mp3",
      lossyQuality: "compact",
      device: null,
    });

    expect(uploads[0].body.get("lossy_quality")).toBe("compact");
  });

  it("omits the quality tier when none was chosen so the backend default wins", async () => {
    mockUploadOnce(
      { jobId: "aud-5", status: "queued", statusUrl: "/api/v1/audio/jobs/aud-5", downloadUrl: null },
      202,
    );
    const file = new File(["binary"], "voice.flac", { type: "audio/flac" });

    await createAudioJob({ file, denoise: null, restore: null, outputFormat: "wav", device: null });

    expect(uploads[0].body.has("lossy_quality")).toBe(false);
  });
});

describe("getAudioJob", () => {
  it("issues a GET to /api/v1/audio/jobs/{id} and returns the typed payload", async () => {
    const payload: AudioJob = {
      ownerId: null,
      id: "aud-1",
      status: "running",
      originalFilename: "voice.wav",
      denoise: "deepfilter",
      restore: "apollo",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: null,
      progressPct: 30,
      stages: null,
      error: null,
      downloadUrl: null,
    };
    mockFetchOnce(payload);

    const result = await getAudioJob("aud-1");

    expect(fetch).toHaveBeenCalledWith("/api/v1/audio/jobs/aud-1", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("cancelAudioJob", () => {
  it("issues a POST to /api/v1/audio/jobs/{id}/cancel and returns the updated job", async () => {
    const payload: AudioJob = {
      ownerId: null,
      id: "aud-1",
      status: "cancelled",
      originalFilename: "voice.wav",
      denoise: "deepfilter",
      restore: null,
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      progressPct: null,
      stages: null,
      error: null,
      downloadUrl: null,
    };
    mockFetchOnce(payload);

    const result = await cancelAudioJob("aud-1");

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/audio/jobs/aud-1/cancel",
      expect.objectContaining({ method: "POST" }),
    );
    expect(result).toEqual(payload);
  });
});

describe("fetchAudioCapabilities", () => {
  it("issues a GET to /api/v1/audio/capabilities and returns the typed payload", async () => {
    const payload: AudioCapabilities = {
      denoiseModes: ["deepfilter", "rnnoise"],
      restoreAvailable: true,
      restoreModes: ["apollo"],
    };
    mockFetchOnce(payload);

    const result = await fetchAudioCapabilities();

    expect(fetch).toHaveBeenCalledWith("/api/v1/audio/capabilities", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("fetchVoiceCatalog", () => {
  it("issues a GET to /api/v1/audio/voice-catalog and returns the typed payload", async () => {
    const payload: VoiceCatalog = {
      steps: [
        {
          id: "deesser",
          labelKey: "voice.step.deesser.label",
          descriptionKey: "voice.step.deesser.description",
          kind: "filter",
          defaultEnabled: true,
        },
      ],
      deliveries: [
        {
          id: "streaming",
          labelKey: "voice.delivery.streaming.label",
          descriptionKey: "voice.delivery.streaming.description",
          lufs: -14,
          truePeakDb: -1,
        },
      ],
    };
    mockFetchOnce(payload);

    const result = await fetchVoiceCatalog();

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/audio/voice-catalog",
      expect.objectContaining({ method: "GET" }),
    );
    expect(result).toEqual(payload);
  });
});

describe("createAudioJob voice fields", () => {
  const file = new File(["binary"], "voice.wav", { type: "audio/wav" });

  function baseParams() {
    return { file, denoise: null, restore: null, outputFormat: "flac", device: null };
  }

  function sentBody(): FormData {
    return uploads[0].body;
  }

  function mockAccepted() {
    mockUploadOnce(
      { jobId: "aud-9", status: "queued", statusUrl: "/x", downloadUrl: null },
      202,
    );
  }

  it("sends the selected steps as a comma separated list", async () => {
    mockAccepted();
    await createAudioJob({
      ...baseParams(),
      voiceSteps: ["highpass", "compress", "deesser"],
    });
    expect(sentBody().get("voice_steps")).toBe("highpass,compress,deesser");
  });

  it("omits voice_steps entirely when nothing is selected", async () => {
    // Un string vacio pediria una cadena de CERO pasos; el campo ausente
    // significa "sin mejora de voz". No son lo mismo para el backend.
    mockAccepted();
    await createAudioJob({ ...baseParams(), voiceSteps: [] });
    expect(sentBody().has("voice_steps")).toBe(false);
  });

  it("omits the voice fields when the caller does not pass them", async () => {
    mockAccepted();
    await createAudioJob(baseParams());
    const body = sentBody();
    expect(body.has("voice_steps")).toBe(false);
    expect(body.has("voice_delivery")).toBe(false);
    expect(body.has("voice_presence_db")).toBe(false);
  });

  it("sends the cleanup chain as a comma separated list", async () => {
    mockAccepted();
    await createAudioJob({ ...baseParams(), cleanupSteps: ["denoise", "reverb_hq"] });
    expect(sentBody().get("cleanup_steps")).toBe("denoise,reverb_hq");
  });

  it("omits cleanup_steps entirely when nothing is selected", async () => {
    mockAccepted();
    await createAudioJob({ ...baseParams(), cleanupSteps: [] });
    expect(sentBody().has("cleanup_steps")).toBe(false);
  });

  it("sends the cleanup chain alongside master in the same job", async () => {
    // Esto es lo que la limpieza gana al dejar de ser un modo exclusivo.
    mockAccepted();
    await createAudioJob({
      ...baseParams(),
      cleanupSteps: ["denoise"],
      master: "streaming",
    });
    const body = sentBody();
    expect(body.get("cleanup_steps")).toBe("denoise");
    expect(body.get("master")).toBe("streaming");
  });

  it("sends the delivery target and the presence amount", async () => {
    mockAccepted();
    await createAudioJob({
      ...baseParams(),
      voiceSteps: ["loudness", "presence"],
      voiceDelivery: "ebu_r128",
      voicePresenceDb: 4.5,
    });
    const body = sentBody();
    expect(body.get("voice_delivery")).toBe("ebu_r128");
    expect(body.get("voice_presence_db")).toBe("4.5");
  });

  it("sends a presence amount of zero instead of dropping it", async () => {
    // 0 es un valor valido: un truthy check lo descartaria en silencio.
    mockAccepted();
    await createAudioJob({ ...baseParams(), voiceSteps: ["presence"], voicePresenceDb: 0 });
    expect(sentBody().get("voice_presence_db")).toBe("0");
  });

  it("omits the presence amount when it is null", async () => {
    mockAccepted();
    await createAudioJob({ ...baseParams(), voicePresenceDb: null });
    expect(sentBody().has("voice_presence_db")).toBe(false);
  });
});

describe("createAudioJob practice fields", () => {
  const file = new File(["binary"], "song.wav", { type: "audio/wav" });

  function separateParams() {
    return {
      file,
      denoise: null,
      restore: null,
      outputFormat: "flac",
      device: null,
      separate: true,
      separationModel: "umx_4stem",
    };
  }

  function sentBody(): FormData {
    return uploads[0].body;
  }

  function mockAccepted() {
    mockUploadOnce({ jobId: "prac-1", status: "queued", statusUrl: "/x", downloadUrl: null }, 202);
  }

  it("sends the practice stems as a comma separated list with the guide percent", async () => {
    mockAccepted();
    await createAudioJob({
      ...separateParams(),
      practiceStems: ["drums", "bass"],
      practiceGuidePercent: 20,
    });
    const body = sentBody();
    expect(body.get("practice_stems")).toBe("drums,bass");
    expect(body.get("practice_guide_percent")).toBe("20");
  });

  it("omits practice_guide_percent when it is zero so the backend default wins", async () => {
    mockAccepted();
    await createAudioJob({
      ...separateParams(),
      practiceStems: ["drums"],
      practiceGuidePercent: 0,
    });
    const body = sentBody();
    expect(body.get("practice_stems")).toBe("drums");
    expect(body.has("practice_guide_percent")).toBe(false);
  });

  it("omits both fields when no practice stem is selected", async () => {
    // Sin stems no hay minus-one que hornear: una guia suelta seria un pedido
    // sin sujeto y el campo ausente es lo que el backend espera.
    mockAccepted();
    await createAudioJob({
      ...separateParams(),
      practiceStems: [],
      practiceGuidePercent: 20,
    });
    const body = sentBody();
    expect(body.has("practice_stems")).toBe(false);
    expect(body.has("practice_guide_percent")).toBe(false);
  });

  it("omits the practice fields when the caller does not pass them", async () => {
    mockAccepted();
    await createAudioJob(separateParams());
    const body = sentBody();
    expect(body.has("practice_stems")).toBe(false);
    expect(body.has("practice_guide_percent")).toBe(false);
  });
});
