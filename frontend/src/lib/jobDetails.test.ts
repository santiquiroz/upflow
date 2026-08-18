import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import type {
  AudioJob,
  DownloadJob,
  GenerationJob,
  JobResponse,
  Shape3dJob,
  TranscribeJob,
  VideoJobResponse,
} from "./apiTypes";
import { buildJobDetailSections, type DetailContext, type DetailItem } from "./jobDetails";

const NOW = Date.parse("2026-01-01T00:05:00Z");

function translate(key: string, params: Record<string, string> = {}): string {
  const template = en[key as keyof typeof en];
  if (template === undefined) {
    return key;
  }
  return Object.entries(params).reduce(
    (text, [name, value]) => text.replace(`{{${name}}}`, value),
    template as string,
  );
}

function context(overrides: Partial<DetailContext> = {}): DetailContext {
  return {
    t: translate,
    deviceLabel: (id) => `Radeon (${id})`,
    defaultDeviceId: "dml:0",
    labelFor: (_catalog, id) => id,
    nowMs: NOW,
    ...overrides,
  };
}

function labels(items: DetailItem[]): string[] {
  return items.map((item) => item.labelKey);
}

function valueOf(items: DetailItem[], labelKey: string): string | undefined {
  return items.find((item) => item.labelKey === labelKey)?.value;
}

const IMAGE_JOB: JobResponse = {
  jobId: "img-1",
  status: "completed",
  originalFilename: "foto.png",
  modelName: "realesrgan-x4plus",
  scale: 4,
  outputFormat: "png",
  modelId: null,
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:01:00Z",
  error: null,
  ownerId: null,
  metadata: {},
  progressPct: 100,
  downloadUrl: null,
};

const VIDEO_JOB: VideoJobResponse = {
  jobId: "vid-1",
  status: "running",
  originalFilename: "clip.mp4",
  modelName: "realesr-animevideov3-x2",
  scale: 2,
  outputContainer: "mkv",
  videoCodec: "libx265",
  videoPreset: "medium",
  crf: 20,
  keepAudio: true,
  fpsMultiplier: 1,
  targetFps: null,
  audioEnhance: null,
  audioRestore: null,
  interpEngine: "rife",
  modelId: null,
  device: "dml:0",
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: null,
  error: null,
  ownerId: null,
  metadata: {},
  progressPct: 30,
  downloadUrl: null,
};

const AUDIO_JOB: AudioJob = {
  id: "aud-1",
  status: "completed",
  originalFilename: "cancion.mp3",
  denoise: null,
  restore: "audiosr",
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:04:00Z",
  progressPct: 100,
  stages: null,
  error: null,
  ownerId: null,
  downloadUrl: null,
};

const GENERATION_JOB: GenerationJob = {
  id: "gen-1",
  status: "completed",
  prompt: "a red apple",
  negativePrompt: null,
  modelId: "sdxl",
  steps: 30,
  guidance: 7,
  width: 1024,
  height: 704,
  seed: 123456,
  seedWasRandom: true,
  device: "dml:0",
  autoUpscale: false,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:01:00Z",
  progressPct: 100,
  stages: null,
  error: null,
  ownerId: null,
  downloadUrl: null,
};

const TRANSCRIBE_JOB: TranscribeJob = {
  id: "tr-1",
  status: "completed",
  originalFilename: "charla.mp4",
  modelId: "whisper-small",
  language: null,
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:02:00Z",
  progressPct: 100,
  text: "hola mundo",
  error: null,
  ownerId: null,
  downloadUrl: null,
};

const DOWNLOAD_JOB: DownloadJob = {
  id: "dl-1",
  status: "running",
  url: "https://example.com/watch?v=1",
  maxHeight: 1080,
  audioOnly: false,
  audioFormat: "mp3",
  audioBitrateKbps: null,
  videoContainer: "mp4",
  mediaTitle: "Un video",
  mediaUploader: "Alguien",
  extractor: "youtube",
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: null,
  progressPct: 40,
  downloadedBytes: 1024 * 1024,
  totalBytes: 4 * 1024 * 1024,
  outputFiles: [],
  outputDirectory: "",
  error: null,
  ownerId: null,
  thenSeparate: false,
  followupJobIds: [],
  followupError: null,
};

const SHAPE3D_JOB: Shape3dJob = {
  id: "3d-1",
  status: "completed",
  prompt: "una maceta",
  printer: "ender-3",
  source: "mesh",
  code: null,
  retries: 0,
  targetMm: 80,
  targetMmSource: "estimate",
  targetMmReference: "una taza",
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:03:00Z",
  canPrint: true,
  sizeMm: [80, 40.25, 40],
  triangleCount: 4200,
  blockers: [],
  advice: [],
  error: null,
  downloadUrl: null,
};

describe("buildJobDetailSections", () => {
  it("returns empty groups when the job has not been fetched yet", () => {
    const sections = buildJobDetailSections(undefined, context());

    expect(sections).toEqual({ parameters: [], timing: [], result: [] });
  });

  it("labels every family with its own kind", () => {
    const kinds: [unknown, string][] = [
      [IMAGE_JOB, "Image"],
      [VIDEO_JOB, "Video"],
      [AUDIO_JOB, "Audio"],
      [GENERATION_JOB, "Generation"],
      [TRANSCRIBE_JOB, "Transcription"],
      [DOWNLOAD_JOB, "Download"],
      [SHAPE3D_JOB, "3D model"],
    ];

    for (const [job, expected] of kinds) {
      const sections = buildJobDetailSections(job as never, context());
      expect(valueOf(sections.parameters, "job.detail.field.type")).toBe(expected);
    }
  });

  it("never emits a label that is missing from the catalog", () => {
    const jobs = [IMAGE_JOB, VIDEO_JOB, AUDIO_JOB, GENERATION_JOB, TRANSCRIBE_JOB, DOWNLOAD_JOB, SHAPE3D_JOB];

    for (const job of jobs) {
      const sections = buildJobDetailSections(job as never, context());
      const allLabels = [...labels(sections.parameters), ...labels(sections.timing), ...labels(sections.result)];
      const missing = allLabels.filter((key) => !(key in en));
      expect(missing, `familia ${valueOf(sections.parameters, "job.detail.field.type")}`).toEqual([]);
    }
  });

  describe("device", () => {
    it("names the pinned device", () => {
      const sections = buildJobDetailSections(VIDEO_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.device")).toBe("Radeon (dml:0)");
    });

    it("shows the effective device when the job did not pin one", () => {
      const sections = buildJobDetailSections(AUDIO_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.device")).toBe(
        "Radeon (dml:0) (by default)",
      );
    });

    it("omits the row when there is no device to name at all", () => {
      const sections = buildJobDetailSections(AUDIO_JOB, context({ defaultDeviceId: null }));

      expect(labels(sections.parameters)).not.toContain("job.detail.field.device");
    });
  });

  describe("timing", () => {
    it("reports how long a finished job took", () => {
      const sections = buildJobDetailSections(IMAGE_JOB, context());

      expect(valueOf(sections.timing, "job.detail.field.duration")).toBe("1m 0s");
      expect(labels(sections.timing)).not.toContain("job.detail.field.elapsed");
    });

    it("reports how long a running job has been going instead", () => {
      const sections = buildJobDetailSections(VIDEO_JOB, context());

      expect(valueOf(sections.timing, "job.detail.field.elapsed")).toBe("5m 0s");
      expect(labels(sections.timing)).not.toContain("job.detail.field.duration");
    });

    it("omits both when the job has not started", () => {
      const sections = buildJobDetailSections(
        { ...VIDEO_JOB, status: "queued", startedAt: null },
        context(),
      );

      expect(labels(sections.timing)).toEqual(["job.detail.field.createdAt"]);
    });
  });

  describe("image", () => {
    it("shows model, scale and format", () => {
      const sections = buildJobDetailSections(IMAGE_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.model")).toBe("realesrgan-x4plus");
      expect(valueOf(sections.parameters, "job.detail.field.scale")).toBe("4x");
      expect(valueOf(sections.parameters, "job.detail.field.format")).toBe("PNG");
    });

    it("hides video-only fields", () => {
      const sections = buildJobDetailSections(IMAGE_JOB, context());

      expect(labels(sections.parameters)).not.toContain("job.detail.field.container");
      expect(labels(sections.parameters)).not.toContain("job.detail.field.fps");
    });

    it("reports the runtime that actually ran the upscale", () => {
      const sections = buildJobDetailSections(
        { ...IMAGE_JOB, metadata: { upscaleBackend: "onnx", upscalePrecision: "fp16", upscaleTiled: true } },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.upscaleRuntime")).toBe("onnx fp16 · tiled");
    });
  });

  describe("video", () => {
    it("shows the encoding parameters", () => {
      const sections = buildJobDetailSections(VIDEO_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.container")).toBe("MKV");
      expect(valueOf(sections.parameters, "job.detail.field.videoCodec")).toBe("libx265");
      expect(valueOf(sections.parameters, "job.detail.field.crf")).toBe("20");
    });

    it("hides the interpolation engine when nothing is interpolated", () => {
      const sections = buildJobDetailSections(VIDEO_JOB, context());

      expect(labels(sections.parameters)).not.toContain("job.detail.field.interpEngine");
    });

    it("shows the interpolation engine once a multiplier is requested", () => {
      const sections = buildJobDetailSections({ ...VIDEO_JOB, fpsMultiplier: 2 }, context());

      expect(valueOf(sections.parameters, "job.detail.field.interpEngine")).toBe("rife");
      expect(valueOf(sections.parameters, "job.detail.field.fps")).toBe("2x");
    });

    it("explains why the requested container changed", () => {
      const sections = buildJobDetailSections(
        { ...VIDEO_JOB, metadata: { containerUpgradedReason: "subtitles need mkv" } },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.containerUpgraded")).toBe(
        "subtitles need mkv",
      );
    });

    it("explains why the slower pipeline ran", () => {
      const sections = buildJobDetailSections(
        { ...VIDEO_JOB, metadata: { streamPipelineFallback: "ffmpeg pipe closed" } },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.slowerPipeline")).toBe("ffmpeg pipe closed");
    });

    it("says the audio was dropped when the job did not keep it", () => {
      const sections = buildJobDetailSections({ ...VIDEO_JOB, keepAudio: false }, context());

      expect(valueOf(sections.parameters, "job.detail.field.audio")).toBe("Dropped");
      expect(labels(sections.parameters)).not.toContain("job.detail.field.audioTracks");
    });

    it("lists the selected audio tracks when the job picked some", () => {
      const sections = buildJobDetailSections({ ...VIDEO_JOB, audioTrackIndices: [1, 3] }, context());

      expect(valueOf(sections.parameters, "job.detail.field.audioTracks")).toBe("1, 3");
    });
  });

  describe("audio", () => {
    it("shows the finishing preset, format and restore engine", () => {
      const sections = buildJobDetailSections(
        { ...AUDIO_JOB, master: "broadcast", outputFormat: "flac" },
        context({ labelFor: (_catalog, id) => (id === "broadcast" ? "Broadcast" : id) }),
      );

      expect(valueOf(sections.parameters, "job.detail.field.mastering")).toBe("Broadcast");
      expect(valueOf(sections.parameters, "job.detail.field.format")).toBe("FLAC");
      expect(valueOf(sections.parameters, "job.detail.field.restore")).toBe("AudioSR");
    });

    it("keeps the cleanup chain in execution order", () => {
      const sections = buildJobDetailSections(
        { ...AUDIO_JOB, cleanupSteps: ["denoise", "deecho", "dereverb"] },
        context(),
      );

      expect(valueOf(sections.parameters, "job.detail.field.cleanupChain")).toBe(
        "denoise → deecho → dereverb",
      );
    });

    // Una conversion promete conservar tasa y profundidad. El detalle es donde
    // eso se COMPRUEBA: sin estas filas, "sin tocar nada" es solo una promesa.
    it("shows what the conversion actually produced", () => {
      const sections = buildJobDetailSections(
        {
          ...AUDIO_JOB,
          outputFormat: "mp3",
          metadata: {
            conversionSourceFormat: "FLAC",
            conversionTargetFormat: "MP3",
            conversionSampleRate: 44100,
            conversionBitrate: "320k",
          },
        },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.conversion")).toBe("FLAC → MP3");
      expect(valueOf(sections.result, "job.detail.field.sampleRate")).toBe("44.1 kHz");
      expect(valueOf(sections.result, "job.detail.field.bitrate")).toBe("320k");
    });

    it("shows the resulting bit depth for a lossless conversion", () => {
      const sections = buildJobDetailSections(
        {
          ...AUDIO_JOB,
          outputFormat: "wav",
          metadata: {
            conversionSourceFormat: "FLAC",
            conversionTargetFormat: "WAV",
            conversionSampleRate: 48000,
            conversionBitDepth: 24,
          },
        },
        context({ t: (key, params) => (params ? `${key}:${params.bits}` : key) }),
      );

      expect(valueOf(sections.result, "job.detail.field.sampleRate")).toBe("48 kHz");
      expect(valueOf(sections.result, "job.detail.field.bitDepth")).toBe(
        "job.detail.bitDepth.value:24",
      );
    });

    it("surfaces a forced resample instead of leaving it silent", () => {
      const sections = buildJobDetailSections(
        {
          ...AUDIO_JOB,
          outputFormat: "mp3",
          metadata: {
            conversionSourceFormat: "FLAC",
            conversionTargetFormat: "MP3",
            conversionSampleRate: 48000,
            conversionResampled: "MP3 no admite 96000 Hz: la salida quedo a 48000 Hz.",
          },
        },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.conversionResampled")).toContain("96000");
    });

    it("does not add conversion rows to a job that only processed audio", () => {
      const sections = buildJobDetailSections({ ...AUDIO_JOB, denoise: "rnnoise" }, context());

      expect(valueOf(sections.result, "job.detail.field.conversion")).toBeUndefined();
      expect(valueOf(sections.result, "job.detail.field.sampleRate")).toBeUndefined();
    });

    it("names the quality tier only when the format is lossy", () => {
      const lossy = buildJobDetailSections(
        { ...AUDIO_JOB, outputFormat: "mp3", lossyQuality: "compact" },
        context({ t: (key) => key }),
      );
      const lossless = buildJobDetailSections(
        { ...AUDIO_JOB, outputFormat: "flac", lossyQuality: "compact" },
        context({ t: (key) => key }),
      );

      expect(valueOf(lossy.parameters, "job.detail.field.lossyQuality")).toBe(
        "audio.quality.compact",
      );
      expect(valueOf(lossless.parameters, "job.detail.field.lossyQuality")).toBeUndefined();
    });

    it("shows the voice chain and its delivery target", () => {
      const sections = buildJobDetailSections(
        { ...AUDIO_JOB, voiceSteps: ["deesser", "presence"], voiceDelivery: "podcast" },
        context(),
      );

      expect(valueOf(sections.parameters, "job.detail.field.voiceSteps")).toBe("deesser → presence");
      expect(valueOf(sections.parameters, "job.detail.field.voiceDelivery")).toBe("podcast");
    });

    it("shows how far the loudness actually moved", () => {
      const sections = buildJobDetailSections(
        { ...AUDIO_JOB, metadata: { loudnessBefore: -21.53, loudnessTarget: -16 } },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.loudness")).toBe("-21.5 → -16.0 LUFS");
    });

    it("surfaces the decisions the pipeline took on its own", () => {
      const sections = buildJobDetailSections(
        {
          ...AUDIO_JOB,
          metadata: {
            masteringSkipped: "no se pudo medir la sonoridad",
            voiceLoudnessSkipped: "mastering activo",
          },
        },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.masteringSkipped")).toBe(
        "no se pudo medir la sonoridad",
      );
      expect(valueOf(sections.result, "job.detail.field.voiceLoudnessSkipped")).toBe(
        "mastering activo",
      );
    });

    it("hides the whole separation block on a normal cleanup job", () => {
      const sections = buildJobDetailSections(AUDIO_JOB, context());

      expect(labels(sections.parameters)).not.toContain("job.detail.field.separationModel");
      expect(labels(sections.parameters)).not.toContain("job.detail.field.stems");
      expect(labels(sections.parameters)).not.toContain("job.detail.field.cleanupChain");
    });

    it("names the separation model and its stems in karaoke mode", () => {
      const sections = buildJobDetailSections(
        {
          ...AUDIO_JOB,
          separate: true,
          separationModel: "mel_band_roformer_kim",
          stems: [
            { id: "instrumental", labelKey: "audio.stem.instrumental", url: "/a" },
            { id: "vocals", labelKey: "audio.stem.vocals", url: "/b" },
          ],
        },
        context({ labelFor: (_catalog, id) => (id === "mel_band_roformer_kim" ? "Kim Roformer" : id) }),
      );

      expect(valueOf(sections.parameters, "job.detail.field.separationModel")).toBe("Kim Roformer");
      expect(valueOf(sections.parameters, "job.detail.field.stems")).toBeDefined();
    });
  });

  describe("generation", () => {
    it("marks a random seed as random", () => {
      const sections = buildJobDetailSections(GENERATION_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.seed")).toBe("123456 (random)");
    });

    it("leaves a chosen seed unmarked", () => {
      const sections = buildJobDetailSections({ ...GENERATION_JOB, seedWasRandom: false }, context());

      expect(valueOf(sections.parameters, "job.detail.field.seed")).toBe("123456");
    });

    it("shows precision, execution provider and speed class when known", () => {
      const sections = buildJobDetailSections(
        { ...GENERATION_JOB, precision: "fp16", executionProvider: "DmlExecutionProvider", speedClass: "turbo" },
        context(),
      );

      expect(valueOf(sections.parameters, "job.detail.field.precision")).toBe("fp16");
      expect(valueOf(sections.parameters, "job.detail.field.executionProvider")).toBe(
        "DmlExecutionProvider",
      );
      expect(valueOf(sections.parameters, "job.detail.field.speedClass")).toBe("turbo");
    });

    it("reports the pace and the upscale failure", () => {
      const sections = buildJobDetailSections(
        { ...GENERATION_JOB, upscaleError: "out of memory" },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.pace")).toBe("~2.0 s/step");
      expect(valueOf(sections.result, "job.detail.field.upscaleError")).toBe("out of memory");
    });

    it("omits strength when the job was not image to image", () => {
      const sections = buildJobDetailSections(GENERATION_JOB, context());

      expect(labels(sections.parameters)).not.toContain("job.detail.field.strength");
    });
  });

  describe("transcribe", () => {
    it("says the language was detected when none was pinned", () => {
      const sections = buildJobDetailSections(TRANSCRIBE_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.language")).toBe(
        "Detected automatically",
      );
    });

    it("shows the output mode and the dubbing language", () => {
      const sections = buildJobDetailSections(
        { ...TRANSCRIBE_JOB, outputMode: "dubbed_video", targetLanguage: "en" },
        context(),
      );

      expect(valueOf(sections.parameters, "job.detail.field.outputMode")).toBe("Dubbed video");
      expect(valueOf(sections.parameters, "job.detail.field.targetLanguage")).toBe("en");
    });

    it("hides the dubbing language on a plain transcription", () => {
      const sections = buildJobDetailSections({ ...TRANSCRIBE_JOB, outputMode: "text" }, context());

      expect(labels(sections.parameters)).not.toContain("job.detail.field.targetLanguage");
    });

    it("warns about dub lines that did not fit", () => {
      const sections = buildJobDetailSections(
        { ...TRANSCRIBE_JOB, dubOverflowSegments: 3 },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.dubOverflow")).toBe("3");
    });
  });

  describe("download", () => {
    it("shows where the media came from", () => {
      const sections = buildJobDetailSections(DOWNLOAD_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.title")).toBe("Un video");
      expect(valueOf(sections.parameters, "job.detail.field.uploader")).toBe("Alguien");
      expect(valueOf(sections.parameters, "job.detail.field.extractor")).toBe("youtube");
    });

    it("compares downloaded bytes against the total", () => {
      const sections = buildJobDetailSections(DOWNLOAD_JOB, context());

      expect(valueOf(sections.result, "job.detail.field.transferred")).toBe("1.0 MB / 4.0 MB");
    });

    it("swaps the video options for the audio ones in audio-only mode", () => {
      const sections = buildJobDetailSections(
        { ...DOWNLOAD_JOB, audioOnly: true, audioBitrateKbps: 320 },
        context(),
      );

      expect(valueOf(sections.parameters, "job.detail.field.audioOnly")).toBe("MP3");
      expect(valueOf(sections.parameters, "job.detail.field.audioBitrate")).toBe("320 kbps");
      expect(labels(sections.parameters)).not.toContain("job.detail.field.maxHeight");
    });

    it("lists the produced files once there are any", () => {
      const sections = buildJobDetailSections(
        { ...DOWNLOAD_JOB, outputFiles: ["a.mp4", "b.mp4"], outputDirectory: "C:/uploads" },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.outputFiles")).toBe("a.mp4, b.mp4");
      expect(valueOf(sections.result, "job.detail.field.outputDirectory")).toBe("C:/uploads");
    });
  });

  describe("3D", () => {
    it("says where the measurement came from, not just the number", () => {
      const sections = buildJobDetailSections(SHAPE3D_JOB, context());

      expect(valueOf(sections.parameters, "job.detail.field.targetSize")).toBe(
        "80 mm · the model suggested it from una taza and you confirmed it",
      );
    });

    it("marks a program-chosen measurement as such", () => {
      const sections = buildJobDetailSections(
        { ...SHAPE3D_JOB, targetMmSource: "default", targetMmReference: null },
        context(),
      );

      expect(valueOf(sections.parameters, "job.detail.field.targetSize")).toBe(
        "80 mm · program default",
      );
    });

    it("reports the print verdict and the measured size", () => {
      const sections = buildJobDetailSections(SHAPE3D_JOB, context());

      expect(valueOf(sections.result, "job.detail.field.verdict")).toBe("Ready to print");
      expect(valueOf(sections.result, "job.detail.field.sizeMm")).toBe("80.0 × 40.3 × 40.0");
    });

    it("lists the blockers when the piece cannot be printed", () => {
      const sections = buildJobDetailSections(
        { ...SHAPE3D_JOB, canPrint: false, blockers: ["no cierra"], advice: ["girala"] },
        context(),
      );

      expect(valueOf(sections.result, "job.detail.field.verdict")).toBe("Cannot be printed as is");
      expect(valueOf(sections.result, "job.detail.field.blockers")).toBe("no cierra");
      expect(valueOf(sections.result, "job.detail.field.advice")).toBe("girala");
    });

    it("shows the CAD code only on the CAD lane", () => {
      const meshSections = buildJobDetailSections(SHAPE3D_JOB, context());
      const cadSections = buildJobDetailSections(
        { ...SHAPE3D_JOB, source: "cad", code: "cube([10,10,10]);" },
        context(),
      );

      expect(labels(meshSections.result)).not.toContain("job.detail.field.code");
      expect(valueOf(cadSections.result, "job.detail.field.code")).toBe("cube([10,10,10]);");
      expect(valueOf(cadSections.parameters, "job.detail.field.source")).toBe(
        "CAD code with dimensions",
      );
    });
  });
});
