import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type {
  DevicesResponse,
  HfModelSearchResultResponse,
  ModelResponse,
  TranscribeJob,
} from "../../lib/apiTypes";
import * as transcribeService from "../../services/transcribe";
import { TranscribePanel } from "./TranscribePanel";

vi.mock("../../services/transcribe", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../../services/transcribe")>();
  return {
    ...actual,
    createTranscribeJob: vi.fn(),
    getTranscribeJob: vi.fn(),
    cancelTranscribeJob: vi.fn(),
    searchAsrModels: vi.fn(),
    installAsrModel: vi.fn(),
    getAsrInstallStatus: vi.fn(),
    fetchInstalledAsrModels: vi.fn(),
    fetchTranscribeDevices: vi.fn(),
    fetchTranslationPairs: vi.fn(),
  };
});

const MODEL_A: ModelResponse = {
  id: "asr-a",
  name: "Whisper Small ONNX",
  kind: "asr-onnx",
  source: "hf",
  scale: null,
  arch: "whisper",
  sizeBytes: 100,
  status: "ready",
  error: null,
};

const MODEL_B: ModelResponse = {
  ...MODEL_A,
  id: "asr-b",
  name: "Whisper Large ONNX",
};

const DEVICES: DevicesResponse = {
  devices: [
    { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
    {
      id: "dml:0",
      kind: "gpu",
      name: "AMD Radeon",
      backend: "directml",
    },
  ],
  defaultDeviceId: "dml:0",
};

const SEARCH_RESULT: HfModelSearchResultResponse = {
  id: "owner/whisper-onnx",
  author: "owner",
  pipelineTag: "automatic-speech-recognition",
  downloads: 1200,
  likes: 35,
  tags: ["automatic-speech-recognition"],
  compat: "ready_onnx",
  compatReasonKey: "compat.asr.readyOnnx",
  compatReasonParams: {},
  availablePrecisions: [],
};

function job(overrides: Partial<TranscribeJob> = {}): TranscribeJob {
  return {
    id: "tr-1",
    status: "completed",
    originalFilename: "interview.wav",
    modelId: "asr-a",
    language: null,
    device: "cpu",
    createdAt: "2026-07-29T00:00:00Z",
    startedAt: "2026-07-29T00:00:01Z",
    finishedAt: "2026-07-29T00:00:20Z",
    progressPct: 100,
    text: "Completed transcription",
    error: null,
    ownerId: null,
    downloadUrl: "/api/v1/transcribe/jobs/tr-1/download",
    ...overrides,
  };
}

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <LocaleProvider initialLocale="en">{children}</LocaleProvider>
      </QueryClientProvider>
    );
  }
  return render(
    <TranscribePanel
      pollIntervalMs={10}
      installPollIntervalMs={10}
      searchDebounceMs={0}
    />,
    { wrapper: Wrapper },
  );
}

async function selectAudioFile(): Promise<void> {
  const input = await screen.findByLabelText(
    en["transcribe.file.inputLabel"],
  );
  fireEvent.change(input, {
    target: {
      files: [
        new File(["audio"], "interview.wav", { type: "audio/wav" }),
      ],
    },
  });
}

async function selectVideoFile(): Promise<void> {
  const input = await screen.findByLabelText(
    en["transcribe.file.inputLabel"],
  );
  fireEvent.change(input, {
    target: {
      files: [new File(["video"], "charla.mp4", { type: "video/mp4" })],
    },
  });
}

async function submitAudio(): Promise<void> {
  await selectAudioFile();
  const button = screen.getByRole("button", {
    name: en["transcribe.submit"],
  });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  await waitFor(() =>
    expect(transcribeService.createTranscribeJob).toHaveBeenCalled(),
  );
}

beforeEach(() => {
  vi.mocked(transcribeService.fetchInstalledAsrModels).mockResolvedValue([
    MODEL_A,
  ]);
  vi.mocked(transcribeService.fetchTranscribeDevices).mockResolvedValue(
    DEVICES,
  );
  vi.mocked(transcribeService.searchAsrModels).mockResolvedValue({
    results: [],
  });
  vi.mocked(transcribeService.createTranscribeJob).mockResolvedValue({
    jobId: "tr-1",
    status: "queued",
    statusUrl: "/api/v1/transcribe/jobs/tr-1",
    downloadUrl: null,
  });
  vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue(job());
  vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({ pairs: [], installable: [] });
  vi.mocked(transcribeService.cancelTranscribeJob).mockResolvedValue(
    job({ status: "cancelled", text: null, downloadUrl: null }),
  );
  vi.mocked(transcribeService.installAsrModel).mockResolvedValue({
    installId: "install-1",
    statusUrl: "/api/v1/asr/models/install/install-1",
  });
  vi.mocked(transcribeService.getAsrInstallStatus).mockResolvedValue({
    installId: "install-1",
    repoId: SEARCH_RESULT.id,
    status: "installed",
    progressPct: 100,
    modelId: "asr-new",
    error: null,
  });
});

afterEach(() => {
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

describe("TranscribePanel", () => {
  it("explains the empty installed-model state and offers installation", async () => {
    vi.mocked(transcribeService.fetchInstalledAsrModels).mockResolvedValue([]);
    vi.mocked(transcribeService.searchAsrModels).mockResolvedValue({
      results: [SEARCH_RESULT],
    });

    renderPanel();

    expect(
      await screen.findByText(en["transcribe.models.noneTitle"]),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("combobox", {
        name: en["transcribe.model.label"],
      }),
    ).not.toBeInTheDocument();
    const installButton = await screen.findByRole("button", {
      name: en["transcribe.models.install"],
    });
    expect(installButton).toBeEnabled();
  });

  it("lets the user choose an installed model and submit it", async () => {
    vi.mocked(transcribeService.fetchInstalledAsrModels).mockResolvedValue([
      MODEL_A,
      MODEL_B,
    ]);
    renderPanel();

    const modelSelect = await screen.findByRole("combobox", {
      name: en["transcribe.model.label"],
    });
    fireEvent.change(modelSelect, { target: { value: MODEL_B.id } });
    await submitAudio();

    expect(
      vi.mocked(transcribeService.createTranscribeJob).mock.calls[0][0],
    ).toEqual(expect.objectContaining({ modelId: MODEL_B.id }));
  });

  it("offers the video outputs only when the input is a video", async () => {
    // Pedir "video con subtitulos" para un .wav no tiene sentido: no hay imagen
    // a la que pegarle nada. Se afirma la PROPIEDAD y no el mecanismo: antes el
    // selector entero se escondia, y eso dejo de ser cierto cuando aparecio un
    // modo de salida en video que no necesita video de entrada.
    renderPanel();
    await selectAudioFile();
    const conAudio = await screen.findByLabelText(en["transcribe.output.label"]);

    expect(
      within(conAudio).queryByRole("option", { name: en["transcribe.output.video"] }),
    ).not.toBeInTheDocument();
    expect(
      within(conAudio).queryByRole("option", { name: en["transcribe.output.videoBurned"] }),
    ).not.toBeInTheDocument();

    await selectVideoFile();
    const conVideo = await screen.findByLabelText(en["transcribe.output.label"]);

    expect(
      within(conVideo).getByRole("option", { name: en["transcribe.output.video"] }),
    ).toBeInTheDocument();
  });

  it("offers karaoke for an audio file, which needs no picture of its own", async () => {
    // Es el unico modo con video de SALIDA que no pide video de ENTRADA: el
    // fondo lo genera el servidor.
    renderPanel();
    await selectAudioFile();
    const selector = await screen.findByLabelText(en["transcribe.output.label"]);

    expect(
      within(selector).getByRole("option", { name: en["transcribe.output.karaoke"] }),
    ).toBeInTheDocument();
  });

  it("sends karaoke for an audio file instead of dropping it", async () => {
    renderPanel();
    await selectAudioFile();
    const selector = await screen.findByLabelText(en["transcribe.output.label"]);
    fireEvent.change(selector, { target: { value: "karaoke" } });

    fireEvent.click(screen.getByRole("button", { name: en["transcribe.submit"] }));

    // El filtro que descarta modos de video para archivos sin imagen no puede
    // llevarse puesto al unico que si funciona sin ella.
    await waitFor(() =>
      expect(transcribeService.createTranscribeJob).toHaveBeenCalledWith(
        expect.objectContaining({ outputMode: "karaoke" }),
        expect.anything(),
      ),
    );
  });

  it("sends the chosen output mode", async () => {
    renderPanel();
    await selectVideoFile();
    const select = await screen.findByLabelText(en["transcribe.output.label"]);
    fireEvent.change(select, { target: { value: "video_burned" } });
    const button = screen.getByRole("button", { name: en["transcribe.submit"] });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() =>
      expect(transcribeService.createTranscribeJob).toHaveBeenCalled(),
    );
    expect(
      vi.mocked(transcribeService.createTranscribeJob).mock.calls[0][0],
    ).toEqual(expect.objectContaining({ outputMode: "video_burned" }));
  });

  it("offers the subtitle file and the video when the job produced them", async () => {
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue(
      job({ videoUrl: "/api/v1/transcribe/jobs/tr-1/download?fmt=video" }),
    );
    renderPanel();
    await submitAudio();

    const subtitles = await screen.findByRole("link", {
      name: en["transcribe.result.downloadSubtitles"],
    });
    expect(subtitles).toHaveAttribute(
      "href",
      "/api/v1/transcribe/jobs/tr-1/download?fmt=srt",
    );
    expect(
      screen.getByRole("link", { name: en["transcribe.result.downloadVideo"] }),
    ).toHaveAttribute("href", "/api/v1/transcribe/jobs/tr-1/download?fmt=video");
  });

  it("hides the video download when the job never made one", async () => {
    renderPanel();
    await submitAudio();

    await screen.findByRole("link", {
      name: en["transcribe.result.downloadSubtitles"],
    });
    expect(
      screen.queryByRole("link", { name: en["transcribe.result.downloadVideo"] }),
    ).not.toBeInTheDocument();
  });

  it("asks into which language to dub, and only when dubbing is chosen", async () => {
    vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({
      pairs: [{ source: "en", target: "es" }],
      installable: [],
    });
    renderPanel();
    await selectVideoFile();
    const salida = await screen.findByLabelText(en["transcribe.output.label"]);

    expect(
      screen.queryByLabelText(en["transcribe.dub.language"]),
    ).not.toBeInTheDocument();

    fireEvent.change(salida, { target: { value: "dubbed_video" } });

    expect(
      await screen.findByLabelText(en["transcribe.dub.language"]),
    ).toBeInTheDocument();
  });

  it("sends the language to dub into", async () => {
    vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({
      pairs: [{ source: "en", target: "es" }],
      installable: [],
    });
    renderPanel();
    await selectVideoFile();
    fireEvent.change(await screen.findByLabelText(en["transcribe.output.label"]), {
      target: { value: "dubbed_video" },
    });
    const button = screen.getByRole("button", { name: en["transcribe.submit"] });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() =>
      expect(transcribeService.createTranscribeJob).toHaveBeenCalled(),
    );
    expect(
      vi.mocked(transcribeService.createTranscribeJob).mock.calls[0][0],
    ).toEqual(
      expect.objectContaining({ outputMode: "dubbed_video", targetLanguage: "es" }),
    );
  });

  it("cannot dub without an installed language pair", async () => {
    // Sin par instalado no hay a que idioma doblar: la opcion no se ofrece.
    vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({ pairs: [], installable: [] });
    renderPanel();
    await selectVideoFile();
    const salida = await screen.findByLabelText(en["transcribe.output.label"]);

    expect(
      within(salida).queryByText(en["transcribe.output.dubbed"]),
    ).not.toBeInTheDocument();
  });

  it("says how many lines had to run past their slot", async () => {
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue(
      job({ dubOverflowSegments: 4 }),
    );
    renderPanel();
    await submitAudio();

    expect(await screen.findByRole("status")).toHaveTextContent(/4/);
  });

  it("offers to translate the subtitles when a language pair is installed", async () => {
    vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({
      pairs: [{ source: "en", target: "es" }],
      installable: [],
    });
    renderPanel();
    await submitAudio();

    const select = await screen.findByLabelText(
      en["transcribe.translate.label"],
    );
    fireEvent.change(select, { target: { value: "es" } });

    expect(
      screen.getByRole("link", {
        name: en["transcribe.result.downloadSubtitles"],
      }),
    ).toHaveAttribute(
      "href",
      "/api/v1/transcribe/jobs/tr-1/download?fmt=srt&translate_to=es",
    );
  });

  it("hides the translation picker when no pair is installed", async () => {
    vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({ pairs: [], installable: [] });
    renderPanel();
    await submitAudio();

    await screen.findByRole("link", {
      name: en["transcribe.result.downloadSubtitles"],
    });
    expect(
      screen.queryByLabelText(en["transcribe.translate.label"]),
    ).not.toBeInTheDocument();
  });

  it("ofrece bajar un par cuando no hay ninguno instalado", async () => {
    // Antes esto era un callejon sin salida: sin pares, la traduccion
    // desaparecia de la pantalla y no habia forma de conseguirla.
    vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({
      pairs: [],
      installable: ["en-es", "es-en"],
    });
    renderPanel();
    await submitAudio();

    expect(
      await screen.findByLabelText(en["transcribe.translate.pickPair"]),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /download|descargar/i })).toBeInTheDocument();
  });

  it("no ofrece bajar nada cuando ya hay un par instalado", async () => {
    vi.mocked(transcribeService.fetchTranslationPairs).mockResolvedValue({
      pairs: [{ source: "en", target: "es" }],
      installable: ["es-en"],
    });
    renderPanel();
    await submitAudio();

    await screen.findByLabelText(en["transcribe.translate.label"]);
    expect(
      screen.queryByLabelText(en["transcribe.translate.pickPair"]),
    ).not.toBeInTheDocument();
  });

  it("shows how much of the file has been uploaded", async () => {
    // La subida se queda colgada a proposito: sin resolver, la pantalla no
    // pasa a "en cola" y se puede mirar lo que muestra mientras sube.
    vi.mocked(transcribeService.createTranscribeJob).mockImplementation(
      (_params, options) => {
        options?.onProgress?.(37);
        return new Promise(() => {});
      },
    );
    renderPanel();
    await submitAudio();

    expect(
      await screen.findByRole("progressbar", {
        name: en["transcribe.job.uploading"],
      }),
    ).toHaveAttribute("aria-valuenow", "37");
    expect(screen.getByText("37%")).toBeInTheDocument();
  });

  it("shows live job progress and can cancel the running job", async () => {
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue(
      job({
        status: "running",
        progressPct: 61,
        text: null,
        finishedAt: null,
        downloadUrl: null,
      }),
    );
    renderPanel();
    await submitAudio();

    expect(
      await screen.findByRole("progressbar", {
        name: en["transcribe.job.running"],
      }),
    ).toHaveAttribute("aria-valuenow", "61");
    fireEvent.click(
      screen.getByRole("button", {
        name: en["transcribe.job.cancel"],
      }),
    );

    await waitFor(() =>
      expect(transcribeService.cancelTranscribeJob).toHaveBeenCalledWith(
        "tr-1",
      ),
    );
  });

  it("shows text from the job response without downloading it first", async () => {
    const inlineText =
      "This text came directly from the polled transcription response.";
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue(
      job({ text: inlineText }),
    );
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    renderPanel();
    await submitAudio();

    expect(await screen.findByText(inlineText)).toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: en["transcribe.result.download"],
      }),
    ).toHaveAttribute(
      "href",
      "/api/v1/transcribe/jobs/tr-1/download",
    );
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("does not include language when automatic detection is selected", async () => {
    renderPanel();
    await submitAudio();

    const sent = vi.mocked(transcribeService.createTranscribeJob).mock
      .calls[0][0];
    expect(sent).not.toHaveProperty("language");
  });

  it("includes the selected two-letter language code", async () => {
    renderPanel();
    const languageSelect = await screen.findByRole("combobox", {
      name: en["transcribe.language.label"],
    });
    fireEvent.change(languageSelect, { target: { value: "es" } });
    await submitAudio();

    expect(
      vi.mocked(transcribeService.createTranscribeJob).mock.calls[0][0],
    ).toEqual(expect.objectContaining({ language: "es" }));
  });

  it("sends romanize when Japanese and the romaji checkbox are selected", async () => {
    renderPanel();
    const languageSelect = await screen.findByRole("combobox", {
      name: en["transcribe.language.label"],
    });
    fireEvent.change(languageSelect, { target: { value: "ja" } });
    fireEvent.click(
      screen.getByRole("checkbox", { name: en["transcribe.romanize.label"] }),
    );
    await submitAudio();

    expect(
      vi.mocked(transcribeService.createTranscribeJob).mock.calls[0][0],
    ).toEqual(expect.objectContaining({ language: "ja", romanize: true }));
  });

  it("hides the romaji checkbox for non-Japanese languages", async () => {
    renderPanel();
    const languageSelect = await screen.findByRole("combobox", {
      name: en["transcribe.language.label"],
    });
    fireEvent.change(languageSelect, { target: { value: "es" } });

    expect(
      screen.queryByRole("checkbox", { name: en["transcribe.romanize.label"] }),
    ).not.toBeInTheDocument();
  });

  it("shows the backend error from a failed job", async () => {
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue(
      job({
        status: "failed",
        progressPct: null,
        text: null,
        error: "The selected model cannot decode this audio.",
        finishedAt: null,
        downloadUrl: null,
      }),
    );
    renderPanel();
    await submitAudio();

    expect(
      await screen.findByRole("alert"),
    ).toHaveTextContent("The selected model cannot decode this audio.");
  });

  it("shows ASR installation progress after Install is clicked", async () => {
    vi.mocked(transcribeService.searchAsrModels).mockResolvedValue({
      results: [SEARCH_RESULT],
    });
    vi.mocked(transcribeService.getAsrInstallStatus).mockResolvedValue({
      installId: "install-1",
      repoId: SEARCH_RESULT.id,
      status: "downloading",
      progressPct: 38,
      modelId: null,
      error: null,
    });
    const view = renderPanel();

    fireEvent.click(
      await screen.findByRole("button", {
        name: en["transcribe.models.install"],
      }),
    );

    expect(
      await screen.findByRole("progressbar", {
        name: en["transcribe.models.install.downloading"],
      }),
    ).toHaveAttribute("aria-valuenow", "38");
    view.unmount();
  });
});
