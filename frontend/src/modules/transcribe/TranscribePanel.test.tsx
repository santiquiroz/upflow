import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
