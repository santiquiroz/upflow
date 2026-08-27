import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import * as karaokeService from "../../services/karaoke";
import { KaraokeStudioPanel } from "./KaraokeStudioPanel";

vi.mock("../../services/karaoke");

vi.mock("../../hooks/useAudioJob", () => ({
  useAudioCapabilities: () => ({
    data: {
      denoiseModes: [],
      restoreAvailable: true,
      restoreModes: ["apollo", "audiosr"],
      separationModels: [
        {
          id: "mdx_inst_hq3",
          name: "MDX-Net Inst HQ 3",
          category: "karaoke",
          installed: true,
        },
        {
          id: "denoise",
          name: "UVR DeNoise",
          category: "cleanup",
          installed: true,
        },
      ],
      cleanupSteps: [
        {
          id: "denoise",
          name: "UVR DeNoise by FoxJoy",
          family: "denoise",
          covers: ["denoise"],
          installed: true,
          descriptionKey: "audio.karaoke.model.denoise.description",
        },
      ],
      cleanupOverprocessingThreshold: 3,
    },
  }),
}));

vi.mock("../../hooks/useTranscribeJob", () => ({
  useInstalledAsrModels: () => ({
    data: [{ id: "asr-1", name: "onnx-community/whisper-tiny", kind: "asr-onnx" }],
  }),
  useTranscribeDevices: () => ({
    data: {
      devices: [{ id: "cpu", name: "CPU" }],
      defaultDeviceId: "cpu",
    },
  }),
  useAsrModelInstall: () => ({
    phase: "idle",
    progressPct: null,
    errorMessage: null,
    modelId: null,
    install: vi.fn(),
    reset: vi.fn(),
  }),
}));

vi.mock("../../services/transcribe", () => ({
  fetchTranslationPairs: () =>
    Promise.resolve({ pairs: [{ source: "ja", target: "es" }], installable: [] }),
}));

function renderPanel() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <LocaleProvider initialLocale="en">{children}</LocaleProvider>
      </QueryClientProvider>
    );
  }
  return render(<KaraokeStudioPanel />, { wrapper: Wrapper });
}

function reviewJob(): karaokeService.KaraokeJob {
  return {
    id: "kj-1",
    status: "completed",
    phase: "review",
    originalFilename: "cancion.mp4",
    asrModelId: "asr-1",
    separationModelId: null,
    cleanupSteps: [],
    restoreMode: null,
    language: "ja",
    romanize: true,
    translateTo: "es",
    device: "cpu",
    backgroundKind: "generated",
    progressPct: 100,
    error: null,
    lines: [
      { index: 0, start: 0, end: 1.5, text: "teppeki", translation: "muralla" },
    ],
    instrumentalUrl: "/api/v1/karaoke/jobs/kj-1/instrumental",
    sourceHasPicture: true,
    downloadUrl: null,
  };
}

beforeEach(() => {
  vi.mocked(karaokeService.createKaraokeJob).mockResolvedValue({
    jobId: "kj-1",
    status: "queued",
    statusUrl: "/api/v1/karaoke/jobs/kj-1",
    downloadUrl: null,
  });
  vi.mocked(karaokeService.getKaraokeJob).mockResolvedValue(reviewJob());
  vi.mocked(karaokeService.renderKaraokeJob).mockResolvedValue({
    ...reviewJob(),
    phase: "rendering",
  });
});

async function pickFile(): Promise<void> {
  const input = await screen.findByLabelText(en["karaoke.file.inputLabel"]);
  fireEvent.change(input, {
    target: { files: [new File(["v"], "cancion.mp4", { type: "video/mp4" })] },
  });
}

describe("KaraokeStudioPanel", () => {
  it("submits the configured pipeline", async () => {
    renderPanel();
    await pickFile();
    const languageSelect = await screen.findByRole("combobox", {
      name: en["transcribe.language.label"],
    });
    fireEvent.change(languageSelect, { target: { value: "ja" } });
    fireEvent.click(
      await screen.findByRole("checkbox", { name: en["transcribe.romanize.label"] }),
    );
    fireEvent.click(
      screen.getByRole("radio", { name: en["karaoke.config.restore.apollo"] }),
    );

    const submit = screen.getByRole("button", { name: en["karaoke.submit"] });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.click(submit);

    await waitFor(() => expect(karaokeService.createKaraokeJob).toHaveBeenCalled());
    const sent = vi.mocked(karaokeService.createKaraokeJob).mock.calls[0][0];
    expect(sent).toEqual(
      expect.objectContaining({
        asrModelId: "asr-1",
        language: "ja",
        romanize: true,
        restoreMode: "apollo",
      }),
    );
  });

  it("shows the review stage with editable lyrics and render button", async () => {
    renderPanel();
    await pickFile();
    const submit = screen.getByRole("button", { name: en["karaoke.submit"] });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.click(submit);

    expect(
      await screen.findByRole("button", { name: en["karaoke.render"] }),
    ).toBeInTheDocument();
    expect(screen.getByDisplayValue("teppeki")).toBeInTheDocument();
    expect(screen.getByDisplayValue("muralla")).toBeInTheDocument();
  });

  it("sends the chosen background and style on render", async () => {
    renderPanel();
    await pickFile();
    const submit = screen.getByRole("button", { name: en["karaoke.submit"] });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.click(submit);
    const renderButton = await screen.findByRole("button", {
      name: en["karaoke.render"],
    });

    fireEvent.click(
      screen.getByRole("button", { name: en["karaoke.background.generated"] }),
    );
    fireEvent.click(renderButton);

    await waitFor(() => expect(karaokeService.renderKaraokeJob).toHaveBeenCalled());
    const [jobId, params] = vi.mocked(karaokeService.renderKaraokeJob).mock.calls[0];
    expect(jobId).toBe("kj-1");
    expect(params).toEqual(
      expect.objectContaining({
        backgroundKind: "generated",
        subtitleSize: "medium",
        subtitlePosition: "bottom",
      }),
    );
  });
});
