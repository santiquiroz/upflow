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
      { index: 0, start: 0, end: 1.5, text: "teppeki", translation: "muralla", singer: null },
    ],
    singers: [],
    instrumentalUrl: "/api/v1/karaoke/jobs/kj-1/instrumental",
    sourceHasPicture: true,
    downloadUrl: null,
    practiceMixUrl: null,
  };
}

function singersJob(): karaokeService.KaraokeJob {
  return {
    ...reviewJob(),
    singers: [
      { id: "s1", label: "Singer 1" },
      { id: "s2", label: "Singer 2" },
    ],
    lines: [
      { index: 0, start: 0, end: 1.5, text: "teppeki", translation: "muralla", singer: "s1" },
      { index: 1, start: 2, end: 3.5, text: "kokoro", translation: "corazón", singer: "s2" },
    ],
  };
}

beforeEach(() => {
  // El historial de llamadas sobrevive entre tests del mismo archivo: sin
  // limpiarlo, un assert sobre mock.calls[0] lee la llamada de OTRO test.
  vi.clearAllMocks();
  vi.mocked(karaokeService.createKaraokeJob).mockResolvedValue({
    jobId: "kj-1",
    status: "queued",
    statusUrl: "/api/v1/karaoke/jobs/kj-1",
    downloadUrl: null,
  });
  vi.mocked(karaokeService.getKaraokeJob).mockResolvedValue(reviewJob());
  vi.mocked(karaokeService.updateKaraokeLyrics).mockResolvedValue(reviewJob());
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

// --- deteccion de cantantes (F2a) ---------------------------------------

async function submitDefault(): Promise<void> {
  await pickFile();
  const submit = screen.getByRole("button", { name: en["karaoke.submit"] });
  await waitFor(() => expect(submit).toBeEnabled());
  fireEvent.click(submit);
}

const badgeName = (index: number) =>
  en["karaoke.singers.badgeLabel"].replace("{{index}}", String(index));
const renameName = (label: string) =>
  en["karaoke.singers.renameLabel"].replace("{{label}}", label);
const colorName = (name: string) =>
  en["karaoke.singers.colorLabel"].replace("{{name}}", name);

describe("KaraokeStudioPanel singer detection", () => {
  it("reveals the singer count only when detection is on and sends both fields", async () => {
    renderPanel();
    await pickFile();
    expect(
      screen.queryByRole("combobox", { name: en["karaoke.singers.count"] }),
    ).toBeNull();

    fireEvent.click(
      screen.getByRole("checkbox", { name: en["karaoke.singers.toggle"] }),
    );
    const count = await screen.findByRole("combobox", {
      name: en["karaoke.singers.count"],
    });
    fireEvent.change(count, { target: { value: "3" } });

    const submit = screen.getByRole("button", { name: en["karaoke.submit"] });
    await waitFor(() => expect(submit).toBeEnabled());
    fireEvent.click(submit);

    await waitFor(() => expect(karaokeService.createKaraokeJob).toHaveBeenCalled());
    // lastCall: los mocks del modulo acumulan llamadas de los tests previos.
    const sent = vi.mocked(karaokeService.createKaraokeJob).mock.lastCall?.[0];
    expect(sent).toEqual(
      expect.objectContaining({ detectSingers: true, singerCount: 3 }),
    );
  });

  it("cycles the line's singer badge and saves the reassignment", async () => {
    vi.mocked(karaokeService.getKaraokeJob).mockResolvedValue(singersJob());
    renderPanel();
    await submitDefault();

    const badge = await screen.findByRole("button", { name: badgeName(1) });
    expect(badge).toHaveTextContent("Singer 1");
    fireEvent.click(badge);
    expect(badge).toHaveTextContent("Singer 2");

    fireEvent.click(screen.getByRole("button", { name: en["karaoke.review.save"] }));
    await waitFor(() =>
      expect(karaokeService.updateKaraokeLyrics).toHaveBeenCalled(),
    );
    const [jobId, lines] =
      vi.mocked(karaokeService.updateKaraokeLyrics).mock.lastCall ?? [];
    expect(jobId).toBe("kj-1");
    expect(lines).toEqual([{ index: 0, singer: "s2" }]);
  });

  it("saves singer renames through the lyrics endpoint", async () => {
    vi.mocked(karaokeService.getKaraokeJob).mockResolvedValue(singersJob());
    renderPanel();
    await submitDefault();

    const rename = await screen.findByLabelText(renameName("Singer 1"));
    fireEvent.change(rename, { target: { value: "Ana" } });
    fireEvent.click(screen.getByRole("button", { name: en["karaoke.review.save"] }));

    await waitFor(() =>
      expect(karaokeService.updateKaraokeLyrics).toHaveBeenCalled(),
    );
    const [, lines, singers] =
      vi.mocked(karaokeService.updateKaraokeLyrics).mock.lastCall ?? [];
    expect(lines).toEqual([]);
    expect(singers).toEqual([
      { id: "s1", label: "Ana" },
      { id: "s2", label: "Singer 2" },
    ]);
  });

  it("sends the singer palette and the practiced singer on render", async () => {
    vi.mocked(karaokeService.getKaraokeJob).mockResolvedValue(singersJob());
    renderPanel();
    await submitDefault();

    const color = await screen.findByLabelText(colorName("Singer 1"));
    fireEvent.change(color, { target: { value: "#123456" } });
    const practice = screen.getByRole("combobox", {
      name: en["karaoke.singers.practiceAs"],
    });
    fireEvent.change(practice, { target: { value: "s2" } });

    fireEvent.click(screen.getByRole("button", { name: en["karaoke.render"] }));
    await waitFor(() => expect(karaokeService.renderKaraokeJob).toHaveBeenCalled());
    const [, params] =
      vi.mocked(karaokeService.renderKaraokeJob).mock.lastCall ?? [];
    expect(params?.muteSinger).toBe("s2");
    expect(params?.singerColors?.s1).toBe("#123456");
    expect(Object.keys(params?.singerColors ?? {}).sort()).toEqual(["s1", "s2"]);
  });

  it("warns that unison singing cannot be split", async () => {
    vi.mocked(karaokeService.getKaraokeJob).mockResolvedValue(singersJob());
    renderPanel();
    await submitDefault();

    await screen.findByRole("button", { name: en["karaoke.render"] });
    expect(
      screen.getByText(en["karaoke.singers.unisonWarning"]),
    ).toBeInTheDocument();
  });

  it("keeps the singer controls out of jobs without detection", async () => {
    renderPanel();
    await submitDefault();

    await screen.findByRole("button", { name: en["karaoke.render"] });
    expect(
      screen.queryByRole("combobox", { name: en["karaoke.singers.practiceAs"] }),
    ).toBeNull();
    expect(screen.queryByText(en["karaoke.singers.unisonWarning"])).toBeNull();
  });

  it("offers the practice mix download when the render muted a singer", async () => {
    vi.mocked(karaokeService.getKaraokeJob).mockResolvedValue({
      ...singersJob(),
      phase: "completed",
      downloadUrl: "/api/v1/karaoke/jobs/kj-1/download",
      practiceMixUrl: "/api/v1/karaoke/jobs/kj-1/practice-mix",
    });
    renderPanel();
    await submitDefault();

    const link = await screen.findByRole("link", {
      name: en["karaoke.singers.practiceDownload"],
    });
    expect(link).toHaveAttribute("href", "/api/v1/karaoke/jobs/kj-1/practice-mix");
  });
});
