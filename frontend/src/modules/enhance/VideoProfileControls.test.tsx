import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { EngineInfoResponse, VideoProfileResponse } from "../../lib/apiTypes";
import { VideoProfileControls } from "./VideoProfileControls";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getEngineInfo: vi.fn() };
});

const GENERAL_PROFILE: VideoProfileResponse = {
  key: "general-balanced-4x",
  label: "General Balanced 4x",
  category: "general",
  description: "Good default for long videos.",
  modelKey: "realesrgan-x4plus",
  scale: 4,
  videoCodec: "libx264",
  videoPreset: "medium",
  crf: 18,
  keepAudio: true,
};

const ANIME_PROFILE: VideoProfileResponse = {
  key: "anime-balanced-2x",
  label: "Anime Balanced 2x",
  category: "anime",
  description: "Best starting point for anime episodes.",
  modelKey: "realesr-animevideov3-x2",
  scale: 2,
  videoCodec: "libx264",
  videoPreset: "medium",
  crf: 17,
  keepAudio: true,
};

function engineInfo(videoProfiles: VideoProfileResponse[]): EngineInfoResponse {
  return {
    engine: "realesrgan-ncnn",
    configuredBinary: "vendor/realesrgan/realesrgan-ncnn-vulkan.exe",
    configuredModelsDir: "vendor/realesrgan/models",
    available: true,
    defaultModel: "realesrgan-x4plus",
    allowedScales: [2, 3, 4],
    supportedModels: [],
    videoProfiles,
    ffmpegAvailable: true,
  };
}

function renderPicker(videoProfiles: VideoProfileResponse[], onChange = vi.fn()) {
  vi.mocked(api.getEngineInfo).mockResolvedValue(engineInfo(videoProfiles));
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { onChange, ...render(<VideoProfileControls value={null} onChange={onChange} />, { wrapper: Wrapper }) };
}

afterEach(() => {
  vi.mocked(api.getEngineInfo).mockReset();
});

describe("VideoProfileControls", () => {
  it("groups profiles by category", async () => {
    renderPicker([GENERAL_PROFILE, ANIME_PROFILE]);

    const generalGroup = await screen.findByRole("group", { name: "General" });
    const animeGroup = screen.getByRole("group", { name: "Anime" });

    expect(within(generalGroup).getByText("General Balanced 4x")).toBeInTheDocument();
    expect(within(animeGroup).getByText("Anime Balanced 2x")).toBeInTheDocument();
  });

  it("shows scale and crf as tabular numbers", async () => {
    renderPicker([GENERAL_PROFILE]);

    const meta = await screen.findByText(/4x · libx264 · CRF 18/);
    expect(meta).toHaveClass("font-mono-tabular");
  });

  it("calls onChange with the full profile object when picked", async () => {
    const { onChange } = renderPicker([GENERAL_PROFILE, ANIME_PROFILE]);

    const radio = await screen.findByRole("radio", { name: /Anime Balanced 2x/ });
    radio.click();

    expect(onChange).toHaveBeenCalledWith(ANIME_PROFILE);
  });

  it("shows an error message when the engine request fails", async () => {
    vi.mocked(api.getEngineInfo).mockRejectedValue(new Error("network down"));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    function Wrapper({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
    }
    render(<VideoProfileControls value={null} onChange={vi.fn()} />, { wrapper: Wrapper });

    expect(await screen.findByText(/Could not load video profiles/i)).toBeInTheDocument();
  });
});
