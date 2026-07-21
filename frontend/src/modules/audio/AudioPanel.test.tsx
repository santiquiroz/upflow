import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { AudioCapabilities, AudioJob, CreateJobResponse, DevicesResponse } from "../../lib/apiTypes";
import * as audioService from "../../services/audio";
import { AudioPanel } from "./AudioPanel";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});

vi.mock("../../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/audio")>();
  return { ...actual, createAudioJob: vi.fn(), getAudioJob: vi.fn(), fetchAudioCapabilities: vi.fn() };
});

const DEVICES: DevicesResponse = {
  devices: [
    { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
    { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" },
  ],
  defaultDeviceId: "dml:0",
};

const FULL_CAPABILITIES: AudioCapabilities = {
  denoiseModes: ["deepfilter", "rnnoise"],
  restoreAvailable: true,
  restoreModes: ["apollo", "audiosr"],
};

function renderPanel(capabilities: AudioCapabilities = FULL_CAPABILITIES) {
  vi.mocked(api.getDevices).mockResolvedValue(DEVICES);
  vi.mocked(audioService.fetchAudioCapabilities).mockResolvedValue(capabilities);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<AudioPanel />, { wrapper: Wrapper });
}

function makeFile(): File {
  return new File(["binary"], "voice.wav", { type: "audio/wav" });
}

function selectFile() {
  const fileInput = document.getElementById("audio-file-input") as HTMLInputElement;
  fireEvent.change(fileInput, { target: { files: [makeFile()] } });
}

afterEach(() => {
  vi.mocked(api.getDevices).mockReset();
  vi.mocked(audioService.fetchAudioCapabilities).mockReset();
  vi.mocked(audioService.createAudioJob).mockReset();
  vi.mocked(audioService.getAudioJob).mockReset();
});

describe("AudioPanel", () => {
  it("renders only the denoise modes reported by capabilities", async () => {
    renderPanel({ denoiseModes: ["deepfilter"], restoreAvailable: false, restoreModes: [] });

    expect(await screen.findByRole("button", { name: "DeepFilterNet" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "RNNoise" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "None" })).toBeInTheDocument();
  });

  it("keeps the Enhance CTA disabled until at least one mode is chosen", async () => {
    renderPanel({ denoiseModes: ["deepfilter"], restoreAvailable: false, restoreModes: [] });
    const denoiseButton = await screen.findByRole("button", { name: "DeepFilterNet" });

    selectFile();
    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    expect(submitButton).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent(/at least one/i);

    fireEvent.click(denoiseButton);

    await waitFor(() => expect(submitButton).not.toBeDisabled());
  });

  it("shows the Apollo restore option with an Experimental badge when restore is available", async () => {
    renderPanel(FULL_CAPABILITIES);

    fireEvent.click(await screen.findByRole("button", { name: /^Restore/ }));

    expect(await screen.findByRole("button", { name: /Apollo/ })).toBeInTheDocument();
    // Apollo and AudioSR both carry the badge.
    expect(screen.getAllByText("Experimental")).toHaveLength(2);
  });

  it("offers AudioSR as a restore mode and shows its diffusion cost hint", async () => {
    renderPanel(FULL_CAPABILITIES);

    fireEvent.click(await screen.findByRole("button", { name: /^Restore/ }));
    fireEvent.click(await screen.findByRole("button", { name: /AudioSR/ }));

    expect(await screen.findByText(/per minute of audio/i)).toBeInTheDocument();
  });

  it("hides the Restore section entirely when restore is not available", async () => {
    renderPanel({ denoiseModes: ["deepfilter", "rnnoise"], restoreAvailable: false, restoreModes: [] });
    await screen.findByRole("button", { name: "DeepFilterNet" });

    expect(screen.queryByRole("button", { name: /^Restore/ })).not.toBeInTheDocument();
  });

  it("submits a job with the selected denoise, restore and device and surfaces the download link", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    };
    const completedJob: AudioJob = {
      id: "aud-1",
      status: "completed",
      originalFilename: "voice.wav",
      denoise: "deepfilter",
      restore: "apollo",
      device: "auto",
      progressPct: null,
      stages: null,
      error: null,
      downloadUrl: "/api/v1/audio/jobs/aud-1/download",
    };
    vi.mocked(audioService.createAudioJob).mockResolvedValue(createResponse);
    vi.mocked(audioService.getAudioJob).mockResolvedValue(completedJob);

    renderPanel(FULL_CAPABILITIES);

    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));
    fireEvent.click(await screen.findByRole("button", { name: /^Restore/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Apollo/ }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(await screen.findByRole("link", { name: /download/i })).toHaveAttribute(
      "href",
      "/api/v1/audio/jobs/aud-1/download",
    );
    expect(vi.mocked(audioService.createAudioJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ denoise: "deepfilter", restore: "apollo", device: "cpu" }),
    );
  });

  it("shows format options with friendly descriptions and defaults to FLAC", async () => {
    renderPanel();

    expect(await screen.findByRole("radio", { name: /flac/i })).toBeChecked();
    expect(screen.getByText(/lossless.*50%|50%.*lighter|smaller/i)).toBeInTheDocument();
    expect(screen.getByText(/only if.*size/i)).toBeInTheDocument();
  });

  it("submits the selected output format", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "aud-2",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-2",
      downloadUrl: null,
    };
    vi.mocked(audioService.createAudioJob).mockResolvedValueOnce(createResponse);
    vi.mocked(audioService.getAudioJob).mockResolvedValue({
      id: "aud-2",
      status: "queued",
      originalFilename: "voice.wav",
      denoise: "deepfilter",
      restore: null,
      device: null,
      progressPct: null,
      stages: null,
      error: null,
      downloadUrl: null,
    });

    renderPanel();

    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: "DeepFilterNet" }));
    fireEvent.click(screen.getByRole("radio", { name: /^wav$/i }));

    const submitButton = screen.getByRole("button", { name: /enhance audio/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(audioService.createAudioJob).toHaveBeenCalled());
    expect(vi.mocked(audioService.createAudioJob).mock.calls[0][0].outputFormat).toBe("wav");
  });
});
