import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { DevicesResponse, EngineInfoResponse, HealthResponse } from "../../lib/apiTypes";
import { SettingsPage } from "./SettingsPage";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getEngineInfo: vi.fn(), getHealth: vi.fn(), getDevices: vi.fn() };
});

const ENGINE_INFO: EngineInfoResponse = {
  engine: "realesrgan-ncnn",
  configuredBinary: "vendor/realesrgan/realesrgan-ncnn-vulkan.exe",
  configuredModelsDir: "vendor/realesrgan/models",
  available: true,
  defaultModel: "realesrgan-x4plus",
  allowedScales: [2, 3, 4],
  supportedModels: [],
  videoProfiles: [],
  ffmpegAvailable: true,
};

const HEALTH: HealthResponse = {
  status: "ok",
  engine: "realesrgan-ncnn",
  gpuConcurrency: 1,
  queueDepth: 0,
  videoQueueDepth: 2,
};

const DEVICES: DevicesResponse = {
  devices: [
    { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
    { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" },
  ],
  defaultDeviceId: "dml:0",
};

function renderPage(engine: EngineInfoResponse = ENGINE_INFO) {
  vi.mocked(api.getEngineInfo).mockResolvedValue(engine);
  vi.mocked(api.getHealth).mockResolvedValue(HEALTH);
  vi.mocked(api.getDevices).mockResolvedValue(DEVICES);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<SettingsPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.getEngineInfo).mockReset();
  vi.mocked(api.getHealth).mockReset();
  vi.mocked(api.getDevices).mockReset();
});

describe("SettingsPage", () => {
  it("shows a clear explanation that settings are read-only and come from .env", () => {
    renderPage();

    expect(screen.getByText(/\.env configuration/i)).toBeInTheDocument();
    expect(screen.getByText(/nothing on this page is editable/i)).toBeInTheDocument();
  });

  it("shows the read-only engine configuration once loaded", async () => {
    renderPage();

    expect(await screen.findByText("realesrgan-ncnn")).toBeInTheDocument();
    expect(screen.getByText("realesrgan-x4plus")).toBeInTheDocument();
    expect(screen.getByText("2x, 3x, 4x")).toHaveClass("font-mono-tabular");
  });

  it("shows engine and ffmpeg availability as icon+text, not color alone", async () => {
    renderPage();

    const available = await screen.findAllByText("Available");
    expect(available.length).toBeGreaterThanOrEqual(2);
  });

  it("shows unavailable engine binary in a distinguishable way when the engine is not available", async () => {
    renderPage({ ...ENGINE_INFO, available: false });

    expect((await screen.findAllByText("Unavailable")).length).toBeGreaterThanOrEqual(1);
  });

  it("shows live capacity numbers as tabular figures", async () => {
    renderPage();

    const queueDepth = await screen.findByText("0");
    expect(queueDepth).toHaveClass("font-mono-tabular");
    expect(screen.getByText("2")).toHaveClass("font-mono-tabular");
  });

  it("reuses the DeviceDefault panel to show the default device", async () => {
    renderPage();

    expect(await screen.findByText("AMD Radeon RX 7900")).toBeInTheDocument();
    expect(screen.getByText(/chosen automatically/i)).toBeInTheDocument();
  });

  it("shows an error state when the engine request fails without crashing the page", async () => {
    vi.mocked(api.getHealth).mockResolvedValue(HEALTH);
    vi.mocked(api.getDevices).mockResolvedValue(DEVICES);
    vi.mocked(api.getEngineInfo).mockRejectedValue(new Error("network down"));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<SettingsPage />, {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      ),
    });

    expect(await screen.findByText(/could not load engine info/i)).toBeInTheDocument();
  });
});
