import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { DevicesResponse } from "../../lib/apiTypes";
import { DeviceAcceleration } from "./DeviceAcceleration";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});

function renderWith(devices: DevicesResponse) {
  vi.mocked(api.getDevices).mockResolvedValue(devices);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<DeviceAcceleration />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.getDevices).mockReset();
});

describe("DeviceAcceleration", () => {
  it("shows the baseline provider for each device", async () => {
    renderWith({
      devices: [
        { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu", activeEp: "CPUExecutionProvider", epLabel: "CPU", epState: "baseline" },
        { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7800 XT", backend: "directml", activeEp: "DmlExecutionProvider", epLabel: "DirectML", epState: "baseline" },
      ],
      defaultDeviceId: "dml:0",
    });

    expect(await screen.findByText("AMD Radeon RX 7800 XT")).toBeInTheDocument();
    expect(screen.getByText("DirectML")).toBeInTheDocument();
  });

  it("marks a native provider as native", async () => {
    renderWith({
      devices: [
        { id: "dml:0", kind: "gpu", name: "NVIDIA GeForce RTX 4070", backend: "directml", activeEp: "NvTensorRTRTXExecutionProvider", epLabel: "TensorRT-RTX", epState: "native" },
      ],
      defaultDeviceId: "dml:0",
    });

    expect(await screen.findByText("TensorRT-RTX · native")).toBeInTheDocument();
  });

  it("shows the preparing copy while the native EP compiles", async () => {
    renderWith({
      devices: [
        { id: "dml:0", kind: "gpu", name: "NVIDIA GeForce RTX 4070", backend: "directml", activeEp: "DmlExecutionProvider", epLabel: "DirectML", epState: "preparing", epDetail: "preparando aceleración para tu GPU" },
      ],
      defaultDeviceId: "dml:0",
    });

    expect(await screen.findByText("Preparing acceleration for your GPU…")).toBeInTheDocument();
  });

  it("surfaces a native failure as fallback with the detail as tooltip", async () => {
    renderWith({
      devices: [
        { id: "dml:0", kind: "gpu", name: "NVIDIA GeForce RTX 4070", backend: "directml", activeEp: "DmlExecutionProvider", epLabel: "DirectML", epState: "error", epDetail: "TensorRT-RTX: unsupported op" },
      ],
      defaultDeviceId: "dml:0",
    });

    const row = await screen.findByText("DirectML · fallback (native failed)");
    expect(row).toHaveAttribute("title", "TensorRT-RTX: unsupported op");
  });

  it("tolerates devices without EP fields (older backend)", async () => {
    renderWith({
      devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
      defaultDeviceId: "cpu",
    });

    expect(await screen.findByText("CPU")).toBeInTheDocument();
  });
});
