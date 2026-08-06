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

  it("shows a registered-but-unused accelerator as ready, not as native", async () => {
    // Decir "native" antes de que corriera un solo trabajo podia ser mentira:
    // el acelerador esta registrado, pero todavia no lo uso nadie.
    renderWith({
      devices: [
        { id: "dml:0", kind: "gpu", name: "NVIDIA GeForce RTX 4070", backend: "directml", activeEp: "NvTensorRTRTXExecutionProvider", epLabel: "TensorRT-RTX", epState: "ready" },
      ],
      defaultDeviceId: "dml:0",
    });

    expect(await screen.findByText(/TensorRT-RTX · ready/)).toBeInTheDocument();
    expect(screen.queryByText(/· native/)).not.toBeInTheDocument();
  });

  it("shows WHY the accelerator failed, without needing to hover", async () => {
    // El caso real: se bajan 93 MB del acelerador, falta el driver, falla. Con
    // el motivo solo en un tooltip, el usuario no se entera de por que.
    renderWith({
      devices: [
        { id: "dml:0", kind: "gpu", name: "NVIDIA GeForce RTX 4070", backend: "directml", activeEp: "DmlExecutionProvider", epLabel: "DirectML", epState: "error", epDetail: "TensorRT-RTX: falta nvml.dll (Error 126)" },
      ],
      defaultDeviceId: "dml:0",
    });

    expect(await screen.findByText(/falta nvml\.dll/)).toBeInTheDocument();
  });

  it("keeps the detail reachable as a tooltip too", async () => {
    renderWith({
      devices: [
        { id: "dml:0", kind: "gpu", name: "NVIDIA GeForce RTX 4070", backend: "directml", activeEp: "DmlExecutionProvider", epLabel: "DirectML", epState: "error", epDetail: "TensorRT-RTX: unsupported op" },
      ],
      defaultDeviceId: "dml:0",
    });

    // El motivo ahora se pinta visible, asi que el texto vive en un span interno;
    // el tooltip sigue estando en el contenedor.
    const row = await screen.findByText("DirectML · fallback (native failed)");
    expect(row.closest("[title]")).toHaveAttribute("title", "TensorRT-RTX: unsupported op");
    expect(screen.getByText("TensorRT-RTX: unsupported op")).toBeInTheDocument();
  });

  it("shows the CPU fallback in red with the reason visible", async () => {
    // El caso del amigo con la GPU fria: ORT cayo a CPU en silencio y el unico
    // sintoma era un trabajo eterno. Este estado lo tiene que gritar la UI.
    renderWith({
      devices: [
        { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900 XT", backend: "directml", activeEp: "CPUExecutionProvider", epLabel: "CPU", epState: "cpu_fallback", epDetail: "DirectML no inicializó en este dispositivo: los trabajos están corriendo en CPU" },
      ],
      defaultDeviceId: "dml:0",
    });

    expect(await screen.findByText("CPU · no acceleration — running on CPU")).toBeInTheDocument();
    expect(screen.getByText(/DirectML no inicializó/)).toBeInTheDocument();
  });

  it("tolerates devices without EP fields (older backend)", async () => {
    renderWith({
      devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
      defaultDeviceId: "cpu",
    });

    expect(await screen.findByText("CPU")).toBeInTheDocument();
  });
});
