import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { DeviceInfoResponse, DevicesResponse } from "../lib/apiTypes";
import { DevicePicker } from "./DevicePicker";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});

const CPU_DEVICE: DeviceInfoResponse = { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" };
const GPU_DEVICE: DeviceInfoResponse = { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" };

function renderPicker(devices: DeviceInfoResponse[], defaultDeviceId: string, requiresGpu: boolean, onChange = vi.fn()) {
  vi.mocked(api.getDevices).mockResolvedValue({ devices, defaultDeviceId } satisfies DevicesResponse);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return {
    onChange,
    ...render(<DevicePicker value={null} onChange={onChange} requiresGpu={requiresGpu} />, { wrapper: Wrapper }),
  };
}

afterEach(() => {
  vi.mocked(api.getDevices).mockReset();
});

describe("DevicePicker", () => {
  it("marks the default device with a visible label", async () => {
    renderPicker([CPU_DEVICE, GPU_DEVICE], "dml:0", false);

    const gpuOption = await screen.findByRole("radio", { name: /AMD Radeon RX 7900/ });
    const gpuLabel = gpuOption.closest("label");
    expect(gpuLabel).not.toBeNull();
    expect(gpuLabel).toHaveTextContent("Default");

    const cpuOption = screen.getByRole("radio", { name: /CPU/ });
    const cpuLabel = cpuOption.closest("label");
    expect(cpuLabel).not.toHaveTextContent("Default");
  });

  it("disables the cpu device with a hint when the selected model requires a Vulkan GPU", async () => {
    renderPicker([CPU_DEVICE, GPU_DEVICE], "dml:0", true);

    const cpuOption = await screen.findByRole("radio", { name: /CPU/ });
    expect(cpuOption).toBeDisabled();
    expect(cpuOption.closest("label")).toHaveTextContent(/vulkan/i);

    const gpuOption = screen.getByRole("radio", { name: /AMD Radeon RX 7900/ });
    expect(gpuOption).not.toBeDisabled();
  });

  it("leaves the cpu device enabled when the selected model does not require a GPU", async () => {
    renderPicker([CPU_DEVICE, GPU_DEVICE], "dml:0", false);

    const cpuOption = await screen.findByRole("radio", { name: /CPU/ });
    expect(cpuOption).not.toBeDisabled();
  });

  it("calls onChange with the selected device", async () => {
    const { onChange } = renderPicker([CPU_DEVICE, GPU_DEVICE], "dml:0", false);

    const gpuOption = await screen.findByRole("radio", { name: /AMD Radeon RX 7900/ });
    fireEvent.click(gpuOption);

    expect(onChange).toHaveBeenCalledWith(GPU_DEVICE);
  });

  it("shows an Auto option ahead of the real devices", async () => {
    renderPicker([CPU_DEVICE, GPU_DEVICE], "dml:0", false);

    const autoOption = await screen.findByRole("radio", { name: /Auto/ });
    expect(autoOption).not.toBeDisabled();
    expect(autoOption.closest("label")).toHaveTextContent(/least busy compatible device/i);
  });

  it("never disables the Auto option even when the selected model requires a Vulkan GPU", async () => {
    renderPicker([CPU_DEVICE, GPU_DEVICE], "dml:0", true);

    const autoOption = await screen.findByRole("radio", { name: /Auto/ });
    expect(autoOption).not.toBeDisabled();
  });

  it("calls onChange with the auto sentinel when Auto is selected", async () => {
    const { onChange } = renderPicker([CPU_DEVICE, GPU_DEVICE], "dml:0", false);

    const autoOption = await screen.findByRole("radio", { name: /Auto/ });
    fireEvent.click(autoOption);

    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ id: "auto", kind: "auto", name: "Auto" }),
    );
  });

  it("shows an error message when the devices request fails", async () => {
    vi.mocked(api.getDevices).mockRejectedValue(new Error("network down"));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    function Wrapper({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
    }
    render(<DevicePicker value={null} onChange={vi.fn()} requiresGpu={false} />, { wrapper: Wrapper });

    expect(await screen.findByText(/Could not load devices/i)).toBeInTheDocument();
  });
});
