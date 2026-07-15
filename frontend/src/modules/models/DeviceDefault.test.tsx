import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { DevicesResponse } from "../../lib/apiTypes";
import { DeviceDefault } from "./DeviceDefault";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});

function renderPanel() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<DeviceDefault />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.getDevices).mockReset();
});

describe("DeviceDefault", () => {
  it("shows the name of the current default device", async () => {
    const payload: DevicesResponse = {
      devices: [
        { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
        { id: "dml:0", kind: "gpu", name: "AMD GPU", backend: "directml" },
      ],
      defaultDeviceId: "dml:0",
    };
    vi.mocked(api.getDevices).mockResolvedValue(payload);

    renderPanel();

    expect(await screen.findByText("AMD GPU")).toBeInTheDocument();
  });

  it("explains the default device is set automatically since there is no settings endpoint yet", async () => {
    vi.mocked(api.getDevices).mockResolvedValue({
      devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
      defaultDeviceId: "cpu",
    });

    renderPanel();

    expect(await screen.findByText(/chosen automatically/i)).toBeInTheDocument();
  });

  it("shows an error state when the devices request fails", async () => {
    vi.mocked(api.getDevices).mockRejectedValue(new Error("network down"));

    renderPanel();

    expect(await screen.findByText(/could not load device info/i)).toBeInTheDocument();
  });
});
