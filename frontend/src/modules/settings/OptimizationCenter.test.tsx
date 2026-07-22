import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OptimizationCenter } from "./OptimizationCenter";
import * as api from "../../lib/api";
import type { FixLeverResponse } from "../../lib/apiTypes";

const RESIZABLE_BAR_STORAGE_KEY = "upflow.resizableBarConfirmed";

function renderWithClient() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <OptimizationCenter />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("OptimizationCenter", () => {
  it("renders a row per lever with its status", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      levers: [
        { id: "hags", label: "Hardware-accelerated GPU scheduling", status: "ok", detail: "enabled", fixable: false },
        { id: "defender_exclusion", label: "Windows Defender exclusion", status: "unavailable", detail: "not excluded", fixable: true },
      ],
    });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });

    renderWithClient();

    expect(await screen.findByText("Hardware-accelerated GPU scheduling")).toBeInTheDocument();
    expect(screen.getByText("Windows Defender exclusion")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /fix/i })).toBeInTheDocument();
  });

  it("calls fixLever when the Fix button is clicked", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      levers: [{ id: "hags", label: "HAGS", status: "unavailable", detail: "disabled", fixable: true }],
    });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });
    const fixSpy = vi.spyOn(api, "fixLever").mockResolvedValue({
      lever: { id: "hags", label: "HAGS", status: "ok", detail: "fixed", fixable: false },
    });

    renderWithClient();
    const button = await screen.findByRole("button", { name: /fix/i });
    fireEvent.click(button);

    await waitFor(() => expect(fixSpy).toHaveBeenCalledWith("hags"));
  });

  it("disables the Fix button while the fix is in flight and re-enables once settled", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      levers: [{ id: "hags", label: "HAGS", status: "unavailable", detail: "disabled", fixable: true }],
    });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });
    let resolveFix!: (value: FixLeverResponse) => void;
    vi.spyOn(api, "fixLever").mockReturnValue(
      new Promise((resolve) => {
        resolveFix = resolve;
      }),
    );

    renderWithClient();
    const button = await screen.findByRole("button", { name: /fix/i });
    fireEvent.click(button);

    expect(await screen.findByRole("button", { name: /fixing/i })).toBeDisabled();

    resolveFix({ lever: { id: "hags", label: "HAGS", status: "ok", detail: "fixed", fixable: false } });

    await waitFor(() => expect(screen.queryByRole("button", { name: /fix/i })).not.toBeInTheDocument());
  });

  it("renders the Resizable BAR checklist section", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });

    renderWithClient();

    expect(await screen.findByRole("heading", { name: /Resizable BAR/i })).toBeInTheDocument();
  });

  it("persists the Resizable BAR confirmation to localStorage and restores it on remount", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });

    const { unmount } = renderWithClient();
    const checkbox = await screen.findByRole("checkbox", { name: /confirmed resizable bar/i });
    expect(checkbox).not.toBeChecked();

    fireEvent.click(checkbox);

    expect(checkbox).toBeChecked();
    expect(localStorage.getItem(RESIZABLE_BAR_STORAGE_KEY)).toBe("true");

    unmount();
    renderWithClient();

    const restoredCheckbox = await screen.findByRole("checkbox", { name: /confirmed resizable bar/i });
    expect(restoredCheckbox).toBeChecked();
  });
});
