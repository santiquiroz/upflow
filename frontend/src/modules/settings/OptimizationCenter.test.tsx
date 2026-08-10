import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OptimizationCenter } from "./OptimizationCenter";
import * as api from "../../lib/api";
import type { FixLeverResponse } from "../../lib/apiTypes";
import * as settingsService from "../../services/settings";

const RESIZABLE_BAR_STORAGE_KEY = "upflow.resizableBarConfirmed";

vi.mock("../../services/settings", () => ({ fetchEditableSettings: vi.fn(), patchSetting: vi.fn() }));

function renderWithClient() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <OptimizationCenter />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(settingsService.fetchEditableSettings).mockResolvedValue({
    settings: [{ key: "rebar_confirmed", configured: false, value: "false", requiresRestart: false }],
  });
  vi.mocked(settingsService.patchSetting).mockResolvedValue({ key: "rebar_confirmed" });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.mocked(settingsService.fetchEditableSettings).mockReset();
  vi.mocked(settingsService.patchSetting).mockReset();
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

  it("restores the Resizable BAR confirmation from server settings", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });
    vi.mocked(settingsService.fetchEditableSettings).mockResolvedValue({
      settings: [{ key: "rebar_confirmed", configured: true, value: "true", requiresRestart: false }],
    });

    renderWithClient();
    const checkbox = await screen.findByRole("checkbox", { name: /confirmed resizable bar/i });

    await waitFor(() => expect(checkbox).toBeChecked());
  });

  it("patches the server while keeping the Resizable BAR toggle optimistic", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });
    let resolvePatch!: (value: { key: string }) => void;
    vi.mocked(settingsService.patchSetting).mockReturnValue(
      new Promise((resolve) => {
        resolvePatch = resolve;
      }),
    );

    renderWithClient();
    const checkbox = await screen.findByRole("checkbox", { name: /confirmed resizable bar/i });
    await waitFor(() => expect(settingsService.fetchEditableSettings).toHaveBeenCalled());

    fireEvent.click(checkbox);

    expect(checkbox).toBeChecked();
    await waitFor(() =>
      expect(settingsService.patchSetting).toHaveBeenCalledWith("rebar_confirmed", "true"),
    );

    resolvePatch({ key: "rebar_confirmed" });
  });

  it("migrates a legacy localStorage confirmation to the server once", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "getOnnxDiagnostics").mockResolvedValue({ entries: [] });
    localStorage.setItem(RESIZABLE_BAR_STORAGE_KEY, "true");

    renderWithClient();

    const checkbox = await screen.findByRole("checkbox", { name: /confirmed resizable bar/i });
    await waitFor(() => {
      expect(settingsService.patchSetting).toHaveBeenCalledTimes(1);
      expect(settingsService.patchSetting).toHaveBeenCalledWith("rebar_confirmed", "true");
    });
    expect(checkbox).toBeChecked();
    await waitFor(() => expect(localStorage.getItem(RESIZABLE_BAR_STORAGE_KEY)).toBeNull());
  });
});
