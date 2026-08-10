import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { en } from "../i18n/en";
import type { CapabilityTreeResponse, ProvisionJob } from "../lib/apiTypes";
import * as capabilitiesService from "../services/capabilities";
import * as settingsService from "../services/settings";
import { TasksPage } from "./TasksPage";

vi.mock("../services/settings", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/settings")>();
  return { ...actual, patchSetting: vi.fn() };
});

vi.mock("../services/capabilities", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/capabilities")>();
  return {
    ...actual,
    fetchCapabilityTree: vi.fn(),
    provisionCapability: vi.fn(),
    getProvisionStatus: vi.fn(),
  };
});

function leaf(overrides: Partial<CapabilityTreeResponse["domains"][0]["capabilities"][0]> = {}) {
  return {
    id: "video.upscale",
    domain: "video" as const,
    labelKey: "capability.video.upscale",
    status: "available" as const,
    provisioning: "registry" as const,
    jobKind: "video" as string | null,
    strategies: ["model" as const],
    missingPacks: [] as string[],
    unavailableReasonKey: null as string | null,
    setupReasonKey: null as string | null,
    activatableSettings: [] as string[],
    ...overrides,
  };
}

const TREE: CapabilityTreeResponse = {
  domains: [
    {
      domain: "video",
      labelKey: "capability.domain.video",
      capabilities: [
        leaf(),
        leaf({
          id: "video.interpolate",
          labelKey: "capability.video.interpolate",
          status: "needs_setup",
          provisioning: "vendored_pack",
          missingPacks: ["rife"],
          setupReasonKey: "capability.setup.missingPack",
        }),
      ],
      roadmap: [],
    },
  ],
};

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route path="/" element={<TasksPage />} />
        <Route path="/enhance/:medium" element={<p>enhance surface</p>} />
      </Routes>
    </MemoryRouter>,
    { wrapper: Wrapper },
  );
}

function provisionJob(overrides: Partial<ProvisionJob> = {}): ProvisionJob {
  return {
    jobId: "prov-1",
    pack: "rife",
    status: "queued",
    error: null,
    statusUrl: "/api/v1/capabilities/provision/prov-1",
    ...overrides,
  };
}

const NEEDS_FLAG_TREE: CapabilityTreeResponse = {
  domains: [
    {
      domain: "video",
      labelKey: "capability.domain.video",
      capabilities: [
        leaf({
          id: "audio.restoreSr",
          labelKey: "capability.audio.restoreSr",
          status: "needs_setup",
          provisioning: "vendored_pack",
          setupReasonKey: "capability.setup.missingSetting",
          activatableSettings: ["enable_audiosr"],
        }),
      ],
      roadmap: [],
    },
  ],
};

afterEach(() => {
  vi.mocked(capabilitiesService.fetchCapabilityTree).mockReset();
  vi.mocked(capabilitiesService.provisionCapability).mockReset();
  vi.mocked(capabilitiesService.getProvisionStatus).mockReset();
  vi.mocked(settingsService.patchSetting).mockReset();
});

describe("TasksPage", () => {
  it("asks what the user wants to do", () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(TREE);
    renderPage();
    expect(screen.getByRole("heading", { name: en["tasks.title"] })).toBeInTheDocument();
  });

  it("navigates to the surface of the capability picked", async () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(TREE);
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: en["capability.video.upscale"] }),
    );

    expect(await screen.findByText("enhance surface")).toBeInTheDocument();
  });

  it("starts the download of a missing pack", async () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(TREE);
    vi.mocked(capabilitiesService.provisionCapability).mockResolvedValue(provisionJob());
    vi.mocked(capabilitiesService.getProvisionStatus).mockResolvedValue(
      provisionJob({ status: "running" }),
    );
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: /capability\.tree\.download|download package/i }),
    );

    // Se mira el primer argumento y no la llamada entera: TanStack v5 le pasa su
    // propio contexto de mutacion como segundo parametro.
    await waitFor(() =>
      expect(vi.mocked(capabilitiesService.provisionCapability).mock.calls[0][0]).toBe(
        "video.interpolate",
      ),
    );
    expect(await screen.findByText(en["capability.provision.running"])).toBeInTheDocument();
  });

  it("reports a finished download", async () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(TREE);
    vi.mocked(capabilitiesService.provisionCapability).mockResolvedValue(provisionJob());
    vi.mocked(capabilitiesService.getProvisionStatus).mockResolvedValue(
      provisionJob({ status: "done" }),
    );
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /download package/i }));

    expect(await screen.findByText(en["capability.provision.done"])).toBeInTheDocument();
  });

  it("shows the reason when the download fails", async () => {
    // El job puede terminar en error sin que ninguna request falle: se murio el
    // script de descarga, no la API. Ese motivo es el unico que le sirve al
    // usuario.
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(TREE);
    vi.mocked(capabilitiesService.provisionCapability).mockResolvedValue(provisionJob());
    vi.mocked(capabilitiesService.getProvisionStatus).mockResolvedValue(
      provisionJob({ status: "error", error: "404 Not Found" }),
    );
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: /download package/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("404 Not Found");
  });

  it("says nothing about downloads before one is started", () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(TREE);
    renderPage();

    expect(screen.queryByText(en["capability.provision.running"])).not.toBeInTheDocument();
    expect(screen.queryByText(en["capability.provision.done"])).not.toBeInTheDocument();
  });

  it("turns the setting on from the card and re-resolves the tree", async () => {
    // El atajo completo: la tarjeta prende el flag y el arbol se vuelve a
    // resolver, asi la capacidad queda disponible sin pasar por Ajustes.
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(NEEDS_FLAG_TREE);
    vi.mocked(settingsService.patchSetting).mockResolvedValue({ key: "enable_audiosr" });
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: new RegExp(en["capability.tree.activate"]) }),
    );

    await waitFor(() =>
      expect(settingsService.patchSetting).toHaveBeenCalledWith("enable_audiosr", "true"),
    );
    expect(await screen.findByText(en["capability.activation.done"])).toBeInTheDocument();
    await waitFor(() =>
      expect(vi.mocked(capabilitiesService.fetchCapabilityTree).mock.calls.length).toBeGreaterThan(1),
    );
  });

  it("shows the reason when turning the setting on fails", async () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(NEEDS_FLAG_TREE);
    vi.mocked(settingsService.patchSetting).mockRejectedValue(new Error("no es editable"));
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: new RegExp(en["capability.tree.activate"]) }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("no es editable");
  });

  it("says nothing about turning things on before anything is clicked", () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(NEEDS_FLAG_TREE);
    renderPage();

    expect(screen.queryByText(en["capability.activation.done"])).not.toBeInTheDocument();
    expect(screen.queryByText(en["capability.activation.running"])).not.toBeInTheDocument();
  });
});
