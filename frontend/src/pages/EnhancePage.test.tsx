import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as capabilitiesService from "../services/capabilities";
import { EnhancePage, type EnhanceMedium } from "./EnhancePage";
import { treeWithRestore } from "./restoreReleaseTestUtils";

vi.mock("../services/capabilities", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../services/capabilities")>()),
  fetchCapabilityTree: vi.fn(),
}));

function releaseRestore(released: boolean): void {
  vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(treeWithRestore(released));
}

function renderPage(initialMedium: EnhanceMedium = "image") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<EnhancePage initialMedium={initialMedium} />, { wrapper: Wrapper });
}

describe("EnhancePage", () => {
  beforeEach(() => {
    releaseRestore(true);
  });

  it("shows the image panel by default", () => {
    renderPage();

    expect(screen.getByRole("button", { name: /upscale$/i })).toBeInTheDocument();
  });

  it("switches to the video panel when the Video tab is picked", () => {
    renderPage();

    fireEvent.click(screen.getByRole("tab", { name: /video/i }));

    expect(screen.getByRole("button", { name: /upscale video/i })).toBeInTheDocument();
  });

  it("renders a single coherent h1 shared by both panels", () => {
    renderPage();

    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1, name: "Enhance" })).toBeInTheDocument();
  });

  it("wires each tab to its tabpanel via aria-controls and aria-labelledby", () => {
    renderPage();

    const imageTab = screen.getByRole("tab", { name: /image/i });
    const panel = screen.getByRole("tabpanel");

    expect(imageTab).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", imageTab.id);
  });

  it("only the selected tab is reachable via Tab (roving tabindex)", () => {
    renderPage();

    expect(screen.getByRole("tab", { name: /image/i })).toHaveAttribute("tabIndex", "0");
    expect(screen.getByRole("tab", { name: /video/i })).toHaveAttribute("tabIndex", "-1");
  });

  it("moves selection and focus to the next tab on ArrowRight", () => {
    renderPage();

    const imageTab = screen.getByRole("tab", { name: /image/i });
    const videoTab = screen.getByRole("tab", { name: /video/i });
    imageTab.focus();

    fireEvent.keyDown(imageTab, { key: "ArrowRight" });

    expect(videoTab).toHaveAttribute("aria-selected", "true");
    expect(videoTab).toHaveFocus();
    expect(screen.getByRole("button", { name: /upscale video/i })).toBeInTheDocument();
  });

  it("wraps from the last tab back to the first on ArrowRight", async () => {
    renderPage();

    const restoreTab = await screen.findByRole("tab", { name: "Restore photo" });
    const imageTab = screen.getByRole("tab", { name: /image/i });
    restoreTab.focus();
    fireEvent.click(restoreTab);

    fireEvent.keyDown(restoreTab, { key: "ArrowRight" });

    expect(imageTab).toHaveAttribute("aria-selected", "true");
    expect(imageTab).toHaveFocus();
  });

  it("wraps from the first tab back to the last on ArrowLeft", async () => {
    renderPage();

    const restoreTab = await screen.findByRole("tab", { name: "Restore photo" });
    const imageTab = screen.getByRole("tab", { name: /image/i });
    imageTab.focus();

    fireEvent.keyDown(imageTab, { key: "ArrowLeft" });

    expect(restoreTab).toHaveAttribute("aria-selected", "true");
    expect(restoreTab).toHaveFocus();
  });

  it("puts Restore photo third, after Image and Video", async () => {
    renderPage();

    await screen.findByRole("tab", { name: "Restore photo" });
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual(["Image", "Video", "Restore photo"]);
  });

  it("switches to the photo restoration panel when Restore photo is picked", async () => {
    renderPage();

    fireEvent.click(await screen.findByRole("tab", { name: "Restore photo" }));

    expect(screen.getByText("Drop a photo here or click to browse")).toBeInTheDocument();
    expect(screen.getByText("Repair scratches, fading and noise in an old photo.")).toBeInTheDocument();
  });

  it("opens straight on the restore tab when asked to", async () => {
    renderPage("restore");

    expect(await screen.findByRole("tab", { name: "Restore photo" })).toHaveAttribute("aria-selected", "true");
  });

  it("hides Restore photo while its models are not published", async () => {
    releaseRestore(false);
    renderPage();

    await waitFor(() => expect(capabilitiesService.fetchCapabilityTree).toHaveBeenCalled());
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual(["Image", "Video"]);
    expect(screen.queryByRole("tab", { name: "Restore photo" })).not.toBeInTheDocument();
  });

  it("falls back to the image panel when asked for the hidden restore tab", async () => {
    releaseRestore(false);
    renderPage("restore");

    await waitFor(() => expect(capabilitiesService.fetchCapabilityTree).toHaveBeenCalled());
    expect(screen.getByRole("tab", { name: "Image" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByText("Drop a photo here or click to browse")).not.toBeInTheDocument();
  });

  it("wraps between Image and Video only when restore is hidden", async () => {
    releaseRestore(false);
    renderPage();
    await waitFor(() => expect(capabilitiesService.fetchCapabilityTree).toHaveBeenCalled());

    const imageTab = screen.getByRole("tab", { name: /image/i });
    imageTab.focus();
    fireEvent.keyDown(imageTab, { key: "ArrowLeft" });

    expect(screen.getByRole("tab", { name: /video/i })).toHaveAttribute("aria-selected", "true");
  });
});
