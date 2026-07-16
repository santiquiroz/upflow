import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";
import { EnhancePage } from "./EnhancePage";

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<EnhancePage />, { wrapper: Wrapper });
}

describe("EnhancePage", () => {
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

  it("wraps from the last tab back to the first on ArrowRight", () => {
    renderPage();

    const imageTab = screen.getByRole("tab", { name: /image/i });
    const videoTab = screen.getByRole("tab", { name: /video/i });
    videoTab.focus();
    fireEvent.click(videoTab);

    fireEvent.keyDown(videoTab, { key: "ArrowRight" });

    expect(imageTab).toHaveAttribute("aria-selected", "true");
    expect(imageTab).toHaveFocus();
  });

  it("wraps from the first tab back to the last on ArrowLeft", () => {
    renderPage();

    const imageTab = screen.getByRole("tab", { name: /image/i });
    const videoTab = screen.getByRole("tab", { name: /video/i });
    imageTab.focus();

    fireEvent.keyDown(imageTab, { key: "ArrowLeft" });

    expect(videoTab).toHaveAttribute("aria-selected", "true");
    expect(videoTab).toHaveFocus();
  });
});
