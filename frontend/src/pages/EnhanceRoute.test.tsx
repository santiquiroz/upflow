import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as capabilitiesService from "../services/capabilities";
import { EnhanceRoute } from "./EnhanceRoute";
import { treeWithRestore } from "./restoreReleaseTestUtils";

vi.mock("../services/capabilities", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../services/capabilities")>()),
  fetchCapabilityTree: vi.fn(),
}));

function renderAt(path: string) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[path]}>{children}</MemoryRouter>
      </QueryClientProvider>
    );
  }
  return render(<EnhanceRoute />, { wrapper: Wrapper });
}

function selectedTab(): string | null {
  return screen.getByRole("tab", { selected: true }).textContent;
}

describe("EnhanceRoute", () => {
  beforeEach(() => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(treeWithRestore(true));
  });

  it("opens the restore tab from /enhance/restore", async () => {
    renderAt("/enhance/restore");

    await screen.findByRole("tab", { name: "Restore photo" });
    expect(selectedTab()).toBe("Restore photo");
  });

  it("opens the image tab from /enhance/restore while restore is not released", async () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(treeWithRestore(false));
    renderAt("/enhance/restore");

    await waitFor(() => expect(capabilitiesService.fetchCapabilityTree).toHaveBeenCalled());
    expect(selectedTab()).toBe("Image");
  });

  it("opens the video tab from /enhance/video", () => {
    renderAt("/enhance/video");

    expect(selectedTab()).toBe("Video");
  });

  it("falls back to the image tab for /enhance and unknown media", () => {
    renderAt("/enhance/nope");

    expect(selectedTab()).toBe("Image");
  });
});
