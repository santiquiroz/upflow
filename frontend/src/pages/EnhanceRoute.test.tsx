import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { EnhanceRoute } from "./EnhanceRoute";

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
  it("opens the restore tab from /enhance/restore", () => {
    renderAt("/enhance/restore");

    expect(selectedTab()).toBe("Restore photo");
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
