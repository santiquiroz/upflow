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
});
