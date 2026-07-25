import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as authService from "../services/auth";
import { SetupPage } from "./SetupPage";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, setup: vi.fn() };
});

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<SetupPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.setup).mockReset();
});

describe("SetupPage", () => {
  it("submits username and password to create the first admin", async () => {
    vi.mocked(authService.setup).mockResolvedValue({ ok: true });
    renderPage();

    fireEvent.change(screen.getByLabelText(/usuario/i), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText(/contraseña/i), { target: { value: "adminpass1" } });
    fireEvent.click(screen.getByRole("button", { name: /crear/i }));

    await waitFor(() => expect(authService.setup).toHaveBeenCalledWith("admin", "adminpass1"));
  });
});
