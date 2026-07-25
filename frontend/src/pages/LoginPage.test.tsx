import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../lib/api";
import * as authService from "../services/auth";
import { LoginPage } from "./LoginPage";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, login: vi.fn() };
});

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<LoginPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.login).mockReset();
});

describe("LoginPage", () => {
  it("submits username and password", async () => {
    vi.mocked(authService.login).mockResolvedValue({ ok: true });
    renderPage();

    fireEvent.change(screen.getByLabelText(/usuario/i), { target: { value: "alice" } });
    fireEvent.change(screen.getByLabelText(/contraseña/i), { target: { value: "hunter22" } });
    fireEvent.click(screen.getByRole("button", { name: /ingresar/i }));

    await waitFor(() => expect(authService.login).toHaveBeenCalledWith("alice", "hunter22"));
  });

  it("shows an error message when login fails", async () => {
    vi.mocked(authService.login).mockRejectedValue(new ApiError(401, "Usuario o contraseña incorrectos"));
    renderPage();

    fireEvent.change(screen.getByLabelText(/usuario/i), { target: { value: "alice" } });
    fireEvent.change(screen.getByLabelText(/contraseña/i), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: /ingresar/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Usuario o contraseña incorrectos"));
  });
});
