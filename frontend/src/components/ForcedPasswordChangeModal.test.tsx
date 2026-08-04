import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as authService from "../services/auth";
import { ApiError } from "../lib/api";
import { ForcedPasswordChangeModal } from "./ForcedPasswordChangeModal";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, changePassword: vi.fn() };
});

function renderModal() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<ForcedPasswordChangeModal />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.changePassword).mockReset();
});

describe("ForcedPasswordChangeModal", () => {
  it("submits current and new password", async () => {
    vi.mocked(authService.changePassword).mockResolvedValue({ ok: true });
    renderModal();

    fireEvent.change(screen.getByLabelText(/Current password/i), { target: { value: "temp123456" } });
    fireEvent.change(screen.getByLabelText(/New password/i), { target: { value: "newpassword1" } });
    fireEvent.click(screen.getByRole("button", { name: /Save/i }));

    await waitFor(() =>
      expect(authService.changePassword).toHaveBeenCalledWith("temp123456", "newpassword1"),
    );
  });

  it("shows an error message on failure", async () => {
    vi.mocked(authService.changePassword).mockRejectedValue(new ApiError(401, "Contraseña actual incorrecta"));
    renderModal();

    fireEvent.change(screen.getByLabelText(/Current password/i), { target: { value: "wrong" } });
    fireEvent.change(screen.getByLabelText(/New password/i), { target: { value: "newpassword1" } });
    fireEvent.click(screen.getByRole("button", { name: /Save/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Contraseña actual incorrecta"));
  });
});
