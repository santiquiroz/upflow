import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EditableSettingsSection } from "./EditableSettingsSection";

const fetchEditableSettings = vi.hoisted(() => vi.fn());
const patchSetting = vi.hoisted(() => vi.fn());

vi.mock("../../services/settings", () => ({ fetchEditableSettings, patchSetting }));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

afterEach(() => {
  fetchEditableSettings.mockReset();
  patchSetting.mockReset();
});

describe("EditableSettingsSection", () => {
  it("muestra HF token como no configurado y guarda con PATCH", async () => {
    fetchEditableSettings.mockResolvedValue({ settings: [{ key: "hf_token", configured: false }] });
    patchSetting.mockResolvedValue({ key: "hf_token" });
    render(<EditableSettingsSection />, { wrapper: createWrapper() });

    expect(await screen.findByText(/not configured/i)).toBeInTheDocument();
    const input = screen.getByLabelText(/hugging face token/i);
    expect(input).toHaveAttribute("type", "password");

    fireEvent.change(input, { target: { value: "hf_secret" } });
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() => expect(patchSetting).toHaveBeenCalledWith("hf_token", "hf_secret"));
    expect(await screen.findByText(/saved/i)).toBeInTheDocument();
  });

  it("muestra el error del backend si el PATCH falla", async () => {
    fetchEditableSettings.mockResolvedValue({ settings: [{ key: "hf_token", configured: true }] });
    patchSetting.mockRejectedValue(new Error("Valor inválido para hf_token"));
    render(<EditableSettingsSection />, { wrapper: createWrapper() });

    fireEvent.change(await screen.findByLabelText(/hugging face token/i), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Valor inválido");
  });
});
