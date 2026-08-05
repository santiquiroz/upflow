import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import * as printService from "../../services/print";
import { PartBuilder } from "./PartBuilder";

vi.mock("../../services/print", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/print")>();
  return { ...actual, fetchPartKinds: vi.fn(), generatePart: vi.fn() };
});

const TUBO: printService.PartKind = {
  id: "tube",
  labelKey: "part.tube",
  descriptionKey: "part.tube.description",
  params: [
    { name: "outer_diameter", labelKey: "part.param.outerDiameter", default: 20, minimum: 0.1 },
    { name: "inner_diameter", labelKey: "part.param.innerDiameter", default: 8.4, minimum: 0.1 },
    { name: "height", labelKey: "part.param.height", default: 12, minimum: 0.1 },
  ],
};

const HECHA: printService.GeneratedPart = {
  canPrint: true,
  sizeMm: [20, 20, 12],
  volumeMm3: 3104.6,
  triangleCount: 512,
  overhangRatio: 0.16,
  blockers: [],
  advice: [],
  downloadUrl: "/api/v1/print/parts/abc",
};

function renderBuilder() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <LocaleProvider initialLocale="en">{children}</LocaleProvider>
      </QueryClientProvider>
    );
  }
  return render(<PartBuilder printer="ender-3" />, { wrapper: Wrapper });
}

beforeEach(() => {
  vi.mocked(printService.fetchPartKinds).mockResolvedValue({ kinds: [TUBO] });
  vi.mocked(printService.generatePart).mockResolvedValue(HECHA);
});

afterEach(() => {
  vi.mocked(printService.fetchPartKinds).mockReset();
  vi.mocked(printService.generatePart).mockReset();
});

describe("PartBuilder", () => {
  it("shows one field per measurement the part needs", async () => {
    renderBuilder();

    expect(await screen.findByLabelText(en["part.param.outerDiameter"])).toBeInTheDocument();
    expect(screen.getByLabelText(en["part.param.innerDiameter"])).toBeInTheDocument();
    expect(screen.getByLabelText(en["part.param.height"])).toBeInTheDocument();
  });

  it("sends the defaults when nothing was typed", async () => {
    // Sin esto, abrir la pantalla y darle a construir mandaria un objeto vacio.
    renderBuilder();
    fireEvent.click(await screen.findByRole("button", { name: en["part.build"] }));

    await waitFor(() => expect(printService.generatePart).toHaveBeenCalled());
    expect(vi.mocked(printService.generatePart).mock.calls[0][0].params).toEqual({
      outer_diameter: 20,
      inner_diameter: 8.4,
      height: 12,
    });
  });

  it("sends the measurement the user typed", async () => {
    renderBuilder();
    fireEvent.change(await screen.findByLabelText(en["part.param.height"]), {
      target: { value: "35.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: en["part.build"] }));

    await waitFor(() => expect(printService.generatePart).toHaveBeenCalled());
    expect(vi.mocked(printService.generatePart).mock.calls[0][0].params.height).toBe(35.5);
  });

  it("shows the real measurements of what it built", async () => {
    renderBuilder();
    fireEvent.click(await screen.findByRole("button", { name: en["part.build"] }));

    expect(await screen.findByText(/20.0 × 20.0 × 12.0 mm/)).toBeInTheDocument();
  });

  it("offers the file even when it does not fit the printer", async () => {
    // La pieza esta bien hecha; lo que no entra es en ESA maquina.
    vi.mocked(printService.generatePart).mockResolvedValue({
      ...HECHA,
      canPrint: false,
      blockers: ["La pieza mide 500 mm y la cama util son 215 mm."],
    });
    renderBuilder();
    fireEvent.click(await screen.findByRole("button", { name: en["part.build"] }));

    expect(await screen.findByText(en["part.readyButDoesNotFit"])).toBeInTheDocument();
    expect(screen.getByRole("link", { name: en["part.download"] })).toBeInTheDocument();
  });

  it("surfaces a refusal from the server instead of staying silent", async () => {
    vi.mocked(printService.generatePart).mockRejectedValue(
      new Error("La pared quedaria de 0.25 mm"),
    );
    renderBuilder();
    fireEvent.click(await screen.findByRole("button", { name: en["part.build"] }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/0.25 mm/);
  });
});
