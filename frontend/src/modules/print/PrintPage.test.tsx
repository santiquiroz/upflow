import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import * as printService from "../../services/print";
import { PrintPage } from "./PrintPage";

vi.mock("../../services/print", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/print")>();
  return { ...actual, fetchPrinters: vi.fn(), checkPrint: vi.fn() };
});

const RESULTADO_OK: printService.PrintCheckResult = {
  canPrint: true,
  sizeMm: [40, 30, 20],
  triangleCount: 1200,
  watertight: true,
  manifold: true,
  volumeMm3: 12000,
  overhangRatio: 0.12,
  blockers: [],
  advice: [],
};

function renderPage() {
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
  return render(<PrintPage />, { wrapper: Wrapper });
}

function selectFile(name = "pieza.stl") {
  const input = document.getElementById("print-file-input") as HTMLInputElement;
  fireEvent.change(input, {
    target: { files: [new File(["binario"], name, { type: "model/stl" })] },
  });
}

beforeEach(() => {
  vi.mocked(printService.fetchPrinters).mockResolvedValue({
    printers: [
      { id: "ender-3", bedMm: [220, 220, 250] },
      { id: "k1-max", bedMm: [300, 300, 300] },
    ],
  });
  vi.mocked(printService.checkPrint).mockResolvedValue(RESULTADO_OK);
});

afterEach(() => {
  vi.mocked(printService.fetchPrinters).mockReset();
  vi.mocked(printService.checkPrint).mockReset();
});

describe("PrintPage", () => {
  it("cannot check before a file is chosen", async () => {
    renderPage();

    expect(await screen.findByRole("button", { name: en["print.check"] })).toBeDisabled();
  });

  it("sends the file and the chosen printer", async () => {
    renderPage();
    await screen.findByRole("combobox", { name: /printer/i });
    selectFile();
    fireEvent.change(screen.getByRole("combobox", { name: /printer/i }), {
      target: { value: "k1-max" },
    });
    fireEvent.click(screen.getByRole("button", { name: en["print.check"] }));

    await waitFor(() => expect(printService.checkPrint).toHaveBeenCalled());
    expect(vi.mocked(printService.checkPrint).mock.calls[0][0]).toEqual(
      expect.objectContaining({ printer: "k1-max" }),
    );
  });

  it("shows the verdict and the measurements", async () => {
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));

    expect(await screen.findByText(en["print.verdict.yes"])).toBeInTheDocument();
    expect(screen.getByText("40.0 × 30.0 × 20.0 mm")).toBeInTheDocument();
    expect(screen.getByText("12%")).toBeInTheDocument();
  });

  it("lists what stops a part that does not print", async () => {
    vi.mocked(printService.checkPrint).mockResolvedValue({
      ...RESULTADO_OK,
      canPrint: false,
      watertight: false,
      blockers: ["12 aristas de borde: la malla no es estanca."],
    });
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));

    expect(await screen.findByText(en["print.verdict.no"])).toBeInTheDocument();
    expect(screen.getByText(/aristas de borde/)).toBeInTheDocument();
  });

  it("keeps advice separate from blockers", async () => {
    // Mezclarlos entrena a ignorar los dos.
    vi.mocked(printService.checkPrint).mockResolvedValue({
      ...RESULTADO_OK,
      advice: ["Conviene girar la pieza 90° en X."],
    });
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));

    expect(await screen.findByText(/Conviene girar/)).toBeInTheDocument();
    expect(screen.queryByText(en["print.blockers"])).not.toBeInTheDocument();
  });

  it("sends the target size only when both axis and millimetres are set", async () => {
    // Un eje sin medida no dice a cuánto escalar, y una medida sin eje tampoco.
    renderPage();
    selectFile();
    fireEvent.change(screen.getByRole("spinbutton", { name: en["print.targetMm"] }), {
      target: { value: "150" },
    });
    fireEvent.click(screen.getByRole("button", { name: en["print.check"] }));

    await waitFor(() => expect(printService.checkPrint).toHaveBeenCalled());
    expect(vi.mocked(printService.checkPrint).mock.calls[0][0].targetMm).toBeUndefined();
  });

  it("sends the target size when both are set", async () => {
    renderPage();
    selectFile();
    fireEvent.change(screen.getByRole("combobox", { name: en["print.targetAxis"] }), {
      target: { value: "x" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: en["print.targetMm"] }), {
      target: { value: "150" },
    });
    fireEvent.click(screen.getByRole("button", { name: en["print.check"] }));

    await waitFor(() => expect(printService.checkPrint).toHaveBeenCalled());
    expect(vi.mocked(printService.checkPrint).mock.calls[0][0]).toEqual(
      expect.objectContaining({ targetAxis: "x", targetMm: 150 }),
    );
  });

  it("surfaces a failure instead of staying silent", async () => {
    vi.mocked(printService.checkPrint).mockRejectedValue(new Error("El archivo no es un STL"));
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/no es un STL/);
  });
});
