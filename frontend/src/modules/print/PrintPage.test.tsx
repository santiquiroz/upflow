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
  return {
    ...actual,
    fetchPrinters: vi.fn(),
    checkPrint: vi.fn(),
    repairMesh: vi.fn(),
    createShape3dJob: vi.fn(),
    getShape3dJob: vi.fn(),
  };
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
  vi.mocked(printService.repairMesh).mockReset();
  vi.mocked(printService.createShape3dJob).mockReset();
  vi.mocked(printService.getShape3dJob).mockReset();
  vi.unstubAllGlobals();
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

  it("offers to close a mesh that is not watertight", async () => {
    vi.mocked(printService.checkPrint).mockResolvedValue({
      ...RESULTADO_OK,
      canPrint: false,
      watertight: false,
      blockers: ["12 aristas de borde."],
    });
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));

    expect(await screen.findByRole("button", { name: en["print.repair"] })).toBeInTheDocument();
  });

  it("does not offer to close a mesh that is already watertight", async () => {
    // Un boton que no hace nada util entrena a ignorar los botones.
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));

    await screen.findByText(en["print.verdict.yes"]);
    expect(screen.queryByRole("button", { name: en["print.repair"] })).not.toBeInTheDocument();
  });

  it("says plainly when the repair did not close it, and offers the file anyway", async () => {
    vi.mocked(printService.checkPrint).mockResolvedValue({
      ...RESULTADO_OK,
      canPrint: false,
      watertight: false,
      blockers: ["muchas aristas de borde."],
    });
    vi.mocked(printService.repairMesh).mockResolvedValue({
      canPrint: false,
      watertight: false,
      manifold: true,
      triangleCount: 900,
      volumeMm3: null,
      blockers: ["Siguen quedando 40 aristas de borde."],
      downloadUrl: "/api/v1/print/repaired/abc",
    });
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));
    fireEvent.click(await screen.findByRole("button", { name: en["print.repair"] }));

    expect(await screen.findByText(en["print.repair.stillOpen"])).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: en["print.repair.download"] }),
    ).toHaveAttribute("href", "/api/v1/print/repaired/abc");
  });

  it("warns that the caps are flat when it did close", async () => {
    // "Reparada" a secas haria creer que volvio la forma que faltaba.
    vi.mocked(printService.checkPrint).mockResolvedValue({
      ...RESULTADO_OK,
      canPrint: false,
      watertight: false,
      blockers: ["3 aristas de borde."],
    });
    vi.mocked(printService.repairMesh).mockResolvedValue({
      canPrint: true,
      watertight: true,
      manifold: true,
      triangleCount: 1203,
      volumeMm3: 12000,
      blockers: [],
      downloadUrl: "/api/v1/print/repaired/def",
    });
    renderPage();
    selectFile();
    fireEvent.click(await screen.findByRole("button", { name: en["print.check"] }));
    fireEvent.click(await screen.findByRole("button", { name: en["print.repair"] }));

    expect(await screen.findByText(en["print.repair.closed"])).toBeInTheDocument();
  });

  it("hands a generated mesh with problems straight to the repair flow", async () => {
    // El caso real: la palanca de freno salio con aristas de borde y hubo que
    // bajarla y re-subirla a mano. Este boton hace ese viaje por el usuario.
    const generado: printService.Shape3dJob = {
      id: "j9",
      status: "completed",
      prompt: "una palanca de freno",
      printer: "ender-3",
      source: "mesh",
      code: null,
      retries: 0,
      targetMm: 100,
      targetMmSource: "default",
      targetMmReference: null,
      createdAt: "2026-08-05T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      canPrint: false,
      sizeMm: [80, 32, 12],
      triangleCount: 40210,
      blockers: ["3 aristas de borde: la malla no es estanca."],
      advice: [],
      error: null,
      downloadUrl: "/api/v1/print/generate/j9/download",
    };
    vi.mocked(printService.createShape3dJob).mockResolvedValue(generado);
    vi.mocked(printService.getShape3dJob).mockResolvedValue(generado);
    vi.mocked(printService.repairMesh).mockResolvedValue({
      canPrint: true,
      watertight: true,
      manifold: true,
      triangleCount: 40213,
      volumeMm3: 9000,
      blockers: [],
      downloadUrl: "/api/v1/print/repaired/j9",
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(new Blob(["solid palanca"], { type: "model/stl" })),
      }),
    );

    renderPage();
    fireEvent.click(await screen.findByRole("tab", { name: en["print.lane.generate"] }));
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "una palanca de freno" },
    });
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));
    fireEvent.click(await screen.findByRole("button", { name: en["gen3d.repair"] }));

    await waitFor(() => expect(printService.repairMesh).toHaveBeenCalledTimes(1));
    const archivo = vi.mocked(printService.repairMesh).mock.calls[0][0];
    expect(archivo.name).toBe("mesh-j9.stl");

    // Volvio al carril de chequeo con el archivo ya cargado y la reparacion hecha.
    expect(
      screen.getByRole("tab", { name: en["print.lane.check"] }),
    ).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByText(en["print.repair.closed"])).toBeInTheDocument();
    expect(screen.getByText("mesh-j9.stl")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: en["print.repair.download"] }),
    ).toHaveAttribute("href", "/api/v1/print/repaired/j9");
  });
});
