import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import * as printService from "../../services/print";
import { MeshGenerator } from "./MeshGenerator";

vi.mock("../../services/print", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/print")>();
  return {
    ...actual,
    createShape3dJob: vi.fn(),
    getShape3dJob: vi.fn(),
    cancelShape3dJob: vi.fn(),
  };
});

const EN_COLA: printService.Shape3dJob = {
  id: "j1",
  status: "queued",
  prompt: "un soporte",
  printer: "ender-3",
  source: "mesh",
  code: null,
  retries: 0,
  createdAt: "2026-08-05T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  canPrint: null,
  sizeMm: null,
  triangleCount: null,
  blockers: [],
  advice: [],
  error: null,
  downloadUrl: null,
};

const LISTA: printService.Shape3dJob = {
  ...EN_COLA,
  status: "completed",
  canPrint: true,
  sizeMm: [80, 32, 92],
  triangleCount: 67384,
  downloadUrl: "/api/v1/print/generate/j1/download",
};

const CON_PROBLEMAS: printService.Shape3dJob = {
  ...LISTA,
  canPrint: false,
  blockers: ["3 aristas de borde: la malla no es estanca."],
};

function renderGen(onRepairFile: (file: File) => void = vi.fn()) {
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
  return render(<MeshGenerator printer="ender-3" onRepairFile={onRepairFile} />, {
    wrapper: Wrapper,
  });
}

function generar() {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "algo" } });
  fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));
}

beforeEach(() => {
  vi.mocked(printService.createShape3dJob).mockResolvedValue(EN_COLA);
  vi.mocked(printService.getShape3dJob).mockResolvedValue(LISTA);
});

afterEach(() => {
  vi.mocked(printService.createShape3dJob).mockReset();
  vi.mocked(printService.getShape3dJob).mockReset();
  vi.unstubAllGlobals();
});

describe("MeshGenerator", () => {
  it("warns that this gives a shape and not measurements, before anything else", () => {
    // La advertencia decide para que sirve la pieza: no puede estar escondida.
    renderGen();

    expect(screen.getByText(en["gen3d.noDimensions"])).toBeInTheDocument();
  });

  it("cannot generate without a description", () => {
    renderGen();

    expect(screen.getByRole("button", { name: en["gen3d.generate"] })).toBeDisabled();
  });

  it("sends the description and the printer", async () => {
    renderGen();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "un soporte en L" } });
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    await waitFor(() => expect(printService.createShape3dJob).toHaveBeenCalled());
    expect(vi.mocked(printService.createShape3dJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ prompt: "un soporte en L", printer: "ender-3" }),
    );
  });

  it("sends the target size only when it is a real number", async () => {
    renderGen();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "algo" } });
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    await waitFor(() => expect(printService.createShape3dJob).toHaveBeenCalled());
    expect(vi.mocked(printService.createShape3dJob).mock.calls[0][0].targetMm).toBeUndefined();
  });

  it("shows the verdict and the measurements when it finishes", async () => {
    renderGen();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "algo" } });
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    expect(await screen.findByText(en["gen3d.ready"])).toBeInTheDocument();
    expect(screen.getByText(/80.0 × 32.0 × 92.0 mm/)).toBeInTheDocument();
  });

  it("says plainly when what it generated does not print", async () => {
    vi.mocked(printService.getShape3dJob).mockResolvedValue({
      ...LISTA,
      canPrint: false,
      blockers: ["6 aristas de borde: la malla no es estanca."],
    });
    renderGen();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "algo" } });
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    expect(await screen.findByText(en["gen3d.readyButNotPrintable"])).toBeInTheDocument();
    expect(screen.getByText(/aristas de borde/)).toBeInTheDocument();
  });

  it("surfaces a refusal from the server instead of staying silent", async () => {
    vi.mocked(printService.createShape3dJob).mockRejectedValue(
      new Error("El modelo 3D no esta instalado"),
    );
    renderGen();
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "algo" } });
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/no esta instalado/);
  });

  it("offers to repair a generated mesh that does not print", async () => {
    vi.mocked(printService.getShape3dJob).mockResolvedValue(CON_PROBLEMAS);
    renderGen();
    generar();

    expect(await screen.findByRole("button", { name: en["gen3d.repair"] })).toBeInTheDocument();
  });

  it("does not offer to repair a mesh that already prints", async () => {
    // Un boton que no hace nada util entrena a ignorar los botones.
    renderGen();
    generar();

    await screen.findByText(en["gen3d.ready"]);
    expect(screen.queryByRole("button", { name: en["gen3d.repair"] })).not.toBeInTheDocument();
  });

  it("downloads the generated STL and hands it to the repair flow", async () => {
    vi.mocked(printService.getShape3dJob).mockResolvedValue(CON_PROBLEMAS);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(["solid palanca"], { type: "model/stl" })),
    });
    vi.stubGlobal("fetch", fetchMock);
    const onRepairFile = vi.fn();
    renderGen(onRepairFile);
    generar();
    fireEvent.click(await screen.findByRole("button", { name: en["gen3d.repair"] }));

    await waitFor(() => expect(onRepairFile).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledWith(CON_PROBLEMAS.downloadUrl);
    const archivo = onRepairFile.mock.calls[0][0] as File;
    expect(archivo.name).toBe("mesh-j1.stl");
    expect(archivo.type).toBe("model/stl");
    // jsdom no implementa File.text(); el tamano alcanza para probar que el
    // blob bajado es el contenido del archivo.
    expect(archivo.size).toBe("solid palanca".length);
  });

  it("says when the STL could not be downloaded for repair", async () => {
    vi.mocked(printService.getShape3dJob).mockResolvedValue(CON_PROBLEMAS);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    const onRepairFile = vi.fn();
    renderGen(onRepairFile);
    generar();
    fireEvent.click(await screen.findByRole("button", { name: en["gen3d.repair"] }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      en["gen3d.repair.downloadFailed"],
    );
    expect(onRepairFile).not.toHaveBeenCalled();
  });
});
