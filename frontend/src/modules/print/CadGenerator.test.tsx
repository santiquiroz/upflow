import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import * as printService from "../../services/print";
import { CadGenerator } from "./CadGenerator";

// ---------------------------------------------------------------------------
// Este carril existe por UNA cosa: las cotas. Si la pantalla no deja pedir una
// medida y no muestra la que salio, no se distingue del generador de malla y no
// vale la pena tenerlo aparte.
// ---------------------------------------------------------------------------

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
  id: "c1",
  status: "queued",
  prompt: "un espaciador",
  printer: "ender-3",
  source: "cad",
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
  sizeMm: [20, 20, 12],
  triangleCount: 512,
  code: "difference() {\n  cylinder(d=20, h=12);\n}",
  downloadUrl: "/api/v1/print/generate/c1/download",
};

function renderCad() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <LocaleProvider>{children}</LocaleProvider>
      </QueryClientProvider>
    );
  }
  return render(<CadGenerator printer="ender-3" />, { wrapper: Wrapper });
}

function pedir(texto = "un espaciador de 20 mm") {
  fireEvent.change(screen.getByLabelText(/qué es|what/i), {
    target: { value: texto },
  });
}

beforeEach(() => {
  vi.mocked(printService.createShape3dJob).mockResolvedValue(EN_COLA);
  vi.mocked(printService.getShape3dJob).mockResolvedValue(LISTA);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("las cotas viajan al backend", () => {
  it("manda el carril cad, no el de malla", async () => {
    renderCad();
    pedir();

    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    await waitFor(() => {
      expect(printService.createShape3dJob).toHaveBeenCalledWith(
        expect.objectContaining({ source: "cad" }),
      );
    });
  });

  it("manda las tres medidas cuando estan las tres", async () => {
    renderCad();
    pedir();
    fireEvent.change(screen.getByLabelText(/largo|length/i), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText(/ancho|width/i), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText(/alto|height/i), { target: { value: "12" } });

    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    await waitFor(() => {
      expect(printService.createShape3dJob).toHaveBeenCalledWith(
        expect.objectContaining({ expectedSize: [20, 20, 12] }),
      );
    });
  });

  it("no manda medidas a medio llenar", async () => {
    // Dos de tres no es una cota: mandarla haria que el bucle compare contra
    // un cero inventado y reintente tres veces por nada.
    renderCad();
    pedir();
    fireEvent.change(screen.getByLabelText(/largo|length/i), { target: { value: "20" } });

    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    await waitFor(() => {
      expect(printService.createShape3dJob).toHaveBeenCalledWith(
        expect.objectContaining({ expectedSize: undefined }),
      );
    });
  });
});

describe("lo que muestra el resultado", () => {
  it("muestra lo que MIDIO la pieza", async () => {
    renderCad();
    pedir();
    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    expect(await screen.findByText(/20\.0 × 20\.0 × 12\.0 mm/)).toBeInTheDocument();
  });

  it("entrega el codigo, que es la pieza editable", async () => {
    // El STL no se puede ajustar. El .scad si: cambiar un numero y volver a
    // compilar es la unica forma de corregir una cota sin volver a pedirla.
    renderCad();
    pedir();
    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    expect(await screen.findByText(/cylinder\(d=20, h=12\)/)).toBeInTheDocument();
  });

  it("avisa cuando el modelo tuvo que corregirse", async () => {
    vi.mocked(printService.getShape3dJob).mockResolvedValue({ ...LISTA, retries: 2 });
    renderCad();
    pedir();
    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    expect(
      await screen.findByText(/corrigió sola|fixed itself/i),
    ).toHaveTextContent("2");
  });

  it("no habla de correcciones cuando salio a la primera", async () => {
    // Decir "reintentos: 0" es ruido: solo importa cuando paso algo.
    renderCad();
    pedir();
    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    await screen.findByText(/20\.0 × 20\.0 × 12\.0 mm/);
    expect(screen.queryByText(/corrigió sola|fixed itself/i)).toBeNull();
  });

  it("un fallo del modelo se lee, no desaparece", async () => {
    vi.mocked(printService.getShape3dJob).mockResolvedValue({
      ...EN_COLA,
      status: "failed",
      error: "El modelo no logro escribir codigo que compile en 3 intentos.",
    });
    renderCad();
    pedir();
    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/3 intentos/);
  });
});

describe("guardas", () => {
  it("sin descripcion no se puede pedir nada", () => {
    renderCad();
    expect(screen.getByRole("button", { name: /generar|generate/i })).toBeDisabled();
  });

  it("un error del servidor se muestra en vez de tragarse", async () => {
    vi.mocked(printService.createShape3dJob).mockRejectedValue(
      new Error("Hace falta un servidor de modelo local"),
    );
    renderCad();
    pedir();

    fireEvent.click(screen.getByRole("button", { name: /generar|generate/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/servidor de modelo/);
  });
});
