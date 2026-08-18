import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SeparationModel } from "../../lib/apiTypes";
import { jobQueueStore } from "../../lib/jobQueueStore";
import * as audioService from "../../services/audio";
import { ModelComparison } from "./ModelComparison";

vi.mock("../../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/audio")>();
  return { ...actual, compareSeparationModels: vi.fn() };
});

function modelo(id: string, name: string, installed = true): SeparationModel {
  return {
    id,
    name,
    installed,
    primaryStem: "Instrumental",
    category: "karaoke",
    architecture: "mdx",
    descriptionKey: `audio.karaoke.model.${id}.description`,
    stems: [
      { id: "instrumental", labelKey: "audio.stem.instrumental" },
      { id: "vocals", labelKey: "audio.stem.vocals" },
    ],
  } as SeparationModel;
}

const TRES = [modelo("a", "Modelo A"), modelo("b", "Modelo B"), modelo("c", "Modelo C")];

function renderComparison(models: SeparationModel[], file: File | null = new File(["x"], "tema.mp3")) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<ModelComparison file={file} models={models} />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(audioService.compareSeparationModels).mockReset();
  jobQueueStore.getSnapshot().forEach((job) => jobQueueStore.removeTrackedJob(job.id));
});

describe("ModelComparison", () => {
  it("no se ofrece cuando hay un solo modelo instalado", () => {
    // Comparar contra nada: el control existiria para no poder usarse.
    const { container } = renderComparison([modelo("a", "Modelo A"), modelo("b", "B", false)]);

    expect(container).toBeEmptyDOMElement();
  });

  it("no deja comparar hasta que hay dos elegidos", () => {
    renderComparison(TRES);
    const boton = screen.getByRole("button", { name: /Compare/i });

    expect(boton).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo A" }));
    expect(boton).toBeDisabled();

    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo B" }));
    expect(boton).toBeEnabled();
  });

  it("sin archivo elegido no hay nada que comparar", () => {
    renderComparison(TRES, null);

    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo A" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo B" }));

    expect(screen.getByRole("button", { name: /Compare/i })).toBeDisabled();
  });

  it("manda los modelos elegidos con el archivo", async () => {
    vi.mocked(audioService.compareSeparationModels).mockResolvedValue({
      entries: [
        { modelId: "a", jobId: "j-a" },
        { modelId: "b", jobId: "j-b" },
      ],
      offsetSeconds: 105,
      excerptSeconds: 30,
    });
    renderComparison(TRES);
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo A" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo B" }));

    fireEvent.click(screen.getByRole("button", { name: /Compare/i }));

    await waitFor(() =>
      expect(audioService.compareSeparationModels).toHaveBeenCalledWith(
        expect.objectContaining({ models: ["a", "b"] }),
      ),
    );
  });

  it("cada prueba entra a la cola con el nombre del modelo que la hizo", async () => {
    vi.mocked(audioService.compareSeparationModels).mockResolvedValue({
      entries: [
        { modelId: "a", jobId: "j-a" },
        { modelId: "b", jobId: "j-b" },
      ],
      offsetSeconds: 105,
      excerptSeconds: 30,
    });
    renderComparison(TRES);
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo A" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo B" }));

    fireEvent.click(screen.getByRole("button", { name: /Compare/i }));

    await waitFor(() => expect(jobQueueStore.getSnapshot()).toHaveLength(2));
    // Dos entradas iguales en la cola no se pueden comparar: hay que saber cual
    // es cual sin abrir el detalle.
    const nombres = jobQueueStore.getSnapshot().map((job) => job.fileName);
    expect(nombres).toContain("tema.mp3 [Modelo A]");
    expect(nombres).toContain("tema.mp3 [Modelo B]");
  });

  it("dice qué parte del tema se está oyendo", async () => {
    vi.mocked(audioService.compareSeparationModels).mockResolvedValue({
      entries: [{ modelId: "a", jobId: "j-a" }],
      offsetSeconds: 105,
      excerptSeconds: 30,
    });
    renderComparison(TRES);
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo A" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo B" }));

    fireEvent.click(screen.getByRole("button", { name: /Compare/i }));

    // El arranque lo elige el servidor (el medio del tema): sin decirlo, el
    // resultado es un misterio.
    expect(await screen.findByText(/1:45/)).toBeInTheDocument();
  });

  it("no deja elegir un cuarto modelo", () => {
    renderComparison([...TRES, modelo("d", "Modelo D")]);

    ["Modelo A", "Modelo B", "Modelo C", "Modelo D"].forEach((name) => {
      fireEvent.click(screen.getByRole("checkbox", { name }));
    });

    // Más de tres deja de ser comparación y pasa a ser cola: se espera lo mismo
    // que correr el tema entero, que es justo lo que se evita.
    expect(screen.getByRole("checkbox", { name: "Modelo D" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Modelo C" })).toBeChecked();
  });

  it("destildar libera el cupo", () => {
    renderComparison([...TRES, modelo("d", "Modelo D")]);
    ["Modelo A", "Modelo B", "Modelo C"].forEach((name) => {
      fireEvent.click(screen.getByRole("checkbox", { name }));
    });

    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo A" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo D" }));

    expect(screen.getByRole("checkbox", { name: "Modelo D" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Modelo A" })).not.toBeChecked();
  });
});
