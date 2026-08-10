import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type { CleanupStep } from "../../lib/apiTypes";
import { CleanupChainPanel, cleanupSummaryKey } from "./CleanupChainPanel";
import { useCleanupSelection } from "./useCleanupSelection";

function step(
  id: string,
  family: string,
  covers: string[],
  installed = true,
): CleanupStep {
  return {
    id,
    name: `Model ${id}`,
    family,
    covers,
    installed,
    descriptionKey: `audio.karaoke.model.${id}.description`,
  };
}

// Mismo orden que CLEANUP_CHAIN del backend: ruido, eco, reverb.
const CATALOG: CleanupStep[] = [
  step("denoise", "denoise", ["denoise"]),
  step("deecho_normal", "deecho", ["deecho"]),
  step("deecho_aggressive", "deecho", ["deecho"], false),
  step("deecho_dereverb", "deecho", ["deecho", "dereverb"]),
  step("reverb_hq", "dereverb", ["dereverb"]),
];

function Harness({ catalog }: { catalog: CleanupStep[] }) {
  const selection = useCleanupSelection(catalog, 3);
  return <CleanupChainPanel steps={catalog} selection={selection} />;
}

function renderPanel(catalog: CleanupStep[] = CATALOG) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <LocaleProvider>
        <Harness catalog={catalog} />
      </LocaleProvider>
    </QueryClientProvider>,
  );
}

/** El primer checkbox es el opt-in maestro; los siguientes son los pasos. */
function activate() {
  fireEvent.click(
    screen.getByRole("checkbox", { name: new RegExp(en["audio.cleanup.activate"]) }),
  );
}

function stepBoxes(): HTMLInputElement[] {
  return (screen.getAllByRole("checkbox") as HTMLInputElement[]).slice(1);
}

function boxFor(id: string): HTMLInputElement {
  return stepBoxes()[CATALOG.findIndex((entry) => entry.id === id)];
}

function accessibleLabelOf(box: HTMLInputElement): string {
  return document.querySelector(`label[for="${box.id}"]`)?.textContent ?? "";
}

describe("CleanupChainPanel", () => {
  it("shows the passes in the order the backend sent", () => {
    // El orden es la información principal del panel: la cadena tiene
    // causalidad (ruido → eco → reverb) y presentarla desordenada la volvería
    // una lista de opciones sueltas.
    renderPanel();

    // El título de cada fila sale de `covers`, no de `family`: deecho_dereverb
    // es de la familia de-echo pero resuelve las dos, y llamarlo "quitar eco"
    // lo dejaría indistinguible del de-echo simple de la fila de arriba.
    const tasks = CATALOG.map(
      (entry) => en[`audio.cleanup.task.${entry.covers.join("_")}` as keyof typeof en],
    );
    expect(stepBoxes().map((box) => accessibleLabelOf(box))).toEqual(
      CATALOG.map((entry, index) => `${tasks[index]}${entry.name}`),
    );
  });

  it("names the model behind each pass, so the two intensities differ", () => {
    // Los dos De-Echo son el MISMO trabajo en dos intensidades: sin el nombre
    // del modelo las dos filas tendrían el mismo nombre accesible.
    renderPanel();

    const labels = stepBoxes().map((box) => accessibleLabelOf(box));
    expect(new Set(labels).size).toBe(CATALOG.length);
    for (const entry of CATALOG) {
      expect(labels.some((label) => label.includes(entry.name))).toBe(true);
    }
  });

  it("keeps every description visible without hovering", () => {
    renderPanel();

    for (const entry of CATALOG) {
      expect(
        screen.getByText(en[entry.descriptionKey as keyof typeof en]),
      ).toBeInTheDocument();
    }
  });

  it("locks the steps until the chain is switched on", () => {
    renderPanel();

    expect(boxFor("denoise")).toBeDisabled();

    activate();
    expect(boxFor("denoise")).not.toBeDisabled();
  });

  it("never lets a not-downloaded model be ticked, and offers the download", () => {
    // Dejar tildar un modelo que no está en disco sería ofrecer una selección
    // que se sabe que devuelve 400.
    renderPanel();
    activate();

    expect(boxFor("deecho_aggressive")).toBeDisabled();
    expect(boxFor("denoise")).not.toBeDisabled();
    // Solo el que falta trae tarjeta de descarga.
    expect(screen.getAllByText(en["pack.download"])).toHaveLength(1);
    expect(
      screen.getByText(/Model deecho_aggressive model is not downloaded yet/i),
    ).toBeInTheDocument();
  });

  it("says what a step would replace before it is ticked", () => {
    // Enterarse de que otro paso se apagó DESPUÉS de tildar es enterarse tarde.
    renderPanel();

    expect(
      screen.getAllByText(/turns off .*: they do the same job/i).length,
    ).toBeGreaterThan(0);
  });

  it("turns off the conflicting pass when another of its family is ticked", () => {
    renderPanel();
    activate();

    fireEvent.click(boxFor("deecho_normal"));
    expect(boxFor("deecho_normal")).toBeChecked();

    fireEvent.click(boxFor("deecho_dereverb"));
    expect(boxFor("deecho_dereverb")).toBeChecked();
    expect(boxFor("deecho_normal")).not.toBeChecked();
    // Cubre las dos familias: también apaga el de-reverb.
    expect(boxFor("reverb_hq")).not.toBeChecked();
  });

  it("warns from the third pass on, without blocking it", () => {
    renderPanel();
    activate();

    fireEvent.click(boxFor("denoise"));
    fireEvent.click(boxFor("deecho_normal"));
    expect(screen.queryByText(/over-processed/i)).not.toBeInTheDocument();

    fireEvent.click(boxFor("reverb_hq"));
    expect(screen.getByText(/3 chained passes/i)).toBeInTheDocument();
    // Avisar, no bloquear: los tres siguen tildados.
    expect(boxFor("denoise")).toBeChecked();
    expect(boxFor("deecho_normal")).toBeChecked();
    expect(boxFor("reverb_hq")).toBeChecked();
  });

  it("says the chain returns a single file and where to get the stems", () => {
    // Es la diferencia con el modo separación, y sin decirla las dos secciones
    // se leen como la misma cosa duplicada.
    renderPanel();

    expect(screen.getByText(en["audio.cleanup.singleOutputNote"])).toBeInTheDocument();
  });
});

describe("cleanupSummaryKey", () => {
  it("uses a different key per count, because translate does not pluralize", () => {
    expect(cleanupSummaryKey(0)).toBe("audio.cleanup.summary.none");
    expect(cleanupSummaryKey(1)).toBe("audio.cleanup.summary.one");
    expect(cleanupSummaryKey(3)).toBe("audio.cleanup.summary.many");
  });
});
