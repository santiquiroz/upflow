import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { SeparationModel } from "../../lib/apiTypes";
import { compatibleModels, EnsembleSection } from "./EnsembleSection";

function modelo(
  id: string,
  name: string,
  stems: string[],
  installed = true,
): SeparationModel {
  return {
    id,
    name,
    installed,
    primaryStem: stems[0],
    category: "karaoke",
    architecture: "mdx",
    descriptionKey: `audio.karaoke.model.${id}.description`,
    stems: stems.map((stem) => ({ id: stem, labelKey: `audio.stem.${stem}` })),
  } as SeparationModel;
}

const KARAOKE = ["instrumental", "vocals"];
const CUATRO = ["vocals", "drums", "bass", "other"];

const CATALOGO = [
  modelo("a", "Modelo A", KARAOKE),
  modelo("b", "Modelo B", KARAOKE),
  modelo("c", "Modelo C", KARAOKE),
  modelo("cuatro", "Cuatro pistas", CUATRO),
  modelo("d", "Modelo D", KARAOKE),
  modelo("sin-bajar", "Sin bajar", KARAOKE, false),
];

function renderSection(chosen: string[] = [], onChange = vi.fn(), selected = "a") {
  render(
    <EnsembleSection
      models={CATALOGO}
      selectedModel={selected}
      chosen={chosen}
      onChange={onChange}
    />,
  );
  return onChange;
}

describe("compatibleModels", () => {
  it("solo combina modelos que entregan las MISMAS pistas", () => {
    // Promediar un instrumental con un bajo no da un instrumental mejor: da una
    // suma que además nadie puede etiquetar.
    const ids = compatibleModels(CATALOGO, "a").map((model) => model.id);

    expect(ids).toContain("b");
    expect(ids).not.toContain("cuatro");
  });

  it("no se ofrece a sí mismo", () => {
    expect(compatibleModels(CATALOGO, "a").map((model) => model.id)).not.toContain("a");
  });

  it("no ofrece lo que no está bajado", () => {
    expect(compatibleModels(CATALOGO, "a").map((model) => model.id)).not.toContain(
      "sin-bajar",
    );
  });

  it("sin modelo elegido no hay con qué combinar", () => {
    expect(compatibleModels(CATALOGO, null)).toEqual([]);
  });
});

describe("EnsembleSection", () => {
  it("no aparece cuando el modelo elegido no tiene compañero posible", () => {
    // El de cuatro pistas es el único de su forma: ofrecer la sección vacía
    // sería un control que no puede usarse.
    const { container } = render(
      <EnsembleSection
        models={CATALOGO}
        selectedModel="cuatro"
        chosen={[]}
        onChange={vi.fn()}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("suma un modelo al tildarlo", () => {
    const onChange = renderSection([]);

    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo B" }));

    expect(onChange).toHaveBeenCalledWith(["b"]);
  });

  it("destildar lo saca", () => {
    const onChange = renderSection(["b"]);

    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo B" }));

    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("no deja pasar el techo de tres contando el principal", () => {
    const onChange = renderSection(["b", "c"]);

    fireEvent.click(screen.getByRole("checkbox", { name: "Modelo D" }));

    // Cada modelo más es una pasada completa sobre el tema; el cuarto no se oye.
    // Y se IGNORA en vez de reemplazar: cambiar en silencio una elección que el
    // usuario no tocó es peor que no hacer nada visible.
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole("checkbox", { name: "Modelo B" })).toBeChecked();
  });

  it("dice el costo antes y no después", () => {
    renderSection(["b"]);

    // Son N pasadas completas sobre el tema, no un ajuste que sale gratis.
    expect(screen.getByRole("status")).toHaveTextContent("2");
  });

  it("sin nada elegido no avisa de ningún costo", () => {
    renderSection([]);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
