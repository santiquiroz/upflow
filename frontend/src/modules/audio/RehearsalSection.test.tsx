import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SeparationModel } from "../../lib/apiTypes";
import { RehearsalSection } from "./RehearsalSection";
import { useRehearsalSelection } from "./useRehearsalSelection";

function modelo(id: string, stems: string[]): SeparationModel {
  return {
    id,
    name: id,
    installed: true,
    primaryStem: stems[0],
    category: "karaoke",
    architecture: "mdx",
    descriptionKey: `audio.karaoke.model.${id}.description`,
    stems: stems.map((stem) => ({ id: stem, labelKey: `audio.stem.${stem}` })),
  } as SeparationModel;
}

const CUATRO = modelo("umx_4stem", ["vocals", "drums", "bass", "other"]);
const KARAOKE = modelo("inst_hq_3", ["instrumental", "vocals"]);

// La selección vive en el hook real: el harness reproduce cómo AudioPanel
// conecta ambos, que es lo que la sección necesita para reaccionar a los clicks.
function Harness({ model }: { model: SeparationModel | undefined }) {
  const selection = useRehearsalSelection(model);
  return <RehearsalSection model={model} selection={selection} />;
}

function openSection() {
  // ^ ancla al header del acordeon: el tooltip es otro boton "About Rehearsal…".
  fireEvent.click(screen.getByRole("button", { name: /^rehearsal/i }));
}

function activate() {
  openSection();
  fireEvent.click(screen.getByRole("checkbox", { name: /minus-one/i }));
}

describe("RehearsalSection", () => {
  it("does not exist for a two-stem model: minus a stem there is nothing left", () => {
    const { container } = render(<Harness model={KARAOKE} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("does not exist while capabilities are still loading", () => {
    const { container } = render(<Harness model={undefined} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("appears for a four-stem model with the opt-in toggle", () => {
    render(<Harness model={CUATRO} />);
    openSection();

    expect(screen.getByRole("checkbox", { name: /minus-one/i })).not.toBeChecked();
  });

  it("offers one chip per stem of the selected model once toggled on", () => {
    render(<Harness model={CUATRO} />);
    activate();

    expect(screen.getByRole("checkbox", { name: "Vocals" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Drums" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Bass" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Other" })).toBeInTheDocument();
  });

  it("keeps a ticked chip checked", () => {
    render(<Harness model={CUATRO} />);
    activate();

    fireEvent.click(screen.getByRole("checkbox", { name: "Drums" }));

    expect(screen.getByRole("checkbox", { name: "Drums" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Bass" })).not.toBeChecked();
  });

  it("starts with no guide and steps it through the fixed levels", () => {
    render(<Harness model={CUATRO} />);
    activate();

    expect(screen.getByRole("button", { name: /no guide/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    fireEvent.click(screen.getByRole("button", { name: "20%" }));

    expect(screen.getByRole("button", { name: "20%" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /no guide/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });

  it("says up front that removal is imperfect and weak stems leave a ghost", () => {
    render(<Harness model={CUATRO} />);
    activate();

    expect(screen.getByText(/ghost/i)).toBeInTheDocument();
  });
});
