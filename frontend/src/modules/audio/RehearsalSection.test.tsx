import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CapabilityTreeResponse, SeparationModel } from "../../lib/apiTypes";
import * as capabilitiesService from "../../services/capabilities";
import { RehearsalSection } from "./RehearsalSection";
import { useRehearsalSelection } from "./useRehearsalSelection";
import { useTranscribeSelection } from "./useTranscribeSelection";

vi.mock("../../services/capabilities", () => ({
  fetchCapabilityTree: vi.fn(),
  provisionCapability: vi.fn(),
  provisionPack: vi.fn(),
  getProvisionStatus: vi.fn(),
}));

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

function transcriptionCapabilityTree(status: "available" | "needs_setup"): CapabilityTreeResponse {
  return {
    domains: [
      {
        domain: "audio",
        labelKey: "capability.domain.audio",
        capabilities: [
          {
            id: "audio.stemTranscription",
            domain: "audio",
            labelKey: "capability.audio.stemTranscription",
            status,
            provisioning: "vendored_pack",
            jobKind: "audio",
            strategies: ["model"],
            missingPacks: status === "needs_setup" ? ["music-transcription"] : [],
            unavailableReasonKey: null,
            setupReasonKey: status === "needs_setup" ? "capability.setup.missingPack" : null,
            activatableSettings: [],
          },
        ],
        roadmap: [],
      },
    ],
  };
}

// La selección vive en los hooks reales: el harness reproduce cómo AudioPanel
// conecta las tres piezas (modelo, minus-one, transcripción), que es lo que la
// sección necesita para reaccionar a los clicks.
function Harness({ model }: { model: SeparationModel | undefined }) {
  const selection = useRehearsalSelection(model);
  const transcribe = useTranscribeSelection(model);
  return <RehearsalSection model={model} selection={selection} transcribe={transcribe} />;
}

function renderHarness(model: SeparationModel | undefined) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<Harness model={model} />, { wrapper: Wrapper });
}

function openSection() {
  // ^ ancla al header del acordeon: el tooltip es otro boton "About Rehearsal…".
  fireEvent.click(screen.getByRole("button", { name: /^rehearsal/i }));
}

function activate() {
  openSection();
  fireEvent.click(screen.getByRole("checkbox", { name: /minus-one/i }));
}

function activateTranscribe() {
  openSection();
  fireEvent.click(screen.getByRole("checkbox", { name: /transcribe/i }));
}

beforeEach(() => {
  vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(
    transcriptionCapabilityTree("available"),
  );
});

describe("RehearsalSection", () => {
  it("does not exist for a two-stem model: minus a stem there is nothing left", () => {
    const { container } = renderHarness(KARAOKE);

    expect(container).toBeEmptyDOMElement();
  });

  it("does not exist while capabilities are still loading", () => {
    const { container } = renderHarness(undefined);

    expect(container).toBeEmptyDOMElement();
  });

  it("appears for a four-stem model with the opt-in toggle", () => {
    renderHarness(CUATRO);
    openSection();

    expect(screen.getByRole("checkbox", { name: /minus-one/i })).not.toBeChecked();
  });

  it("offers one chip per stem of the selected model once toggled on", () => {
    renderHarness(CUATRO);
    activate();

    expect(screen.getByRole("checkbox", { name: "Vocals" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Drums" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Bass" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Other" })).toBeInTheDocument();
  });

  it("keeps a ticked chip checked", () => {
    renderHarness(CUATRO);
    activate();

    fireEvent.click(screen.getByRole("checkbox", { name: "Drums" }));

    expect(screen.getByRole("checkbox", { name: "Drums" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Bass" })).not.toBeChecked();
  });

  it("starts with no guide and steps it through the fixed levels", () => {
    renderHarness(CUATRO);
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
    renderHarness(CUATRO);
    activate();

    expect(screen.getByText(/ghost/i)).toBeInTheDocument();
  });
});

describe("RehearsalSection transcribe subsection", () => {
  it("offers the transcribe toggle independently of minus-one", () => {
    renderHarness(CUATRO);
    openSection();

    expect(screen.getByRole("checkbox", { name: /transcribe/i })).not.toBeChecked();
  });

  it("offers a checkbox per stem with pitch once toggled on, excluding drums", () => {
    renderHarness(CUATRO);
    activateTranscribe();

    expect(screen.getByRole("checkbox", { name: "Vocals" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Bass" })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Other" })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: "Drums" })).not.toBeInTheDocument();
  });

  it("does not turn minus-one on as a side effect of turning transcribe on", () => {
    renderHarness(CUATRO);
    activateTranscribe();

    expect(screen.getByRole("checkbox", { name: /minus-one/i })).not.toBeChecked();
  });

  it("shows the honest draft-copy line once transcribe is active", () => {
    renderHarness(CUATRO);
    activateTranscribe();

    expect(screen.getByText(/editable draft/i)).toBeInTheDocument();
  });

  it("shows the pack download and disables the toggle when the transcription pack is missing", async () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(
      transcriptionCapabilityTree("needs_setup"),
    );
    renderHarness(CUATRO);
    openSection();

    expect(await screen.findByText(/music transcription pack/i)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /transcribe/i })).toBeDisabled();
    expect(screen.queryByRole("checkbox", { name: "Vocals" })).not.toBeInTheDocument();
  });
});
