import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import * as model3dService from "../../services/model3d";
import { ModelingPage } from "./ModelingPage";

vi.mock("../../services/model3d", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/model3d")>();
  return {
    ...actual,
    fetchModel3dCapabilities: vi.fn(),
    auditMesh: vi.fn(),
    splitSheetViews: vi.fn(),
    buildReferenceScene: vi.fn(),
  };
});

const CAPABILITIES: model3dService.Model3dCapabilities = {
  blender: {
    found: true,
    version: "4.3.0",
    meetsMinimum: true,
  },
  unlocked: ["audit", "referenceScene"],
  missing: null,
};

const SHEET_VIEWS: model3dService.SheetViews = {
  token: "sheet-token",
  views: [
    {
      name: "front",
      image: "/tmp/front.png",
      widthPx: 640,
      heightPx: 1200,
      inkBox: [0, 0, 640, 1200],
    },
    {
      name: "side",
      image: "/tmp/side.png",
      widthPx: 512,
      heightPx: 1180,
      inkBox: [0, 0, 512, 1180],
    },
  ],
  warnings: [],
};

const AUDIT_OK: model3dService.MeshAudit = {
  vertices: 800,
  faces: 600,
  tris: 200,
  quads: 500,
  ngons: 0,
  nonManifoldEdges: 0,
  boundaryEdges: 0,
  looseVerts: 0,
  shells: 1,
  dims: [1.2, 0.8, 1.7],
  thinnestAxisRatio: 0.47,
  hasUvs: true,
  blockers: [],
  warnings: [],
  ok: true,
};

const REFERENCE_SCENE: model3dService.ReferenceScene = {
  token: "scene-token",
  downloadUrl: "/api/v1/model3d/reference-scene/scene-token/download",
  heightMeters: 1.7,
  placed: [
    {
      view: "front",
      image: "/tmp/front.png",
      inkHeightMeters: 1.65,
      planeHeightMeters: 1.7,
      planeWidthMeters: 0.8,
      scaledByInk: true,
    },
    {
      view: "side",
      image: "/tmp/side.png",
      inkHeightMeters: 1.6,
      planeHeightMeters: 1.7,
      planeWidthMeters: 0.74,
      scaledByInk: false,
    },
  ],
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
  return render(<ModelingPage />, { wrapper: Wrapper });
}

function selectSheet(name = "turnaround.png") {
  const file = new File(["imagen"], name, { type: "image/png" });
  const input = document.getElementById("modeling-sheet-input") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

function selectMesh(name = "personaje.glb") {
  const file = new File(["binario"], name, { type: "model/gltf-binary" });
  const input = document.getElementById("modeling-mesh-input") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

beforeEach(() => {
  vi.mocked(model3dService.fetchModel3dCapabilities).mockResolvedValue(CAPABILITIES);
  vi.mocked(model3dService.splitSheetViews).mockResolvedValue(SHEET_VIEWS);
  vi.mocked(model3dService.buildReferenceScene).mockResolvedValue(REFERENCE_SCENE);
  vi.mocked(model3dService.auditMesh).mockResolvedValue(AUDIT_OK);
});

afterEach(() => {
  vi.mocked(model3dService.fetchModel3dCapabilities).mockReset();
  vi.mocked(model3dService.splitSheetViews).mockReset();
  vi.mocked(model3dService.buildReferenceScene).mockReset();
  vi.mocked(model3dService.auditMesh).mockReset();
});

describe("ModelingPage", () => {
  it("shows the capabilities loading state", () => {
    vi.mocked(model3dService.fetchModel3dCapabilities).mockReturnValue(new Promise(() => {}));

    renderPage();

    expect(screen.getByRole("status")).toHaveTextContent(en["capability.tree.loading"]);
  });

  it("shows an error when capabilities cannot be loaded", async () => {
    vi.mocked(model3dService.fetchModel3dCapabilities).mockRejectedValue(
      new Error("capabilities unavailable"),
    );

    renderPage();

    expect(await screen.findByRole("alert")).toHaveTextContent(en["capability.tree.loadFailed"]);
  });

  it("reports missing Blender and keeps splitting disabled after a file is chosen", async () => {
    const missing = "Blender 4.2 or newer was not found.";
    vi.mocked(model3dService.fetchModel3dCapabilities).mockResolvedValue({
      blender: {
        found: false,
        version: null,
        meetsMinimum: false,
      },
      unlocked: [],
      missing,
    });
    renderPage();

    expect(await screen.findByText(missing)).toBeInTheDocument();
    selectSheet();

    expect(screen.getByRole("button", { name: en["modeling.reference.split"] })).toBeDisabled();
  });

  it("shows the Blender version and enables both lanes after files are chosen", async () => {
    renderPage();

    expect(await screen.findByText("Blender 4.3.0 ready")).toBeInTheDocument();
    const splitButton = screen.getByRole("button", { name: en["modeling.reference.split"] });
    expect(splitButton).toBeDisabled();
    selectSheet();
    expect(splitButton).toBeEnabled();

    fireEvent.click(screen.getByRole("tab", { name: en["modeling.lane.audit"] }));
    const auditButton = screen.getByRole("button", { name: en["modeling.audit.run"] });
    expect(auditButton).toBeDisabled();
    selectMesh();
    expect(auditButton).toBeEnabled();
  });

  it("splits a reference sheet and lists view sizes and warnings", async () => {
    vi.mocked(model3dService.splitSheetViews).mockResolvedValue({
      ...SHEET_VIEWS,
      warnings: ["The side view is narrower than expected.", "The back view was not detected."],
    });
    renderPage();
    await screen.findByText("Blender 4.3.0 ready");
    const file = selectSheet();

    fireEvent.click(screen.getByRole("button", { name: en["modeling.reference.split"] }));

    await waitFor(() => expect(model3dService.splitSheetViews).toHaveBeenCalledWith(file, 4));
    expect(await screen.findByText("front")).toBeInTheDocument();
    expect(screen.getByText("640 × 1200 px")).toBeInTheDocument();
    expect(screen.getByText("side")).toBeInTheDocument();
    expect(screen.getByText("512 × 1180 px")).toBeInTheDocument();
    expect(screen.getByText("The side view is narrower than expected.")).toBeInTheDocument();
    expect(screen.getByText("The back view was not detected.")).toBeInTheDocument();
  });

  it("builds the reference scene at the default height and offers its download", async () => {
    renderPage();
    await screen.findByText("Blender 4.3.0 ready");
    selectSheet();
    fireEvent.click(screen.getByRole("button", { name: en["modeling.reference.split"] }));
    fireEvent.click(await screen.findByRole("button", { name: en["modeling.reference.build"] }));

    await waitFor(() =>
      expect(model3dService.buildReferenceScene).toHaveBeenCalledWith("sheet-token", 1.7),
    );
    const downloadLink = await screen.findByRole("link", { name: "Download the .blend (1.7 m)" });
    expect(downloadLink).toHaveAttribute(
      "href",
      "/api/v1/model3d/reference-scene/scene-token/download",
    );
    const sceneResult = within(downloadLink.parentElement as HTMLElement);
    expect(sceneResult.getByText("front")).toBeInTheDocument();
    expect(sceneResult.getByText(`1.65 · 1.7 × 0.8 · ${en["common.yes"]}`)).toBeInTheDocument();
    expect(sceneResult.getByText("side")).toBeInTheDocument();
    expect(sceneResult.getByText(`1.6 · 1.7 × 0.74 · ${en["common.no"]}`)).toBeInTheDocument();
  });

  it("does not build a scene while the real height is invalid", async () => {
    renderPage();
    await screen.findByText("Blender 4.3.0 ready");
    selectSheet();
    fireEvent.click(screen.getByRole("button", { name: en["modeling.reference.split"] }));
    const buildButton = await screen.findByRole("button", {
      name: en["modeling.reference.build"],
    });
    const heightInput = screen.getByRole("spinbutton", {
      name: en["modeling.reference.height"],
    });

    fireEvent.change(heightInput, { target: { value: "0" } });

    expect(buildButton).toBeDisabled();
    expect(
      screen.getByText(`${en["modeling.reference.height"]}: > 0`),
    ).toBeInTheDocument();
    fireEvent.click(buildButton);
    expect(model3dService.buildReferenceScene).not.toHaveBeenCalled();
  });

  it("shows blockers, warnings, and the blocked audit verdict", async () => {
    vi.mocked(model3dService.auditMesh).mockResolvedValue({
      ...AUDIT_OK,
      blockers: ["The mesh has 24 non-manifold edges."],
      warnings: ["The mesh has no UV map."],
      ok: false,
    });
    renderPage();
    await screen.findByText("Blender 4.3.0 ready");
    fireEvent.click(screen.getByRole("tab", { name: en["modeling.lane.audit"] }));
    const file = selectMesh();

    fireEvent.click(screen.getByRole("button", { name: en["modeling.audit.run"] }));

    await waitFor(() => expect(model3dService.auditMesh).toHaveBeenCalledWith(file));
    expect(await screen.findByText(en["modeling.audit.blocked"])).toBeInTheDocument();
    expect(screen.getByText("The mesh has 24 non-manifold edges.")).toBeInTheDocument();
    expect(screen.getByText("The mesh has no UV map.")).toBeInTheDocument();
  });

  it("shows the ok verdict for a passing audit", async () => {
    renderPage();
    await screen.findByText("Blender 4.3.0 ready");
    fireEvent.click(screen.getByRole("tab", { name: en["modeling.lane.audit"] }));
    selectMesh();
    fireEvent.click(screen.getByRole("button", { name: en["modeling.audit.run"] }));

    expect(await screen.findByText(en["modeling.audit.ok"])).toBeInTheDocument();
    expect(screen.queryByText(en["modeling.audit.blocked"])).not.toBeInTheDocument();
  });
});
