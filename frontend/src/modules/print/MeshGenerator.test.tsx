import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import type { CapabilityTreeResponse } from "../../lib/apiTypes";
import * as capabilitiesService from "../../services/capabilities";
import * as generationService from "../../services/generation";
import * as printService from "../../services/print";
import { MeshGenerator } from "./MeshGenerator";

vi.mock("../../services/print", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/print")>();
  return {
    ...actual,
    createShape3dJob: vi.fn(),
    getShape3dJob: vi.fn(),
    cancelShape3dJob: vi.fn(),
    estimatePrintSize: vi.fn(),
  };
});

vi.mock("../../services/capabilities", () => ({
  fetchCapabilityTree: vi.fn(),
  provisionCapability: vi.fn(),
  provisionPack: vi.fn(),
  getProvisionStatus: vi.fn(),
}));

vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return {
    ...actual,
    uploadGenerationInitImage: vi.fn(),
  };
});

function photoCapabilityTree(
  status: "available" | "needs_setup",
  // La estimacion de tamano depende de OTRA cosa que el pack de foto: del
  // servidor de modelo configurado. Por eso se controla aparte.
  estimateStatus: "available" | "needs_setup" = "available",
): CapabilityTreeResponse {
  return {
    domains: [
      {
        domain: "print",
        labelKey: "capability.domain.print",
        capabilities: [
          {
            id: "print.generatePhoto",
            domain: "print",
            labelKey: "capability.print.generatePhoto",
            status,
            provisioning: "vendored_pack",
            jobKind: "print",
            strategies: ["model"],
            missingPacks: status === "needs_setup" ? ["shap-e-img2img"] : [],
            unavailableReasonKey: null,
            setupReasonKey: status === "needs_setup" ? "capability.setup.missingPack" : null,
            activatableSettings: [],
          },
          {
            id: "print.estimateSize",
            domain: "print",
            labelKey: "capability.print.estimateSize",
            status: estimateStatus,
            provisioning: "user_supplied",
            jobKind: null,
            strategies: ["model"],
            missingPacks: [],
            unavailableReasonKey: null,
            setupReasonKey:
              estimateStatus === "needs_setup" ? "capability.setup.missingSetting" : null,
            activatableSettings: [],
          },
        ],
        roadmap: [],
      },
    ],
  };
}

const EN_COLA: printService.Shape3dJob = {
  id: "j1",
  status: "queued",
  prompt: "un soporte",
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
  vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(
    photoCapabilityTree("available"),
  );
  vi.mocked(printService.estimatePrintSize).mockResolvedValue({
    longestMm: 95,
    reference: "standard coffee mug",
  });
  vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValue({
    initImageToken: "abc123",
    originalFilename: "foto.png",
    width: 8,
    height: 8,
  });
  // jsdom no trae createObjectURL; el preview lo necesita.
  URL.createObjectURL = vi.fn(() => "blob:preview") as never;
  URL.revokeObjectURL = vi.fn() as never;
});

afterEach(() => {
  vi.mocked(printService.createShape3dJob).mockReset();
  vi.mocked(printService.getShape3dJob).mockReset();
  vi.mocked(printService.estimatePrintSize).mockReset();
  vi.mocked(capabilitiesService.fetchCapabilityTree).mockReset();
  vi.mocked(generationService.uploadGenerationInitImage).mockReset();
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

// ---------------------------------------------------------------------------
// El modo foto: sube la imagen por el mismo staging que la generación de
// imágenes y manda el token; el copy dice honestamente que es una
// interpretación del objeto, no una réplica.
// ---------------------------------------------------------------------------

function irAFoto() {
  fireEvent.click(screen.getByRole("tab", { name: en["gen3d.mode.photo"] }));
}

function elegirFoto(file: File = new File(["png"], "taza.png", { type: "image/png" })) {
  fireEvent.change(screen.getByLabelText(en["gen3d.photo.choose"]), {
    target: { files: [file] },
  });
  return file;
}

describe("MeshGenerator photo mode", () => {
  it("shows the honest disclaimer before anything else", () => {
    renderGen();
    irAFoto();

    expect(screen.getByText(en["gen3d.photo.disclaimer"])).toBeInTheDocument();
    // La advertencia de cotas sigue aplicando: img2img tampoco da medidas.
    expect(screen.getByText(en["gen3d.noDimensions"])).toBeInTheDocument();
  });

  it("cannot generate until a photo is chosen", async () => {
    renderGen();
    irAFoto();

    await screen.findByLabelText(en["gen3d.photo.choose"]);
    expect(screen.getByRole("button", { name: en["gen3d.generate"] })).toBeDisabled();
  });

  it("uploads the photo and sends its token as a photo job", async () => {
    renderGen();
    irAFoto();
    await screen.findByLabelText(en["gen3d.photo.choose"]);
    const foto = elegirFoto();
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    await waitFor(() => expect(printService.createShape3dJob).toHaveBeenCalled());
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledWith(foto);
    expect(vi.mocked(printService.createShape3dJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({
        source: "photo",
        imageToken: "abc123",
        printer: "ender-3",
      }),
    );
  });

  it("shows a preview of the chosen photo", async () => {
    renderGen();
    irAFoto();
    await screen.findByLabelText(en["gen3d.photo.choose"]);
    elegirFoto();

    expect(screen.getByAltText(en["gen3d.photo.previewAlt"])).toHaveAttribute(
      "src",
      "blob:preview",
    );
  });

  it("offers the pack download instead of the picker when the pack is missing", async () => {
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(
      photoCapabilityTree("needs_setup"),
    );
    renderGen();
    irAFoto();

    expect(await screen.findByText(en["gen3d.photo.missingPack"])).toBeInTheDocument();
    expect(screen.queryByLabelText(en["gen3d.photo.choose"])).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: en["gen3d.generate"] })).toBeDisabled();
  });

  it("surfaces an upload failure instead of staying silent", async () => {
    vi.mocked(generationService.uploadGenerationInitImage).mockRejectedValue(
      new Error("la subida fallo"),
    );
    renderGen();
    irAFoto();
    await screen.findByLabelText(en["gen3d.photo.choose"]);
    elegirFoto();
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/la subida fallo/);
    expect(printService.createShape3dJob).not.toHaveBeenCalled();
  });

  it("keeps the text lane exactly as it was", async () => {
    // Cambiar de modo y volver no puede romper el carril que ya funcionaba.
    renderGen();
    irAFoto();
    fireEvent.click(screen.getByRole("tab", { name: en["gen3d.mode.text"] }));
    generar();

    await waitFor(() => expect(printService.createShape3dJob).toHaveBeenCalled());
    expect(vi.mocked(printService.createShape3dJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ prompt: "algo", printer: "ender-3" }),
    );
  });
});

// ---------------------------------------------------------------------------
// La sugerencia de tamano. Lo que estos tests cuidan es lo unico que puede
// volverla peligrosa: que se aplique sola. Una malla no tiene cotas, asi que la
// medida es lo unico que decide si la pieza sirve — y esa decision es del
// usuario, con la sugerencia a la vista.
// ---------------------------------------------------------------------------

const CAMPO_MEDIDA = en["gen3d.targetMm"];

function escribirPedido(texto = "a coffee mug") {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: texto } });
}

async function estimar() {
  fireEvent.click(await screen.findByRole("button", { name: en["gen3d.estimate.action"] }));
  return screen.findByRole("button", { name: /Use 95 mm/ });
}

describe("MeshGenerator size suggestion", () => {
  it("does not offer to estimate when no model server is configured", async () => {
    // Requisito duro: sin servidor, la pantalla queda igual que siempre.
    vi.mocked(capabilitiesService.fetchCapabilityTree).mockResolvedValue(
      photoCapabilityTree("needs_setup", "needs_setup"),
    );
    renderGen();
    // El aviso del pack de foto sale del MISMO arbol de capacidades: esperarlo
    // prueba que el arbol ya llego, que es lo unico que podria hacer aparecer
    // el boton mas tarde.
    irAFoto();
    await screen.findByText(en["gen3d.photo.missingPack"]);
    fireEvent.click(screen.getByRole("tab", { name: en["gen3d.mode.text"] }));
    escribirPedido();

    expect(
      screen.queryByRole("button", { name: en["gen3d.estimate.action"] }),
    ).not.toBeInTheDocument();
  });

  it("offers to estimate once there is a model server", async () => {
    renderGen();
    escribirPedido();

    expect(
      await screen.findByRole("button", { name: en["gen3d.estimate.action"] }),
    ).toBeEnabled();
  });

  it("cannot estimate with nothing to estimate about", async () => {
    renderGen();

    expect(
      await screen.findByRole("button", { name: en["gen3d.estimate.action"] }),
    ).toBeDisabled();
  });

  it("shows the suggestion with the object it compared against", async () => {
    renderGen();
    escribirPedido();
    await estimar();

    expect(screen.getByText(/95 mm for the longest side/)).toBeInTheDocument();
    expect(screen.getByText(/standard coffee mug/)).toBeInTheDocument();
    expect(printService.estimatePrintSize).toHaveBeenCalledWith("a coffee mug");
  });

  it("never writes the suggestion into the field on its own", async () => {
    // ESTE es el test que importa: una medida que se aplica sola es una cota
    // inventada, y una malla generada no tiene con que respaldarla.
    renderGen();
    escribirPedido();
    await estimar();

    expect(screen.getByLabelText(CAMPO_MEDIDA)).toHaveValue(null);
  });

  it("does not overwrite a size the user already typed", async () => {
    renderGen();
    escribirPedido();
    fireEvent.change(screen.getByLabelText(CAMPO_MEDIDA), { target: { value: "42" } });
    await estimar();

    expect(screen.getByLabelText(CAMPO_MEDIDA)).toHaveValue(42);
  });

  it("applies the suggestion only when it is clicked", async () => {
    renderGen();
    escribirPedido();
    fireEvent.click(await estimar());

    expect(screen.getByLabelText(CAMPO_MEDIDA)).toHaveValue(95);
  });

  it("sends an accepted suggestion as such, with its reference", async () => {
    // El trabajo tiene que poder decir de donde salio su medida: sin esto, el
    // resultado no distingue una decision del usuario de una conjetura.
    renderGen();
    escribirPedido();
    fireEvent.click(await estimar());
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    await waitFor(() => expect(printService.createShape3dJob).toHaveBeenCalled());
    expect(vi.mocked(printService.createShape3dJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({
        targetMm: 95,
        targetMmSource: "estimate",
        targetMmReference: "standard coffee mug",
      }),
    );
  });

  it("stops calling it an estimate once the user edits the size by hand", async () => {
    renderGen();
    escribirPedido();
    fireEvent.click(await estimar());
    fireEvent.change(screen.getByLabelText(CAMPO_MEDIDA), { target: { value: "60" } });
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    await waitFor(() => expect(printService.createShape3dJob).toHaveBeenCalled());
    expect(vi.mocked(printService.createShape3dJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({
        targetMm: 60,
        targetMmSource: "user",
        targetMmReference: undefined,
      }),
    );
  });

  it("says it could not estimate instead of pretending a number", async () => {
    vi.mocked(printService.estimatePrintSize).mockRejectedValue(new Error("502"));
    renderGen();
    escribirPedido();
    fireEvent.click(await screen.findByRole("button", { name: en["gen3d.estimate.action"] }));

    expect(await screen.findByText(en["gen3d.estimate.failed"])).toBeInTheDocument();
    expect(screen.getByLabelText(CAMPO_MEDIDA)).toHaveValue(null);
  });

  it("a failed estimate does not block generating", async () => {
    vi.mocked(printService.estimatePrintSize).mockRejectedValue(new Error("502"));
    renderGen();
    escribirPedido();
    fireEvent.click(await screen.findByRole("button", { name: en["gen3d.estimate.action"] }));
    await screen.findByText(en["gen3d.estimate.failed"]);
    fireEvent.click(screen.getByRole("button", { name: en["gen3d.generate"] }));

    await waitFor(() => expect(printService.createShape3dJob).toHaveBeenCalled());
    expect(vi.mocked(printService.createShape3dJob).mock.calls[0][0].targetMm).toBeUndefined();
  });

  it("drops a suggestion that no longer matches what is being asked for", async () => {
    // Una sugerencia para "una taza" no puede seguir en pantalla cuando el
    // pedido ya dice otra cosa.
    renderGen();
    escribirPedido();
    await estimar();
    escribirPedido("an M3 screw");

    expect(
      screen.queryByRole("button", { name: /Use 95 mm/ }),
    ).not.toBeInTheDocument();
  });

  it("estimates from the photo file name when it says something", async () => {
    renderGen();
    irAFoto();
    await screen.findByLabelText(en["gen3d.photo.choose"]);
    elegirFoto(new File(["png"], "coffee-mug.jpg", { type: "image/jpeg" }));
    fireEvent.click(await screen.findByRole("button", { name: en["gen3d.estimate.action"] }));

    await waitFor(() => expect(printService.estimatePrintSize).toHaveBeenCalled());
    expect(printService.estimatePrintSize).toHaveBeenCalledWith("coffee mug");
  });

  it("does not estimate from a camera file name that says nothing", async () => {
    renderGen();
    irAFoto();
    await screen.findByLabelText(en["gen3d.photo.choose"]);
    elegirFoto(new File(["png"], "IMG_20260808_143255.jpg", { type: "image/jpeg" }));

    expect(
      await screen.findByRole("button", { name: en["gen3d.estimate.action"] }),
    ).toBeDisabled();
  });

  it("says where the size of a finished mesh came from", async () => {
    vi.mocked(printService.getShape3dJob).mockResolvedValue({
      ...LISTA,
      targetMm: 95,
      targetMmSource: "estimate",
      targetMmReference: "standard coffee mug",
    });
    renderGen();
    generar();

    expect(await screen.findByText(/the model suggested and you confirmed/i)).toBeInTheDocument();
  });

  it("says plainly when the size was nobody's decision", async () => {
    renderGen();
    generar();

    expect(await screen.findByText(/nobody chose that size/i)).toBeInTheDocument();
  });
});
