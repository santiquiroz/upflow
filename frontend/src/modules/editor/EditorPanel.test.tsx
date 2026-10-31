import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import * as editorService from "../../services/editor";
import * as generationService from "../../services/generation";
import { composeCommittedMasks } from "./editorMaskCanvas";
import {
  BASE_IMAGE,
  CAPABILITIES,
  enableMaskPixels,
  installEditorPanelHooks,
  loadBaseImage,
  paintOnCanvas,
  renderPanel,
  TestPointerEvent,
  withLamaCapabilities,
} from "./editorPanelTestKit";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});
vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return {
    ...actual,
    uploadGenerationInitImage: vi.fn(),
    createGenerationJob: vi.fn(),
    getGenerationJob: vi.fn(),
    fetchGenerationCapabilities: vi.fn(),
    cancelGenerationJob: vi.fn(),
  };
});
vi.mock("../../services/editor", () => ({
  fetchEditorCapabilities: vi.fn(),
  detectPeople: vi.fn(),
  segmentEditorObject: vi.fn(),
  segmentEditorBox: vi.fn(),
  healEditorImage: vi.fn(),
  fetchInitImage: vi.fn(),
}));
// Cuenta las recomposiciones del historial completo: solo deben ocurrir al cambiar el historial, nunca mientras se pinta.
vi.mock("./editorMaskCanvas", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./editorMaskCanvas")>();
  return { ...actual, composeCommittedMasks: vi.fn(actual.composeCommittedMasks) };
});

installEditorPanelHooks();

describe("EditorPanel", () => {
  it("offers the OSOR download under eraser-auto when OSOR is missing", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
      ...CAPABILITIES,
      models: [{ id: "eraser-auto", name: "Auto", status: "installed", eraseOnly: true, supportsInpaint: true }],
    });
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true, osor: false });
    renderPanel();
    await loadBaseImage();

    expect(await screen.findByText(en["editor.erase.osorHint"])).toBeVisible();
    expect(screen.getByRole("button", { name: "Download" })).toBeVisible();
  });

  it("hides the OSOR hint once OSOR is installed", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
      ...CAPABILITIES,
      models: [{ id: "eraser-auto", name: "Auto", status: "installed", eraseOnly: true, supportsInpaint: true }],
    });
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true, osor: true });
    renderPanel();
    await loadBaseImage();

    expect(screen.queryByText(en["editor.erase.osorHint"])).not.toBeInTheDocument();
  });

  it("defaults to eraser-auto and visibly disables replacement without a fill model", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
      ...CAPABILITIES,
      models: [
        {
          id: "eraser-auto",
          name: "Borrado automático (LaMa + OSOR)",
          status: "installed",
          eraseOnly: true,
          supportsInpaint: true,
        },
      ],
    });
    renderPanel();
    await loadBaseImage();

    expect(screen.getByLabelText(en["editor.model.label"])).toHaveValue("eraser-auto");
    const replaceRadio = screen.getByRole("radio", { name: en["editor.mode.replace"] });
    expect(replaceRadio).toBeDisabled();
    expect(replaceRadio).toHaveAccessibleDescription("Needs a fill model (SDXL)");
    expect(screen.getByText(en["editor.replace.needsSdxl"])).toBeVisible();
  });

  it("does not show replacement as unavailable while capabilities are loading", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockReturnValue(new Promise(() => {}));
    renderPanel();
    await loadBaseImage();

    expect(screen.getByRole("radio", { name: en["editor.mode.replace"] })).toBeEnabled();
    expect(screen.queryByText(en["editor.replace.needsSdxl"])).not.toBeInTheDocument();
  });

  it("only offers installed models with real inpaint support", async () => {
    renderPanel();
    await loadBaseImage();

    const select = screen.getByLabelText(en["editor.model.label"]);
    const labels = Array.from(select.querySelectorAll("option")).map((option) => option.textContent);
    expect(labels).toContain("epicrealism");
    expect(labels).not.toContain("lcm-fast");
    expect(labels).not.toContain("converting-one");
  });

  it("keeps the submit button disabled until there is a mask", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });

    const submit = screen.getByRole("button", { name: en["editor.submit.erase"] });
    expect(submit).toBeDisabled();

    paintOnCanvas();
    await waitFor(() => expect(submit).toBeEnabled());
  });

  it("submits an inpaint job with base and mask tokens", async () => {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "mask-token" });
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({
      id: "job-1",
      status: "queued",
    } as never);
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();

    fireEvent.click(await screen.findByRole("button", { name: en["editor.submit.erase"] }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    const params = vi.mocked(generationService.createGenerationJob).mock.calls[0][0];
    expect(params.initImageToken).toBe("aabbcc");
    expect(params.maskImageToken).toBe("mask-token");
    expect(params.width % 64).toBe(0);
    expect(params.height % 64).toBe(0);
    expect(params.prompt.length).toBeGreaterThan(0);
    expect(params.protectMaskToken).toBeUndefined();
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2);
  });

  it("keeps the remove-selection controls visible after a job completes", async () => {
    enableMaskPixels();
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "mask-token" });
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({
      id: "job-review",
      status: "queued",
    } as never);
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({
      id: "job-review",
      status: "completed",
      prompt: "background",
      negativePrompt: null,
      modelId: "gen--sdxl",
      steps: 30,
      guidance: 7,
      width: 640,
      height: 480,
      seed: null,
      device: "dml:0",
      autoUpscale: false,
      createdAt: "2026-10-30T00:00:00Z",
      startedAt: "2026-10-30T00:00:01Z",
      finishedAt: "2026-10-30T00:00:02Z",
      progressPct: 100,
      stages: [],
      error: null,
      ownerId: null,
      downloadUrl: "/api/v1/generation/jobs/job-review/download",
    });
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();

    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));

    await waitFor(() => expect(generationService.getGenerationJob).toHaveBeenCalledWith("job-review"));
    expect(await screen.findByRole("button", { name: en["editor.useAsBase"] })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove selection" })).toBeInTheDocument();
  });

  it("replace mode requires a prompt before submitting", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();

    fireEvent.click(screen.getByRole("radio", { name: en["editor.mode.replace"] }));
    const submit = screen.getByRole("button", { name: en["editor.submit.replace"] });
    expect(submit).toBeDisabled();

    fireEvent.change(screen.getByLabelText(en["editor.prompt.label"]), {
      target: { value: "a wooden bench" },
    });
    await waitFor(() => expect(submit).toBeEnabled());
  });

  it("tap tool asks the backend for the object mask", async () => {
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    renderPanel();
    await loadBaseImage();

    fireEvent.click(await screen.findByRole("radio", { name: en["editor.tool.tap"] }));
    fireEvent.click(screen.getByTestId("editor-mask-canvas"), { clientX: 5, clientY: 5 });

    await waitFor(() => expect(editorService.segmentEditorObject).toHaveBeenCalledOnce());
    expect(vi.mocked(editorService.segmentEditorObject).mock.calls[0][0]).toBe("aabbcc");
  });

  it("disables the tap tool when the segmenter is not installed", async () => {
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: false });
    renderPanel();
    await loadBaseImage();

    await waitFor(() =>
      expect(screen.getByRole("radio", { name: en["editor.tool.tap"] })).toBeDisabled(),
    );
  });
});

describe("EditorPanel rail and rendering", () => {
  it("keeps one tab stop per radio group so Tab reaches the Retouch group while a Select tool is active", async () => {
    renderPanel();
    await loadBaseImage();

    expect(screen.getByRole("radio", { name: en["editor.tool.brush"] })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("radio", { name: en["editor.tool.eraser"] })).toHaveAttribute("tabindex", "-1");
    expect(screen.getByRole("radio", { name: en["editor.tool.restore"] })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("radio", { name: en["editor.tool.clone"] })).toHaveAttribute("tabindex", "-1");
  });

  it("disables Spot heal with its reason until the LaMa eraser is installed", async () => {
    renderPanel();
    await loadBaseImage();

    const heal = screen.getByRole("radio", { name: en["editor.tool.heal"] });
    expect(heal).toBeDisabled();
    expect(heal).toHaveAttribute("title", en["editor.heal.needsLama"]);
  });

  it("enables Spot heal once the LaMa eraser is installed", async () => {
    withLamaCapabilities();
    renderPanel();
    await loadBaseImage();

    expect(screen.getByRole("radio", { name: en["editor.tool.heal"] })).toBeEnabled();
  });

  it("does not recompose the history while a stroke is in progress", async () => {
    renderPanel();
    await loadBaseImage();
    const canvas = screen.getByTestId("editor-mask-canvas");
    const before = vi.mocked(composeCommittedMasks).mock.calls.length;

    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, pointerId: 1 });
    fireEvent.pointerMove(canvas, { clientX: 30, clientY: 30, pointerId: 1 });
    fireEvent.pointerMove(canvas, { clientX: 60, clientY: 60, pointerId: 1 });

    expect(vi.mocked(composeCommittedMasks).mock.calls.length).toBe(before);
    fireEvent.pointerUp(canvas, { pointerId: 1 });
    expect(vi.mocked(composeCommittedMasks).mock.calls.length).toBe(before + 1);
  });
});

describe("EditorPanel vista avanzada", () => {
  async function setupReadyToSubmit() {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "mask-token" });
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({
      id: "job-2",
      status: "queued",
    } as never);
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();
    fireEvent.click(screen.getByRole("button", { name: (n) => n.startsWith(en["editor.advanced.title"]) }));
  }

  it("advanced params flow into the job payload", async () => {
    await setupReadyToSubmit();

    fireEvent.change(screen.getByLabelText(en["editor.advanced.steps"]), { target: { value: "45" } });
    fireEvent.change(screen.getByLabelText(en["editor.advanced.negativePrompt"]), {
      target: { value: "blurry, low quality" },
    });
    fireEvent.change(screen.getByLabelText(en["editor.advanced.seed"]), { target: { value: "1234" } });

    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));
    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());

    const params = vi.mocked(generationService.createGenerationJob).mock.calls[0][0];
    expect(params.steps).toBe(45);
    expect(params.negativePrompt).toBe("blurry, low quality");
    expect(params.seed).toBe(1234);
    expect(params.strength).toBe(1);
  });

  it("the erase fill prompt is editable and used as the job prompt", async () => {
    await setupReadyToSubmit();

    fireEvent.change(screen.getByLabelText(/Fill prompt/), {
      target: { value: "clean sand, beach" },
    });

    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));
    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0].prompt).toBe(
      "clean sand, beach",
    );
  });

  it("clearing the erase fill prompt blocks submission", async () => {
    await setupReadyToSubmit();

    fireEvent.change(screen.getByLabelText(/Fill prompt/), { target: { value: "  " } });

    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
  });
});

describe("EditorPanel zoom y desplazamiento", () => {
  it("zooms in and out from the toolbar and can reset", async () => {
    renderPanel();
    await loadBaseImage();

    expect(screen.getByText("100%")).toBeInTheDocument();
    const reset = screen.getByRole("button", { name: en["editor.zoom.reset"] });
    expect(reset).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.in"] }));
    expect(screen.getByText("150%")).toBeInTheDocument();
    expect(reset).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.reset"] }));
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("never zooms out past the fitted view", async () => {
    renderPanel();
    await loadBaseImage();

    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.out"] }));

    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("the hand tool drags the view instead of painting", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("button", { name: en["editor.tool.pan"] }));

    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 100, clientY: 100, pointerId: 1, button: 0 });
    fireEvent.pointerMove(canvas, { clientX: 40, clientY: 40, pointerId: 1 });
    fireEvent.pointerUp(canvas, { pointerId: 1 });

    // Sin trazos: el submit sigue bloqueado por falta de máscara.
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
  });

  it("resets the view when a new image is loaded", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.in"] }));
    expect(screen.getByText("150%")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Image to edit|Use another image/), {
      target: { files: [new File(["img2"], "otra.png", { type: "image/png" })] },
    });

    await waitFor(() => expect(screen.getByText("100%")).toBeInTheDocument());
  });
});

describe("EditorPanel: la vista sobrevive a un cambio de tamaño", () => {
  it("re-acota el desplazamiento cuando el lienzo encoge", async () => {
    // Regresión (review adversarial): con la ventana grande se podía arrastrar
    // hasta un desplazamiento legal que, al achicar la ventana, dejaba la foto
    // entera fuera de la vista y el editor en blanco.
    const observers: Array<() => void> = [];
    class FakeResizeObserver {
      constructor(private readonly cb: () => void) {
        observers.push(() => this.cb());
      }
      observe() {}
      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);

    let clientWidth = 1200;
    Object.defineProperty(HTMLDivElement.prototype, "clientWidth", {
      configurable: true,
      get() {
        return this.dataset?.testid === "editor-stage" ? clientWidth : 0;
      },
    });
    Object.defineProperty(HTMLDivElement.prototype, "clientHeight", {
      configurable: true,
      get() {
        return this.dataset?.testid === "editor-stage" ? 800 : 0;
      },
    });

    try {
      renderPanel();
      await loadBaseImage();

      fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.in"] }));
      const stage = screen.getByTestId("editor-stage");
      const inner = stage.firstElementChild as HTMLElement;
      const panned = inner.style.transform;
      expect(panned).toContain("scale(1.5)");

      clientWidth = 300;
      act(() => {
        observers.forEach((fire) => fire());
      });

      const clamped = inner.style.transform;
      const panX = Number(clamped.match(/translate\((-?[\d.]+)px/)?.[1] ?? "0");
      // Con 300 px de ancho a 1.5x el mínimo legal es -150: nunca menos.
      expect(panX).toBeGreaterThanOrEqual(-150);
    } finally {
      vi.unstubAllGlobals();
      delete (HTMLDivElement.prototype as never as Record<string, unknown>).clientWidth;
      delete (HTMLDivElement.prototype as never as Record<string, unknown>).clientHeight;
    }
  });
});

describe("EditorPanel: diseño de tres columnas", () => {
  it("marca la herramienta activa sin usar el relleno ámbar", async () => {
    renderPanel();
    await loadBaseImage();

    const activeTool = screen.getByRole("radio", { name: en["editor.tool.brush"] });
    expect(activeTool).toHaveAttribute("aria-checked", "true");
    expect(activeTool.className).not.toContain("bg-accent");
    expect(activeTool).toHaveStyle({ boxShadow: "inset 3px 0 0 var(--accent)" });
  });

  it("mueve el foco y selecciona herramientas con las flechas del radiogroup", async () => {
    renderPanel();
    await loadBaseImage();

    const brush = screen.getByRole("radio", { name: en["editor.tool.brush"] });
    fireEvent.keyDown(brush, { key: "ArrowDown" });

    const eraser = screen.getByRole("radio", { name: en["editor.tool.eraser"] });
    expect(eraser).toHaveAttribute("aria-checked", "true");
    expect(eraser).toHaveFocus();
  });

  it("muestra la acción dentro del panel complementario", async () => {
    renderPanel();
    await loadBaseImage();

    const sidePanel = screen.getByRole("complementary");
    expect(
      within(sidePanel).getByRole("button", { name: en["editor.submit.erase"] }),
    ).toBeInTheDocument();
  });

  it("cambia el tamaño del pincel con los atajos y respeta los límites", async () => {
    renderPanel();
    await loadBaseImage();

    const brushSize = screen.getByLabelText(en["editor.brushSize"]);
    fireEvent.keyDown(window, { key: "]" });
    expect(brushSize).toHaveValue("36");

    for (let step = 0; step < 30; step += 1) {
      fireEvent.keyDown(window, { key: "]" });
    }
    expect(brushSize).toHaveValue("128");

    for (let step = 0; step < 40; step += 1) {
      fireEvent.keyDown(window, { key: "[" });
    }
    expect(brushSize).toHaveValue("8");
  });

  it("clone without a source shows the Alt+click notice until a source is set", async () => {
    vi.stubGlobal("PointerEvent", TestPointerEvent);
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.clone"] }));
    expect(screen.getByText(en["editor.retouch.cloneNoSource"])).toBeInTheDocument();

    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 40, clientY: 40, pointerId: 1, altKey: true });
    fireEvent.pointerUp(canvas, { pointerId: 1 });
    expect(screen.queryByText(en["editor.retouch.cloneNoSource"])).not.toBeInTheDocument();
  });
});
