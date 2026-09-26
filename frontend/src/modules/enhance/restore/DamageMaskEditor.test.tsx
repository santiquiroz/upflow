import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../../../lib/api";
import type { EngineInfoResponse, JobResponse } from "../../../lib/apiTypes";
import type { RestoreAnalysis, RestoreStepOptions } from "../../../lib/restoreApiTypes";
import * as restoreService from "../../../services/restore";
import { DamageMaskEditor } from "./DamageMaskEditor";
import { maskSettingsFrom, type ProbabilityMap } from "./damageMask";
import { PhotoRestorePanel } from "./PhotoRestorePanel";
import * as probabilityMap from "./probabilityMap";
import { makeAnalysis, makeCapabilities } from "./restoreTestFixtures";
import { useDamageMask, type LoadProbabilityMap } from "./useDamageMask";

vi.mock("../../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/api")>();
  return { ...actual, getEngineInfo: vi.fn(), getJob: vi.fn() };
});

vi.mock("../../../services/restore", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../services/restore")>();
  return {
    ...actual,
    analyzePhoto: vi.fn(),
    setPhotoGeometry: vi.fn(),
    uploadDamageMask: vi.fn(),
    getRestoreCapabilities: vi.fn(),
    createRestoreJob: vi.fn(),
  };
});

vi.mock("./probabilityMap", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./probabilityMap")>();
  return { ...actual, loadProbabilityMap: vi.fn() };
});

const PROB_URL = "/api/v1/restore/analysis/tok-1/damage_prob.png";
const WIDTH = 100;
const HEIGHT = 50;

function mapWith(width: number, height: number, value: (x: number, y: number) => number): ProbabilityMap {
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) data[y * width + x] = value(x, y);
  }
  return { width, height, data };
}

// Una franja de 10 columnas con probabilidad 0,45: marcada a sensibilidad 0,5
// (umbral 0,4) y libre a 0,25 (umbral 0,5).
const STRIPE_MAP = mapWith(WIDTH, HEIGHT, (x) => (x < 10 ? 115 : 0));

function fakeContext() {
  return {
    clearRect: vi.fn(),
    createImageData: vi.fn((width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) })),
    putImageData: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    lineWidth: 0,
    lineCap: "",
    lineJoin: "",
    strokeStyle: "",
    fillStyle: "",
    globalCompositeOperation: "source-over",
  };
}

function stubCanvas(width: number, height: number) {
  HTMLCanvasElement.prototype.getContext = vi.fn(() => fakeContext()) as never;
  HTMLCanvasElement.prototype.getBoundingClientRect = vi.fn(
    () => ({ left: 0, top: 0, width, height, right: width, bottom: height, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect,
  );
}

function wrapperWith(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </MemoryRouter>
    );
  };
}

interface HarnessProps {
  analysis: RestoreAnalysis;
  loadMap: LoadProbabilityMap;
  onConfirm?: () => void;
}

function EditorHarness({ analysis, loadMap, onConfirm = () => undefined }: HarnessProps) {
  const [options, setOptions] = useState<RestoreStepOptions>({ sensitivity: 0.5, grow_px: 0 });
  const settings = maskSettingsFrom(options);
  const damage = useDamageMask({ analysis, revision: 1, settings, active: true, loadMap });
  return (
    <DamageMaskEditor
      previewUrl={analysis.previewUrl}
      alt="photo"
      size={{ width: analysis.width, height: analysis.height }}
      damage={damage}
      settings={settings}
      leaveLargeHoles={options.leave_large_holes === true}
      leaveFaces={options.leave_faces === true}
      damageOverFaces={analysis.damageOverFaces}
      onOptionChange={(option, value) => setOptions((previous) => ({ ...previous, [option]: value }))}
      onConfirm={onConfirm}
    />
  );
}

function editorAnalysis(overrides: Partial<RestoreAnalysis> = {}): RestoreAnalysis {
  return makeAnalysis({
    width: WIDTH,
    height: HEIGHT,
    damage: { coverage: 0.1, probUrl: PROB_URL, largeHoles: 0 },
    ...overrides,
  });
}

function renderEditor(props: HarnessProps) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<EditorHarness {...props} />, { wrapper: wrapperWith(queryClient) });
}

function maskCanvas(): HTMLCanvasElement {
  return screen.getByLabelText("Damage mask. Paint to mark damage, erase to keep the photo as it is.") as HTMLCanvasElement;
}

function drag(canvas: HTMLCanvasElement, from: [number, number], to: [number, number]) {
  fireEvent.pointerDown(canvas, { clientX: from[0], clientY: from[1], pointerId: 1, button: 0 });
  fireEvent.pointerMove(canvas, { clientX: to[0], clientY: to[1], pointerId: 1 });
  fireEvent.pointerUp(canvas, { clientX: to[0], clientY: to[1], pointerId: 1 });
}

// jsdom no trae PointerEvent: sin esto el evento llega sin clientX ni button.
class PointerEventStub extends MouseEvent {
  pointerId: number;

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 0;
  }
}

const originalPointerEvent = window.PointerEvent;

beforeAll(() => {
  window.PointerEvent = (originalPointerEvent ?? PointerEventStub) as typeof PointerEvent;
});

afterAll(() => {
  window.PointerEvent = originalPointerEvent;
});

beforeEach(() => {
  stubCanvas(WIDTH, HEIGHT);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.mocked(api.getEngineInfo).mockReset();
  vi.mocked(api.getJob).mockReset();
  vi.mocked(restoreService.analyzePhoto).mockReset();
  vi.mocked(restoreService.uploadDamageMask).mockReset();
  vi.mocked(restoreService.createRestoreJob).mockReset();
  vi.mocked(restoreService.getRestoreCapabilities).mockReset();
  vi.mocked(probabilityMap.loadProbabilityMap).mockReset();
});

describe("DamageMaskEditor", () => {
  it("overlays the detector map and moves the sensitivity without asking the backend again", async () => {
    const loadMap = vi.fn().mockResolvedValue(STRIPE_MAP);
    renderEditor({ analysis: editorAnalysis(), loadMap });

    expect(await screen.findByText("Marked as damage: 10% of the photo")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("slider", { name: "Detection sensitivity" }), { target: { value: "0.25" } });

    expect(await screen.findByText("Marked as damage: 0% of the photo")).toBeInTheDocument();
    expect(loadMap).toHaveBeenCalledTimes(1);
    expect(loadMap.mock.calls[0][0]).toBe(`${PROB_URL}?v=1`);
    expect(restoreService.analyzePhoto).not.toHaveBeenCalled();
    expect(restoreService.uploadDamageMask).not.toHaveBeenCalled();
  });

  it("paints damage with Add, removes it with Erase and undoes the last stroke", async () => {
    renderEditor({ analysis: editorAnalysis(), loadMap: vi.fn().mockResolvedValue(mapWith(WIDTH, HEIGHT, () => 0)) });
    await screen.findByText("Marked as damage: 0% of the photo");

    drag(maskCanvas(), [30, 25], [70, 25]);
    const painted = await screen.findByText(/Marked as damage: [1-9][\d.]*% of the photo/);
    const paintedText = painted.textContent;

    fireEvent.click(screen.getByRole("radio", { name: "Erase" }));
    drag(maskCanvas(), [40, 25], [60, 25]);
    await waitFor(() => expect(screen.getByText(/Marked as damage:/).textContent).not.toBe(paintedText));

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(await screen.findByText(paintedText as string)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear edits" }));
    expect(await screen.findByText("Marked as damage: 0% of the photo")).toBeInTheDocument();
  });

  it("warns about large holes and offers to leave them unfilled", async () => {
    const hole = mapWith(WIDTH, HEIGHT, (x, y) => (x >= 20 && x < 50 && y >= 10 && y < 40 ? 255 : 0));
    renderEditor({ analysis: editorAnalysis(), loadMap: vi.fn().mockResolvedValue(hole) });

    expect(await screen.findByText("Large missing areas will be filled with a smooth guess.")).toBeInTheDocument();
    const leave = screen.getByRole("checkbox", { name: "Leave large holes unfilled" });
    expect(leave).not.toBeChecked();
    fireEvent.click(leave);

    expect(screen.getByRole("checkbox", { name: "Leave large holes unfilled" })).toBeChecked();
  });

  it("always shows the note about handwriting and captions", async () => {
    renderEditor({ analysis: editorAnalysis(), loadMap: vi.fn().mockResolvedValue(STRIPE_MAP) });

    expect(
      screen.getByText("Check that handwriting, dates or captions on the photo aren't marked as damage."),
    ).toBeInTheDocument();
    await screen.findByText("Marked as damage: 10% of the photo");
  });

  it("offers to leave the faces unrepaired when damage falls on one", async () => {
    renderEditor({ analysis: editorAnalysis({ damageOverFaces: true }), loadMap: vi.fn().mockResolvedValue(STRIPE_MAP) });

    expect(await screen.findByText("Damage over a face will be filled with invented content.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: "Leave faces unrepaired" }));
    expect(screen.getByRole("checkbox", { name: "Leave faces unrepaired" })).toBeChecked();
  });

  it("still lets the user paint when the detector map can't be loaded", async () => {
    renderEditor({ analysis: editorAnalysis(), loadMap: vi.fn().mockRejectedValue(new Error("404")) });

    expect(
      await screen.findByText("Couldn't load the damage map. You can still mark damage with the brush."),
    ).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: "Detection sensitivity" })).toBeDisabled();
    drag(maskCanvas(), [30, 25], [70, 25]);
    expect(await screen.findByText(/Marked as damage: [1-9][\d.]*% of the photo/)).toBeInTheDocument();
  });

  it("hides the overlay with Show mask and confirms the review", async () => {
    const onConfirm = vi.fn();
    renderEditor({ analysis: editorAnalysis(), loadMap: vi.fn().mockResolvedValue(STRIPE_MAP), onConfirm });

    fireEvent.click(screen.getByRole("checkbox", { name: "Show mask" }));
    expect(maskCanvas()).toHaveClass("opacity-0");
    fireEvent.click(screen.getByRole("button", { name: "The mask looks right" }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
  });
});

describe("PhotoRestorePanel: mandatory mask review", () => {
  const QUEUED = { jobId: "job-9", status: "queued" } as JobResponse;
  const PANEL_WIDTH = 120;
  const PANEL_HEIGHT = 80;
  // Franja del 10%: por encima del 3% que obliga a revisar.
  const WIDE_DAMAGE = mapWith(PANEL_WIDTH, PANEL_HEIGHT, (x) => (x < 12 ? 200 : 0));

  function renderPanel() {
    vi.mocked(api.getEngineInfo).mockResolvedValue({ maxUploadMb: 50 } as EngineInfoResponse);
    vi.mocked(restoreService.getRestoreCapabilities).mockResolvedValue(makeCapabilities());
    vi.mocked(restoreService.createRestoreJob).mockResolvedValue(QUEUED);
    vi.mocked(restoreService.uploadDamageMask).mockResolvedValue({ coverage: 0.2, width: PANEL_WIDTH, height: PANEL_HEIGHT });
    vi.mocked(api.getJob).mockResolvedValue({ ...QUEUED, status: "running" } as JobResponse);
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(
      makeAnalysis({ width: PANEL_WIDTH, height: PANEL_HEIGHT, damage: { coverage: 0.1, probUrl: PROB_URL, largeHoles: 0 } }),
    );
    vi.mocked(probabilityMap.loadProbabilityMap).mockResolvedValue(WIDE_DAMAGE);
    stubCanvas(PANEL_WIDTH, PANEL_HEIGHT);
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<PhotoRestorePanel />, { wrapper: wrapperWith(queryClient) });
    const input = document.getElementById("restore-file-input") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["x"], "grandma.jpg", { type: "image/jpeg" })] } });
  }

  it("opens the mask instead of restoring while a high-coverage mask is unchecked", async () => {
    renderPanel();

    expect(await screen.findByText("Check the damage mask before restoring.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    expect(
      await screen.findByText("Check that handwriting, dates or captions on the photo aren't marked as damage."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check damage mask" })).toHaveAttribute("aria-expanded", "true");
    expect(restoreService.createRestoreJob).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "The mask looks right" }));
    expect(screen.getByText("Mask checked.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    await waitFor(() => expect(restoreService.createRestoreJob).toHaveBeenCalledTimes(1));
    expect(restoreService.uploadDamageMask).not.toHaveBeenCalled();
    expect(vi.mocked(restoreService.createRestoreJob).mock.calls[0][0].options.repair).not.toHaveProperty("use_user_mask");
  });

  it("uploads the edited mask before creating the job and asks the backend to use it", async () => {
    renderPanel();
    await screen.findByText("Check the damage mask before restoring.");
    fireEvent.click(screen.getByRole("button", { name: "Check damage mask" }));
    await screen.findAllByText("Marked as damage: 10% of the photo");

    drag(maskCanvas(), [60, 40], [100, 40]);
    await screen.findAllByText(/Marked as damage: (1[1-9]|[2-9]\d)[\d.]*% of the photo/);
    fireEvent.click(screen.getByRole("button", { name: "The mask looks right" }));
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    await waitFor(() => expect(restoreService.createRestoreJob).toHaveBeenCalledTimes(1));
    const [token, mask] = vi.mocked(restoreService.uploadDamageMask).mock.calls[0];
    expect(token).toBe("tok-1");
    expect(mask.type).toBe("image/png");
    expect(vi.mocked(restoreService.uploadDamageMask).mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(restoreService.createRestoreJob).mock.invocationCallOrder[0],
    );
    expect(vi.mocked(restoreService.createRestoreJob).mock.calls[0][0].options.repair).toMatchObject({
      use_user_mask: true,
    });
  });

  it("says so and creates no job when the edited mask can't be encoded", async () => {
    vi.stubGlobal(
      "CompressionStream",
      class {
        constructor() {
          throw new Error("unsupported");
        }
      },
    );
    renderPanel();
    await screen.findByText("Check the damage mask before restoring.");
    fireEvent.click(screen.getByRole("button", { name: "Check damage mask" }));
    await screen.findAllByText("Marked as damage: 10% of the photo");
    drag(maskCanvas(), [60, 40], [100, 40]);
    fireEvent.click(screen.getByRole("button", { name: "The mask looks right" }));
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    expect(await screen.findByText("Couldn't prepare the damage mask. Try again.")).toBeInTheDocument();
    expect(restoreService.uploadDamageMask).not.toHaveBeenCalled();
    expect(restoreService.createRestoreJob).not.toHaveBeenCalled();
  });

  it("asks for a new review after the sensitivity changes", async () => {
    renderPanel();
    await screen.findByText("Check the damage mask before restoring.");
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    fireEvent.click(await screen.findByRole("button", { name: "The mask looks right" }));
    expect(screen.getByText("Mask checked.")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Check damage mask" }));
    fireEvent.change(screen.getAllByRole("slider", { name: "Detection sensitivity" })[0], { target: { value: "0.6" } });

    expect(await screen.findByText("Check the damage mask before restoring.")).toBeInTheDocument();
  });
});
