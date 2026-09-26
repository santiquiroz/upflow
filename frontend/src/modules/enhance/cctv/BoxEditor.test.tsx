import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import type { VideoCapabilities } from "../../../lib/apiTypes";
import * as cctvService from "../../../services/cctv";
import type { CctvBox } from "../../../services/cctv";
import { BoxEditor } from "./BoxEditor";
import { CctvJobSetup } from "./CctvJobSetup";
import { DISPLAYED_LITE, drag, installPointerEvent } from "./boxEditorTestUtils";
import { ANALYSIS, PRESETS_RESPONSE } from "./cctvFixtures";

vi.mock("../../../services/cctv", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../services/cctv")>();
  return { ...actual, checkCctvOsd: vi.fn() };
});

installPointerEvent();

const LITE = { width: 960, height: 1080 };

const CAPS: VideoCapabilities = {
  interpEngines: [],
  cctvAvailable: true,
  cctvReasonKey: null,
  cctvAiAvailable: false,
  cctvAiReasonKey: "capability.setup.needsGpu",
  cctvUnavailableSteps: [],
};

function surfaceOf(name: string): HTMLElement {
  return screen.getByRole("group", { name });
}

function renderEditor(kind: "osd" | "roi", boxes: CctvBox[] = [], disabled = false) {
  const onChange = vi.fn();
  render(
    <div>
      <BoxEditor kind={kind} frameSize={LITE} boxes={boxes} onChange={onChange} flagged={[1]} disabled={disabled} />
    </div>,
  );
  return onChange;
}

function renderSetup() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  render(<CctvJobSetup analysis={ANALYSIS} presets={PRESETS_RESPONSE} capabilities={CAPS} busy={false} onSubmit={vi.fn()} />, {
    wrapper: Wrapper,
  });
}

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(DISPLAYED_LITE as DOMRect);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.mocked(cctvService.checkCctvOsd).mockReset();
});

describe("BoxEditor", () => {
  it("reads the size in recorded pixels while the region is drawn", () => {
    const onChange = renderEditor("roi");

    drag(surfaceOf(en["cctv.box.surfaceRoi"]), [100, 100], [300, 200], false);

    expect(screen.getByRole("status")).toHaveTextContent("100 × 100 px in the recording");
    expect(screen.getByRole("status")).toHaveTextContent(en["cctv.roi.tight"]);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("adds the drawn box in stored-frame coordinates with even sides", () => {
    const onChange = renderEditor("osd", [[0, 0, 10, 10]]);

    drag(surfaceOf(en["cctv.box.surfaceOsd"]), [101, 100], [305, 201]);

    expect(onChange).toHaveBeenCalledWith([
      [0, 0, 10, 10],
      [50, 100, 102, 100],
    ]);
  });

  it("keeps a single region of interest", () => {
    const onChange = renderEditor("roi", [[0, 0, 10, 10]]);

    drag(surfaceOf(en["cctv.box.surfaceRoi"]), [100, 100], [300, 200]);

    expect(onChange).toHaveBeenCalledWith([[50, 100, 100, 100]]);
  });

  it("shows the size of the drawn region of interest", () => {
    renderEditor("roi", [[50, 100, 42, 18]]);

    expect(screen.getByRole("status")).toHaveTextContent("42 × 18 px in the recording");
  });

  it("ignores a click that doesn't draw a box", () => {
    const onChange = renderEditor("osd");

    drag(surfaceOf(en["cctv.box.surfaceOsd"]), [100, 100], [101, 101]);

    expect(onChange).not.toHaveBeenCalled();
  });

  it("moves, resizes and removes a box from the keyboard", () => {
    const onChange = renderEditor("osd", [[10, 10, 20, 20]]);
    const box = screen.getByRole("button", { name: "Box 1: 20 × 20 px at 10, 10" });

    fireEvent.keyDown(box, { key: "ArrowRight" });
    fireEvent.keyDown(box, { key: "ArrowDown", shiftKey: true });
    fireEvent.keyDown(box, { key: "Delete" });

    expect(onChange.mock.calls).toEqual([[[[12, 10, 20, 20]]], [[[10, 10, 20, 22]]], [[]]]);
  });

  it("keeps the focus on the box while it moves", () => {
    function Harness() {
      const [boxes, setBoxes] = useState<CctvBox[]>([[10, 10, 20, 20]]);
      return <BoxEditor kind="osd" frameSize={LITE} boxes={boxes} onChange={setBoxes} />;
    }
    render(<Harness />);
    const box = screen.getByRole("button", { name: "Box 1: 20 × 20 px at 10, 10" });
    box.focus();

    fireEvent.keyDown(box, { key: "ArrowRight" });
    fireEvent.keyDown(document.activeElement as Element, { key: "ArrowRight" });

    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Box 1: 20 × 20 px at 14, 10" }));
  });

  it("doesn't start a new box when the pointer lands on an existing one", () => {
    const onChange = renderEditor("osd", [[10, 10, 20, 20]]);
    const box = screen.getByRole("button", { name: "Box 1: 20 × 20 px at 10, 10" });

    fireEvent.pointerDown(box, { clientX: 30, clientY: 30, pointerId: 1 });
    fireEvent.pointerUp(surfaceOf(en["cctv.box.surfaceOsd"]), { pointerId: 1 });

    expect(onChange).not.toHaveBeenCalled();
  });

  it("marks the boxes that don't look like text", () => {
    renderEditor("osd", [[10, 10, 20, 20], [40, 40, 20, 20]]);

    expect(screen.getByRole("button", { name: "Box 2: 20 × 20 px at 40, 40" })).toHaveClass("border-warn");
    expect(screen.getByRole("button", { name: "Box 1: 20 × 20 px at 10, 10" })).toHaveClass("border-accent");
  });

  it("doesn't draw while disabled", () => {
    const onChange = renderEditor("osd", [], true);

    drag(surfaceOf(en["cctv.box.surfaceOsd"]), [100, 100], [300, 200]);

    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("on-screen text confirmation", () => {
  it("keeps Start disabled until the suggested boxes are confirmed or there is no on-screen text", async () => {
    vi.mocked(cctvService.checkCctvOsd).mockResolvedValue({ checks: [], warnings: [] });
    renderSetup();
    const start = screen.getByRole("button", { name: en["cctv.start"] });

    expect(start).toBeDisabled();
    expect(screen.getByRole("button", { name: "Box 1: 384 × 64 px at 19, 22" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: en["cctv.osd.confirmAction"] }));

    expect(start).toBeEnabled();
    await waitFor(() => expect(cctvService.checkCctvOsd).toHaveBeenCalled());
    expect(vi.mocked(cctvService.checkCctvOsd).mock.calls[0]).toEqual([
      "tok-1",
      [
        [19, 22, 384, 64],
        [701, 994, 240, 64],
      ],
      0,
    ]);
  });

  it("asks for a new confirmation after a box moves", async () => {
    vi.mocked(cctvService.checkCctvOsd).mockResolvedValue({ checks: [], warnings: [] });
    renderSetup();
    fireEvent.click(screen.getByRole("button", { name: en["cctv.osd.confirmAction"] }));
    await screen.findByText("Boxes confirmed on frame 0.");

    fireEvent.keyDown(screen.getByRole("button", { name: "Box 1: 384 × 64 px at 19, 22" }), { key: "ArrowUp" });

    expect(screen.getByRole("button", { name: en["cctv.start"] })).toBeDisabled();
    expect(screen.queryByText("Boxes confirmed on frame 0.")).not.toBeInTheDocument();
  });

  it("warns without blocking when a confirmed box doesn't look like text", async () => {
    vi.mocked(cctvService.checkCctvOsd).mockResolvedValue({
      checks: [
        { box: [19, 22, 384, 64], frames: 10, contrast: 80, staticFraction: 0.9, looksLikeText: true, warningKey: null },
        { box: [701, 994, 240, 64], frames: 10, contrast: 12, staticFraction: 0.2, looksLikeText: false, warningKey: "cctv.osd.notText" },
      ],
      warnings: ["cctv.osd.notText"],
    });
    renderSetup();

    fireEvent.click(screen.getByRole("button", { name: en["cctv.osd.confirmAction"] }));

    const list = screen.getByRole("list", { name: en["cctv.osd.legend"] });
    const rows = within(list).getAllByRole("listitem");
    await within(rows[1]).findByText(en["cctv.osd.notText"]);
    expect(within(rows[0]).queryByText(en["cctv.osd.notText"])).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: en["cctv.start"] })).toBeEnabled();
  });

  it("hides the boxes when there is no on-screen text", () => {
    renderSetup();

    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));

    expect(screen.queryByRole("button", { name: /^Box 1/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: en["cctv.start"] })).toBeEnabled();
  });

  it("brings back the suggested boxes after removing them", () => {
    renderSetup();
    fireEvent.click(screen.getByRole("button", { name: "Remove box 2" }));
    fireEvent.click(screen.getByRole("button", { name: "Remove box 1" }));

    expect(screen.getByRole("button", { name: en["cctv.osd.confirmAction"] })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: en["cctv.osd.useSuggested"] }));

    expect(screen.getByRole("button", { name: "Box 2: 240 × 64 px at 701, 994" })).toBeInTheDocument();
  });
});
