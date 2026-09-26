import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import type { VideoCapabilities } from "../../../lib/apiTypes";
import { DISPLAYED_LITE, drag, installPointerEvent } from "./boxEditorTestUtils";
import { CctvJobSetup } from "./CctvJobSetup";
import { ANALYSIS, PRESETS_RESPONSE } from "./cctvFixtures";
import { EMPTY_ROI, type RoiChoice } from "./cctvRoi";
import { RoiFusionPanel } from "./RoiFusionPanel";

installPointerEvent();

const CAPS: VideoCapabilities = {
  interpEngines: [],
  cctvAvailable: true,
  cctvReasonKey: null,
  cctvAiAvailable: true,
  cctvAiReasonKey: null,
  cctvUnavailableSteps: [],
};

function StatefulPanel({ initial, frame, onShowFrame }: { initial: RoiChoice; frame: number; onShowFrame: (frame: number) => void }) {
  const [roi, setRoi] = useState(initial);
  return (
    <RoiFusionPanel roi={roi} frame={frame} frameCount={750} index={ANALYSIS.frameIndex} onChange={setRoi} onShowFrame={onShowFrame} />
  );
}

function renderPanel(initial: RoiChoice = EMPTY_ROI, frame = 12) {
  const onShowFrame = vi.fn();
  render(<StatefulPanel initial={initial} frame={frame} onShowFrame={onShowFrame} />);
  return onShowFrame;
}


function renderSetup(onSubmit = vi.fn()) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  render(<CctvJobSetup analysis={ANALYSIS} presets={PRESETS_RESPONSE} capabilities={CAPS} busy={false} onSubmit={onSubmit} />, {
    wrapper: Wrapper,
  });
  return onSubmit;
}

function openMultiFrameStill(): void {
  fireEvent.click(screen.getByRole("tab", { name: en["cctv.task.roi_fusion"] }));
}

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(DISPLAYED_LITE as DOMRect);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("RoiFusionPanel", () => {
  it("sets the range from the frame on screen and counts it against the limit", () => {
    renderPanel();

    expect(screen.getByText(en["cctv.roi.range.empty"].replace("{{max}}", "60"))).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: en["cctv.trim.start.useCurrent"] }));
    fireEvent.change(screen.getByLabelText(en["cctv.trim.end"]), { target: { value: "41" } });

    expect(screen.getByLabelText(en["cctv.trim.start"])).toHaveValue(12);
    expect(screen.getByText("30 frames selected (at most 60)")).toBeInTheDocument();
  });

  it("keeps a typed frame inside the video", () => {
    renderPanel();

    fireEvent.change(screen.getByLabelText(en["cctv.trim.end"]), { target: { value: "9999" } });

    expect(screen.getByLabelText(en["cctv.trim.end"])).toHaveValue(749);
  });

  it("warns live when a face is narrower than about 40 recorded pixels", () => {
    renderPanel({ ...EMPTY_ROI, box: [10, 10, 32, 40], reference: 12 });

    fireEvent.click(screen.getByRole("radio", { name: en["cctv.roi.kind.face_or_object"] }));

    expect(screen.getByRole("note")).toHaveTextContent("This face is about 32 px wide in the recording.");
  });

  it("states a plate's recorded height", () => {
    renderPanel({ ...EMPTY_ROI, box: [10, 10, 60, 14], reference: 12 });

    expect(screen.getByRole("note")).toHaveTextContent("The plate is about 14 px tall in the recording.");
  });

  it("offers 2x, 3x and 4x and both ways to combine", () => {
    renderPanel();

    fireEvent.click(screen.getByRole("radio", { name: "3x" }));
    fireEvent.click(screen.getByRole("radio", { name: en["cctv.roi.method.trimmed_mean"] }));

    expect(screen.getByRole("radio", { name: "3x" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: en["cctv.roi.method.trimmed_mean"] })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("radio", { name: en["cctv.roi.method.median"] })).toHaveAttribute("aria-checked", "false");
  });

  it("takes the user back to the reference frame", () => {
    const onShowFrame = renderPanel({ ...EMPTY_ROI, box: [10, 10, 60, 14], reference: 20 });

    expect(screen.getByText("Reference frame: 20")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: en["cctv.roi.reference.show"] }));

    expect(onShowFrame).toHaveBeenCalledWith(20);
  });
});

describe("multi-frame still in the job setup", () => {
  it("swaps the on-screen text tools for the region tools and keeps only the pre-alignment steps", () => {
    renderSetup();

    openMultiFrameStill();

    expect(screen.getByRole("group", { name: en["cctv.box.surfaceRoi"] })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: en["cctv.osd.none"] })).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: en["cctv.trim.toggle"] })).not.toBeInTheDocument();
    const steps = within(screen.getByRole("list", { name: en["cctv.steps.legend"] })).getAllByRole("checkbox");
    expect(steps.map((box) => box.closest("label")?.textContent)).toEqual([en["cctv.step.deinterlace"], en["cctv.step.deblock"]]);
  });

  it("blocks Start until the region and its frames are set, then sends them without the AI dialog", () => {
    const onSubmit = renderSetup();
    fireEvent.click(screen.getByRole("radio", { name: new RegExp(en["cctv.lane.ai"].replace(/[()]/g, "\\$&")) }));
    openMultiFrameStill();
    const start = screen.getByRole("button", { name: en["cctv.start"] });

    expect(start).toBeDisabled();
    expect(screen.getByText(en["cctv.roi.blocked.box"])).toBeInTheDocument();

    drag(screen.getByRole("group", { name: en["cctv.box.surfaceRoi"] }), [100, 100], [300, 200]);
    expect(screen.getByText(en["cctv.roi.blocked.range"])).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(en["cctv.trim.start"]), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText(en["cctv.trim.end"]), { target: { value: "29" } });
    fireEvent.click(start);

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const request = onSubmit.mock.calls[0][0];
    expect(request.task).toBe("roi_fusion");
    expect(request.roi).toEqual({
      firstFrame: 0,
      lastFrame: 29,
      referenceFrame: 0,
      box: [50, 100, 100, 100],
      kind: "plate",
      scale: 2,
      method: "median",
    });
    expect(request.steps.map((step: { id: string }) => step.id)).toEqual(["deblock"]);
    expect(request).not.toHaveProperty("modelId");
  });

  it("refuses more frames than the limit", () => {
    renderSetup();
    openMultiFrameStill();
    drag(screen.getByRole("group", { name: en["cctv.box.surfaceRoi"] }), [100, 100], [300, 200]);

    fireEvent.change(screen.getByLabelText(en["cctv.trim.start"]), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText(en["cctv.trim.end"]), { target: { value: "60" } });

    expect(screen.getByRole("button", { name: en["cctv.start"] })).toBeDisabled();
    expect(screen.getByText(en["cctv.roi.blocked.range"])).toBeInTheDocument();
  });
});
