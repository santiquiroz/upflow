import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { useState, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import type { VideoCapabilities } from "../../../lib/apiTypes";
import { DISPLAYED_LITE, drag, installPointerEvent } from "./boxEditorTestUtils";
import { CctvJobSetup } from "./CctvJobSetup";
import { ANALYSIS, PRESETS_RESPONSE } from "./cctvFixtures";
import type { TrimRange } from "./cctvFrames";
import { EMPTY_REDACTION, withTrackAdded, type RedactionChoice } from "./cctvRedaction";
import { RedactionPanel } from "./RedactionPanel";

installPointerEvent();

const CAPS: VideoCapabilities = {
  interpEngines: [],
  cctvAvailable: true,
  cctvReasonKey: null,
  cctvAiAvailable: true,
  cctvAiReasonKey: null,
  cctvUnavailableSteps: [],
};

interface StatefulPanelProps {
  initial: RedactionChoice;
  frame: number;
  span: TrimRange;
  onShowFrame: (frame: number) => void;
}

function StatefulPanel({ initial, frame, span, onShowFrame }: StatefulPanelProps) {
  const [redaction, setRedaction] = useState(initial);
  return <RedactionPanel redaction={redaction} frame={frame} span={span} onChange={setRedaction} onShowFrame={onShowFrame} />;
}

function renderPanel(initial: RedactionChoice, frame = 40, span: TrimRange = [0, 749]) {
  const onShowFrame = vi.fn();
  render(<StatefulPanel initial={initial} frame={frame} span={span} onShowFrame={onShowFrame} />);
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

function openRedaction(): void {
  fireEvent.click(screen.getByRole("tab", { name: en["cctv.task.redact"] }));
}

const ONE_BOX = withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 5, [0, 749]);

beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(DISPLAYED_LITE as DOMRect);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("RedactionPanel", () => {
  it("says the copy is for sharing and never part of the handover package, and that nothing is detected", () => {
    renderPanel(EMPTY_REDACTION);

    expect(screen.getByRole("note")).toHaveTextContent(en["cctv.redact.notice"]);
    expect(screen.getByText(en["cctv.redact.noDetector"])).toBeInTheDocument();
    expect(screen.getByText(en["cctv.redact.empty"])).toBeInTheDocument();
  });

  it("offers the solid box by default, then pixelate and blur with a warning that they can be reversed", () => {
    renderPanel(ONE_BOX);

    expect(screen.getByRole("radio", { name: en["cctv.redact.style.fill"] })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText(en["cctv.redact.style.fillHint"])).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: en["cctv.redact.style.blur"] }));
    expect(screen.getByRole("radio", { name: en["cctv.redact.style.blur"] })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText(en["cctv.redact.style.blurHint"])).toBeInTheDocument();
  });

  it("names the frames of the copy that no box covers", () => {
    renderPanel(withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 150, [120, 180]), 150, [100, 200]);

    expect(screen.getByRole("status")).toHaveTextContent("No box covers frames 100–119, 181–200.");
  });

  it("marks a box that hides nothing because it is outside the trim", () => {
    renderPanel(withTrackAdded(EMPTY_REDACTION, [10, 10, 40, 40], 150, [120, 180]), 150, [300, 400]);

    expect(screen.getByText("Outside the trim (300–400): this box hides nothing.")).toBeInTheDocument();
  });

  it("says nothing about uncovered frames while there are no boxes or every frame is covered", () => {
    renderPanel(ONE_BOX);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("moves the start and the end of a box to the frame on screen", () => {
    renderPanel(ONE_BOX, 40);

    fireEvent.click(screen.getByRole("button", { name: en["cctv.redact.endHere"] }));
    expect(screen.getByText("Box 1: frames 0–40 · keyframes: 1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: en["cctv.redact.startHere"] }));
    expect(screen.getByText("Box 1: frames 40–40 · keyframes: 1")).toBeInTheDocument();
  });

  it("jumps to the first frame of a box and deletes it", () => {
    const onShowFrame = renderPanel(ONE_BOX);

    fireEvent.click(screen.getByRole("button", { name: en["cctv.redact.showStart"] }));
    expect(onShowFrame).toHaveBeenCalledWith(0);
    fireEvent.click(screen.getByRole("button", { name: "Remove box 1" }));
    expect(screen.getByText(en["cctv.redact.empty"])).toBeInTheDocument();
  });

  it("only removes a keyframe that is on this frame and is not the last one", () => {
    renderPanel(ONE_BOX, 5);

    expect(screen.getByRole("button", { name: en["cctv.redact.removeKeyframe"] })).toBeDisabled();
  });
});

describe("redacted copy in the job setup", () => {
  it("hides the filters, the on-screen text tools and the case form", () => {
    renderSetup();

    openRedaction();

    expect(screen.getByRole("group", { name: en["cctv.box.surfaceRedact"] })).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: en["cctv.steps.legend"] })).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: en["cctv.osd.none"] })).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: en["cctv.trim.toggle"] })).toBeInTheDocument();
  });

  it("blocks Start until a box is drawn, then sends the boxes over the whole clip", () => {
    const onSubmit = renderSetup();
    openRedaction();
    const start = screen.getByRole("button", { name: en["cctv.start"] });

    expect(start).toBeDisabled();
    expect(screen.getByText(en["cctv.redact.blocked.noBoxes"])).toBeInTheDocument();

    drag(screen.getByRole("group", { name: en["cctv.box.surfaceRedact"] }), [100, 100], [300, 200]);
    fireEvent.click(screen.getByRole("radio", { name: en["cctv.redact.style.blur"] }));
    fireEvent.click(start);

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    const request = onSubmit.mock.calls[0][0];
    expect(request).toMatchObject({ task: "redact", preset: null, steps: [], osdBoxes: [], noOsd: false, trim: null });
    expect(request.redaction).toEqual({
      style: "blur",
      tracks: [{ firstFrame: 0, lastFrame: 749, keyframes: [{ frame: 0, box: [50, 100, 100, 100] }] }],
    });
  });
});
