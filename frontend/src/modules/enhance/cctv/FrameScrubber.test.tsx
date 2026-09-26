import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import type { CctvFrameIndex } from "../../../services/cctv";
import { ANALYSIS } from "./cctvFixtures";
import { FrameScrubber } from "./FrameScrubber";

const INDEX = ANALYSIS.frameIndex as CctvFrameIndex;

function Harness({ onFrameChange }: { onFrameChange: (frame: number) => void }) {
  const [frame, setFrame] = useState(0);
  function change(next: number): void {
    setFrame(next);
    onFrameChange(next);
  }
  return (
    <FrameScrubber
      token="tok 1"
      frameCount={750}
      frame={frame}
      index={INDEX}
      displaySize={{ width: 1920, height: 1080 }}
      onFrameChange={change}
      overlay={<span>overlay</span>}
    />
  );
}

function renderScrubber() {
  const onFrameChange = vi.fn();
  render(<Harness onFrameChange={onFrameChange} />);
  return onFrameChange;
}

function frameImage(): HTMLImageElement {
  return screen.getByRole("img") as HTMLImageElement;
}

describe("FrameScrubber", () => {
  it("asks the backend to decode the frame, shown at the real aspect ratio", () => {
    renderScrubber();

    expect(frameImage().getAttribute("src")).toBe("/api/v1/video/cctv/tok%201/preview?frame=0");
    expect(frameImage().parentElement).toHaveStyle({ aspectRatio: "1920 / 1080" });
    expect(screen.getByText("overlay")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(en["cctv.frames.loading"]);
  });

  it("moves through the frames and requests the frame once the slider settles", async () => {
    const onFrameChange = renderScrubber();

    fireEvent.change(screen.getByRole("slider", { name: en["cctv.frames.slider"] }), { target: { value: "11" } });

    expect(onFrameChange).toHaveBeenLastCalledWith(11);
    expect(screen.getByText("Frame 11 (0–749)")).toBeInTheDocument();
    expect(screen.getByText("≈ 00:00:01.600")).toBeInTheDocument();
    await waitFor(() => expect(frameImage().getAttribute("src")).toBe("/api/v1/video/cctv/tok%201/preview?frame=11"));
  });

  it("steps one frame at a time and never leaves the recording", () => {
    const onFrameChange = renderScrubber();

    expect(screen.getByRole("button", { name: en["cctv.frames.previous"] })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: en["cctv.frames.next"] }));
    expect(onFrameChange).toHaveBeenLastCalledWith(1);

    fireEvent.change(screen.getByRole("spinbutton", { name: en["cctv.frames.number"] }), { target: { value: "9000" } });

    expect(onFrameChange).toHaveBeenLastCalledWith(749);
    expect(screen.getByRole("button", { name: en["cctv.frames.next"] })).toBeDisabled();
  });

  it("says when the frame can't be decoded", () => {
    renderScrubber();

    fireEvent.error(frameImage());

    expect(screen.getByRole("alert")).toHaveTextContent(en["cctv.frames.loadFailed"]);
  });

  it("drops the loading note once the frame arrives", () => {
    renderScrubber();

    fireEvent.load(frameImage());

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});
