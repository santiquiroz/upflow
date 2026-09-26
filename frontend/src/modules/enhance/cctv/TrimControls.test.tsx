import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import type { CctvFrameIndex } from "../../../services/cctv";
import { ANALYSIS } from "./cctvFixtures";
import type { TrimRange } from "./cctvFrames";
import { TrimControls } from "./TrimControls";

const INDEX = ANALYSIS.frameIndex as CctvFrameIndex;

function Harness({ currentFrame = 0, onChange }: { currentFrame?: number; onChange: (trim: TrimRange | null) => void }) {
  const [trim, setTrim] = useState<TrimRange | null>(null);
  function change(next: TrimRange | null): void {
    setTrim(next);
    onChange(next);
  }
  return <TrimControls frameCount={750} currentFrame={currentFrame} index={INDEX} trim={trim} onChange={change} />;
}

function renderTrim(currentFrame = 0) {
  const onChange = vi.fn();
  render(<Harness currentFrame={currentFrame} onChange={onChange} />);
  return onChange;
}

function enableTrim(): void {
  fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.trim.toggle"] }));
}

describe("TrimControls", () => {
  it("starts with the whole recording when turned on", () => {
    const onChange = renderTrim();

    enableTrim();

    expect(onChange).toHaveBeenLastCalledWith([0, 749]);
    expect(screen.getByLabelText(en["cctv.trim.start"])).toHaveValue(0);
    expect(screen.getByLabelText(en["cctv.trim.end"])).toHaveValue(749);
    expect(screen.getByText("750 frames")).toBeInTheDocument();
  });

  it("sends no trim when turned off again", () => {
    const onChange = renderTrim();
    enableTrim();

    enableTrim();

    expect(onChange).toHaveBeenLastCalledWith(null);
    expect(screen.queryByLabelText(en["cctv.trim.start"])).not.toBeInTheDocument();
  });

  it("shows the timecode next to each frame number", () => {
    renderTrim();
    enableTrim();

    fireEvent.change(screen.getByLabelText(en["cctv.trim.start"]), { target: { value: "11" } });

    expect(screen.getByText("≈ 00:00:01.600")).toBeInTheDocument();
  });

  it("takes the first and last frame from the frame on screen", () => {
    const onChange = renderTrim(120);
    enableTrim();

    fireEvent.click(screen.getByRole("button", { name: en["cctv.trim.start.useCurrent"] }));

    expect(onChange).toHaveBeenLastCalledWith([120, 749]);

    fireEvent.click(screen.getByRole("button", { name: en["cctv.trim.end.useCurrent"] }));

    expect(onChange).toHaveBeenLastCalledWith([120, 120]);
    expect(screen.getByText("1 frame")).toBeInTheDocument();
  });

  it("explains a reversed or out-of-range trim", () => {
    renderTrim();
    enableTrim();

    fireEvent.change(screen.getByLabelText(en["cctv.trim.end"]), { target: { value: "900" } });

    expect(screen.getByRole("alert")).toHaveTextContent(en["cctv.trim.invalid"]);
  });
});
