import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AudioEnhanceControls } from "./AudioEnhanceControls";

describe("AudioEnhanceControls", () => {
  it("enables every option and marks the active one when keepAudio is on", () => {
    render(<AudioEnhanceControls value={null} onChange={vi.fn()} keepAudio={true} />);

    expect(screen.getByRole("button", { name: "Off" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "Off" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "RNNoise" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "DeepFilterNet" })).not.toBeDisabled();
  });

  it("marks the selected mode as active", () => {
    render(<AudioEnhanceControls value="deepfilter" onChange={vi.fn()} keepAudio={true} />);

    expect(screen.getByRole("button", { name: "DeepFilterNet" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Off" })).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onChange with the picked mode", () => {
    const onChange = vi.fn();
    render(<AudioEnhanceControls value={null} onChange={onChange} keepAudio={true} />);

    screen.getByRole("button", { name: "RNNoise" }).click();

    expect(onChange).toHaveBeenCalledWith("rnnoise");
  });

  it("calls onChange with null when Off is picked", () => {
    const onChange = vi.fn();
    render(<AudioEnhanceControls value="rnnoise" onChange={onChange} keepAudio={true} />);

    screen.getByRole("button", { name: "Off" }).click();

    expect(onChange).toHaveBeenCalledWith(null);
  });

  it("disables every option and explains why when keepAudio is off", () => {
    render(<AudioEnhanceControls value={null} onChange={vi.fn()} keepAudio={false} />);

    expect(screen.getByRole("button", { name: "Off" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "RNNoise" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "DeepFilterNet" })).toBeDisabled();
    expect(screen.getByText(/keep original audio/i)).toBeInTheDocument();
  });
});
