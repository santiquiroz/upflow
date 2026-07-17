import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EncoderPicker, formatEncoderSummary } from "./EncoderPicker";

describe("EncoderPicker", () => {
  it("renders the two encoder options as radios", () => {
    render(<EncoderPicker value="software" onChange={vi.fn()} />);

    expect(screen.getByRole("radio", { name: /Software/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Auto \(GPU\)/ })).toBeInTheDocument();
  });

  it("marks Software as selected by default", () => {
    render(<EncoderPicker value="software" onChange={vi.fn()} />);

    expect(screen.getByRole("radio", { name: /Software/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /Auto \(GPU\)/ })).not.toBeChecked();
  });

  it("calls onChange with 'auto' when the GPU option is selected", () => {
    const onChange = vi.fn();
    render(<EncoderPicker value="software" onChange={onChange} />);

    fireEvent.click(screen.getByRole("radio", { name: /Auto \(GPU\)/ }));

    expect(onChange).toHaveBeenCalledWith("auto");
  });

  it("calls onChange with 'software' when the software option is selected", () => {
    const onChange = vi.fn();
    render(<EncoderPicker value="auto" onChange={onChange} />);

    fireEvent.click(screen.getByRole("radio", { name: /Software/ }));

    expect(onChange).toHaveBeenCalledWith("software");
  });

  it("reflects a non-default selection in the checked radio", () => {
    render(<EncoderPicker value="auto" onChange={vi.fn()} />);

    expect(screen.getByRole("radio", { name: /Auto \(GPU\)/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /Software/ })).not.toBeChecked();
  });
});

describe("formatEncoderSummary", () => {
  it("maps each encoder value to its human label", () => {
    expect(formatEncoderSummary("software")).toBe("Software (x264/x265)");
    expect(formatEncoderSummary("auto")).toBe("Auto (GPU)");
  });
});
