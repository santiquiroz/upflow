import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FormatOptionFieldset } from "./FormatOptionFieldset";

const OPTIONS = [
  { value: "flac", label: "FLAC (recommended)", description: "Lossless quality, about 50% smaller than WAV." },
  { value: "wav", label: "WAV", description: "Lossless, uncompressed. Universal compatibility." },
] as const;

describe("FormatOptionFieldset", () => {
  it("renders every option with its description and marks the current value as checked", () => {
    render(<FormatOptionFieldset legend="Output format" name="fmt" options={OPTIONS} value="flac" onChange={vi.fn()} />);

    expect(screen.getByLabelText(/flac/i)).toBeChecked();
    expect(screen.getByLabelText(/^wav$/i)).not.toBeChecked();
    expect(screen.getByText(/50% smaller than WAV/i)).toBeInTheDocument();
    expect(screen.getByText(/universal compatibility/i)).toBeInTheDocument();
  });

  it("does not let a description mentioning another option's label cause an ambiguous match", () => {
    render(<FormatOptionFieldset legend="Output format" name="fmt" options={OPTIONS} value="wav" onChange={vi.fn()} />);

    expect(screen.getByLabelText(/^wav$/i)).toBeChecked();
  });

  it("calls onChange with the clicked option's value", () => {
    const onChange = vi.fn();
    render(<FormatOptionFieldset legend="Output format" name="fmt" options={OPTIONS} value="flac" onChange={onChange} />);

    fireEvent.click(screen.getByLabelText(/^wav$/i));

    expect(onChange).toHaveBeenCalledWith("wav");
  });
});
