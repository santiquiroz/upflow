import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RuntimePicker, formatRuntimeSummary } from "./RuntimePicker";

describe("RuntimePicker", () => {
  it("renders the three runtime options as radios", async () => {
    render(<RuntimePicker value="auto" onChange={vi.fn()} />);

    expect(screen.getByRole("radio", { name: /Auto/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /NCNN Vulkan/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /ONNX DirectML/ })).toBeInTheDocument();
  });

  it("marks the Auto option as selected when the value is auto", () => {
    render(<RuntimePicker value="auto" onChange={vi.fn()} />);

    expect(screen.getByRole("radio", { name: /Auto/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /NCNN Vulkan/ })).not.toBeChecked();
    expect(screen.getByRole("radio", { name: /ONNX DirectML/ })).not.toBeChecked();
  });

  it("describes the Auto option as the best backend for the device", () => {
    render(<RuntimePicker value="auto" onChange={vi.fn()} />);

    expect(screen.getByRole("radio", { name: /Auto/ }).closest("label")).toHaveTextContent(
      /best backend for your device/i,
    );
  });

  it("calls onChange with 'onnx' when the ONNX DirectML option is selected", () => {
    const onChange = vi.fn();
    render(<RuntimePicker value="auto" onChange={onChange} />);

    fireEvent.click(screen.getByRole("radio", { name: /ONNX DirectML/ }));

    expect(onChange).toHaveBeenCalledWith("onnx");
  });

  it("calls onChange with 'ncnn' when the NCNN Vulkan option is selected", () => {
    const onChange = vi.fn();
    render(<RuntimePicker value="onnx" onChange={onChange} />);

    fireEvent.click(screen.getByRole("radio", { name: /NCNN Vulkan/ }));

    expect(onChange).toHaveBeenCalledWith("ncnn");
  });

  it("reflects a non-default selection in the checked radio", () => {
    render(<RuntimePicker value="onnx" onChange={vi.fn()} />);

    expect(screen.getByRole("radio", { name: /ONNX DirectML/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /Auto/ })).not.toBeChecked();
  });
});

describe("formatRuntimeSummary", () => {
  it("maps each backend value to its human label", () => {
    expect(formatRuntimeSummary("auto")).toBe("Auto");
    expect(formatRuntimeSummary("ncnn")).toBe("NCNN Vulkan");
    expect(formatRuntimeSummary("onnx")).toBe("ONNX DirectML");
  });
});
