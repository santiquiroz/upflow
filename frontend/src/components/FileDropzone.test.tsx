import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FileDropzone } from "./FileDropzone";

function renderDropzone(files: File[] = [], onFilesSelected = vi.fn(), multiple = true) {
  render(
    <FileDropzone
      inputId="test-file-input"
      accept="image/*,.tif,.tiff"
      multiple={multiple}
      files={files}
      emptyLabel="Drop a file here"
      formatsHint="PNG, TIFF"
      onFilesSelected={onFilesSelected}
    />,
  );
  return onFilesSelected;
}

function makeFile(name: string): File {
  return new File(["binary"], name, { type: "image/png" });
}

describe("FileDropzone", () => {
  it("shows the empty label and the formats hint before a file is chosen", () => {
    renderDropzone();

    expect(screen.getByText("Drop a file here")).toBeInTheDocument();
    expect(screen.getByText("PNG, TIFF")).toBeInTheDocument();
  });

  it("shows the file name when one file is chosen", () => {
    renderDropzone([makeFile("scan.tif")]);

    expect(screen.getByText("scan.tif")).toBeInTheDocument();
  });

  it("shows the count when several files are chosen", () => {
    renderDropzone([makeFile("a.png"), makeFile("b.png")]);

    expect(screen.getByText("2 files selected")).toBeInTheDocument();
  });

  it("wires the input with the requested id, accept list and multiple flag", () => {
    renderDropzone([], vi.fn(), false);

    const input = document.getElementById("test-file-input") as HTMLInputElement;
    expect(input).toHaveAttribute("accept", "image/*,.tif,.tiff");
    expect(input.multiple).toBe(false);
  });

  it("reports the files picked in the browse dialog", () => {
    const onFilesSelected = renderDropzone();
    const input = document.getElementById("test-file-input") as HTMLInputElement;
    const picked = makeFile("photo.png");

    fireEvent.change(input, { target: { files: [picked] } });

    expect(onFilesSelected).toHaveBeenCalledWith([picked]);
  });

  it("reports dropped files", () => {
    const onFilesSelected = renderDropzone();
    const dropped = makeFile("dropped.png");

    fireEvent.drop(screen.getByText("Drop a file here"), { dataTransfer: { files: [dropped] } });

    expect(onFilesSelected).toHaveBeenCalledWith([dropped]);
  });

  it("keeps only the first dropped file when multiple is off", () => {
    const onFilesSelected = renderDropzone([], vi.fn(), false);
    const first = makeFile("first.png");

    fireEvent.drop(screen.getByText("Drop a file here"), {
      dataTransfer: { files: [first, makeFile("second.png")] },
    });

    expect(onFilesSelected).toHaveBeenCalledWith([first]);
  });

  it("ignores an empty drop", () => {
    const onFilesSelected = renderDropzone();

    fireEvent.drop(screen.getByText("Drop a file here"), { dataTransfer: { files: [] } });

    expect(onFilesSelected).not.toHaveBeenCalled();
  });
});
