import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { isTextEntry, useEditorShortcuts } from "./useEditorShortcuts";

function ShortcutHarness({
  onUndo,
  onBrushSmaller,
  onBrushBigger,
  enabled = true,
}: {
  onUndo: () => void;
  onBrushSmaller: () => void;
  onBrushBigger: () => void;
  enabled?: boolean;
}) {
  useEditorShortcuts({
    onUndo,
    onBrushSmaller,
    onBrushBigger,
    enabled,
  });
  return (
    <div>
      <input aria-label="Prompt" />
      <textarea aria-label="Description" />
      <div aria-label="Editable text" contentEditable="true" />
      <input type="range" aria-label="Brush size" />
    </div>
  );
}

describe("useEditorShortcuts", () => {
  it("recognizes text-entry targets but not sliders", () => {
    expect(isTextEntry(document.createElement("textarea"))).toBe(true);
    const editable = document.createElement("div");
    editable.setAttribute("contenteditable", "true");
    expect(isTextEntry(editable)).toBe(true);
    expect(isTextEntry(document.createElement("input"))).toBe(true);
    const range = document.createElement("input");
    range.type = "range";
    expect(isTextEntry(range)).toBe(false);
    const invalidType = document.createElement("input");
    invalidType.setAttribute("type", "custom");
    expect(isTextEntry(invalidType)).toBe(false);
  });

  it("undoes with Ctrl+Z and Cmd+Z", () => {
    const onUndo = vi.fn();
    render(
      <ShortcutHarness
        onUndo={onUndo}
        onBrushSmaller={vi.fn()}
        onBrushBigger={vi.fn()}
      />,
    );

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    fireEvent.keyDown(window, { key: "z", metaKey: true });

    expect(onUndo).toHaveBeenCalledTimes(2);
  });

  it("changes brush size with [ and ]", () => {
    const onBrushSmaller = vi.fn();
    const onBrushBigger = vi.fn();
    render(
      <ShortcutHarness
        onUndo={vi.fn()}
        onBrushSmaller={onBrushSmaller}
        onBrushBigger={onBrushBigger}
      />,
    );

    fireEvent.keyDown(window, { key: "[" });
    fireEvent.keyDown(window, { key: "]" });

    expect(onBrushSmaller).toHaveBeenCalledOnce();
    expect(onBrushBigger).toHaveBeenCalledOnce();
  });

  it("ignores shortcuts while an input has focus", () => {
    const onUndo = vi.fn();
    const onBrushSmaller = vi.fn();
    const onBrushBigger = vi.fn();
    render(
      <ShortcutHarness
        onUndo={onUndo}
        onBrushSmaller={onBrushSmaller}
        onBrushBigger={onBrushBigger}
      />,
    );

    const input = screen.getByRole("textbox", { name: "Prompt" });
    fireEvent.keyDown(input, { key: "z", ctrlKey: true });
    fireEvent.keyDown(input, { key: "[" });
    fireEvent.keyDown(input, { key: "]" });

    expect(onUndo).not.toHaveBeenCalled();
    expect(onBrushSmaller).not.toHaveBeenCalled();
    expect(onBrushBigger).not.toHaveBeenCalled();
  });

  it("ignores shortcuts while a textarea has focus", () => {
    const onUndo = vi.fn();
    const onBrushSmaller = vi.fn();
    const onBrushBigger = vi.fn();
    render(
      <ShortcutHarness
        onUndo={onUndo}
        onBrushSmaller={onBrushSmaller}
        onBrushBigger={onBrushBigger}
      />,
    );

    const textarea = screen.getByRole("textbox", { name: "Description" });
    fireEvent.keyDown(textarea, { key: "z", ctrlKey: true });
    fireEvent.keyDown(textarea, { key: "[" });
    fireEvent.keyDown(textarea, { key: "]" });

    expect(onUndo).not.toHaveBeenCalled();
    expect(onBrushSmaller).not.toHaveBeenCalled();
    expect(onBrushBigger).not.toHaveBeenCalled();
  });

  it("ignores shortcuts while a contenteditable has focus", () => {
    const onUndo = vi.fn();
    const onBrushSmaller = vi.fn();
    const onBrushBigger = vi.fn();
    render(
      <ShortcutHarness
        onUndo={onUndo}
        onBrushSmaller={onBrushSmaller}
        onBrushBigger={onBrushBigger}
      />,
    );

    const editable = screen.getByLabelText("Editable text");
    fireEvent.keyDown(editable, { key: "z", ctrlKey: true });
    fireEvent.keyDown(editable, { key: "[" });
    fireEvent.keyDown(editable, { key: "]" });

    expect(onUndo).not.toHaveBeenCalled();
    expect(onBrushSmaller).not.toHaveBeenCalled();
    expect(onBrushBigger).not.toHaveBeenCalled();
  });

  it("handles shortcuts when a range input has focus", () => {
    const onBrushSmaller = vi.fn();
    render(
      <ShortcutHarness
        onUndo={vi.fn()}
        onBrushSmaller={onBrushSmaller}
        onBrushBigger={vi.fn()}
      />,
    );

    fireEvent.keyDown(screen.getByRole("slider", { name: "Brush size" }), { key: "[" });

    expect(onBrushSmaller).toHaveBeenCalledOnce();
  });

  it("removes the key listener on unmount", () => {
    const onUndo = vi.fn();
    const onBrushSmaller = vi.fn();
    const onBrushBigger = vi.fn();
    const { unmount } = render(
      <ShortcutHarness
        onUndo={onUndo}
        onBrushSmaller={onBrushSmaller}
        onBrushBigger={onBrushBigger}
      />,
    );
    unmount();

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    fireEvent.keyDown(window, { key: "[" });
    fireEvent.keyDown(window, { key: "]" });

    expect(onUndo).not.toHaveBeenCalled();
    expect(onBrushSmaller).not.toHaveBeenCalled();
    expect(onBrushBigger).not.toHaveBeenCalled();
  });

  it("does not register shortcuts when disabled", () => {
    const onUndo = vi.fn();
    const onBrushSmaller = vi.fn();
    const onBrushBigger = vi.fn();
    render(
      <ShortcutHarness
        onUndo={onUndo}
        onBrushSmaller={onBrushSmaller}
        onBrushBigger={onBrushBigger}
        enabled={false}
      />,
    );

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    fireEvent.keyDown(window, { key: "[" });
    fireEvent.keyDown(window, { key: "]" });

    expect(onUndo).not.toHaveBeenCalled();
    expect(onBrushSmaller).not.toHaveBeenCalled();
    expect(onBrushBigger).not.toHaveBeenCalled();
  });
});
