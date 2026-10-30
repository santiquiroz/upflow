import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useEditorShortcuts } from "./useEditorShortcuts";

function ShortcutHarness({
  onUndo,
  onBrushSmaller,
  onBrushBigger,
}: {
  onUndo: () => void;
  onBrushSmaller: () => void;
  onBrushBigger: () => void;
}) {
  useEditorShortcuts({
    onUndo,
    onBrushSmaller,
    onBrushBigger,
    enabled: true,
  });
  return <input aria-label="Prompt" />;
}

describe("useEditorShortcuts", () => {
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
});
