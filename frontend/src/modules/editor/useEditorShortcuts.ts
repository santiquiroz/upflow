import { useEffect } from "react";

interface UseEditorShortcutsOptions {
  onUndo: () => void;
  onBrushSmaller: () => void;
  onBrushBigger: () => void;
  enabled: boolean;
}

function isTextEntryTarget(target: EventTarget | null): boolean {
  return target instanceof Element && target.closest("input, textarea, [contenteditable]") !== null;
}

export function useEditorShortcuts({
  onUndo,
  onBrushSmaller,
  onBrushBigger,
  enabled,
}: UseEditorShortcutsOptions) {
  useEffect(() => {
    if (!enabled) return;

    function handleKeyDown(event: KeyboardEvent) {
      if (isTextEntryTarget(event.target)) return;

      if (
        event.key.toLowerCase() === "z" &&
        (event.ctrlKey || event.metaKey) &&
        !event.altKey &&
        !event.shiftKey
      ) {
        event.preventDefault();
        onUndo();
        return;
      }

      if (event.key === "[") {
        event.preventDefault();
        onBrushSmaller();
      } else if (event.key === "]") {
        event.preventDefault();
        onBrushBigger();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [enabled, onBrushBigger, onBrushSmaller, onUndo]);
}
