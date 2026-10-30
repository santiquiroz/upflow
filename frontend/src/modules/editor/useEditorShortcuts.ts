import { useEffect } from "react";

interface UseEditorShortcutsOptions {
  onUndo: () => void;
  onBrushSmaller: () => void;
  onBrushBigger: () => void;
  enabled: boolean;
}

const TEXT_INPUT_TYPES = new Set(["text", "search", "email", "url", "tel", "password", "number"]);

export function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  if (target.closest("textarea")) return true;

  const editable = target.closest("[contenteditable]");
  if (editable) return editable.getAttribute("contenteditable")?.toLowerCase() !== "false";

  const input = target.closest("input");
  if (!input) return false;
  const type = input.getAttribute("type");
  return type === null || TEXT_INPUT_TYPES.has(type.toLowerCase());
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
      if (isTextEntry(event.target)) return;

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
