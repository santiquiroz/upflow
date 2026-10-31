import { useCallback, type Dispatch, type SetStateAction } from "react";
import { withoutEdits, type HistoryEntry } from "./editorMaskCanvas";
import { useUndoLog } from "./useUndoLog";

interface EditorUndoOptions {
  history: HistoryEntry[];
  setHistory: Dispatch<SetStateAction<HistoryEntry[]>>;
  undoBase: () => boolean;
  applying: boolean;
}

// Máscaras y versiones de base comparten un solo registro de Deshacer, en el orden en que pasaron.
export function useEditorUndo({ history, setHistory, undoBase, applying }: EditorUndoOptions) {
  const undoLog = useUndoLog();
  const { actions, record, remove, removeOldest, keepMasks } = undoLog;

  // Toda entrada de máscara pasa por aquí: el registro de Deshacer y el historial nacen juntos.
  function appendHistory(entry: HistoryEntry) {
    const id = record("mask");
    setHistory((previous) => [...previous, { ...entry, id }]);
  }

  function dropHistoryEntry(id: number) {
    remove(id);
    setHistory((previous) => previous.filter((entry) => entry.id !== id));
  }

  // Un cambio de base (retoque o corrección) entra al registro; si el tope descartó una versión, sale la más vieja.
  function recordBase(evicted: boolean) {
    record("base");
    if (evicted) removeOldest("base");
  }

  // Una base que ya no tiene versión que deshacer se descarta del registro y se sigue con la anterior.
  const handleUndo = useCallback(() => {
    if (applying) return;
    for (let index = actions.length - 1; index >= 0; index -= 1) {
      const action = actions[index];
      if (action.kind === "mask") {
        remove(action.id);
        setHistory((previous) => previous.filter((entry) => entry.id !== action.id));
        return;
      }
      remove(action.id);
      if (undoBase()) return;
    }
  }, [actions, remove, undoBase, applying, setHistory]);

  // Limpiar la selección quita las máscaras; las versiones de base siguen en el registro.
  function clearSelection() {
    const kept = withoutEdits(history).flatMap((entry) => (entry.id === undefined ? [] : [entry.id]));
    setHistory(withoutEdits);
    keepMasks(new Set(kept));
  }

  return { undoLog, appendHistory, dropHistoryEntry, recordBase, handleUndo, clearSelection };
}
