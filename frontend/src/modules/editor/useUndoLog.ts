import { useCallback, useRef, useState } from "react";

export type UndoKind = "mask" | "base";

export interface UndoAction {
  id: number;
  kind: UndoKind;
}

// Registro único de lo que Ctrl+Z puede deshacer, en orden. Cada acción de
// máscara lleva el id de su entrada en el historial, así se quita por id y no por posición.
export function useUndoLog() {
  const [actions, setActions] = useState<UndoAction[]>([]);
  const lastIdRef = useRef(0);

  const record = useCallback((kind: UndoKind): number => {
    lastIdRef.current += 1;
    const id = lastIdRef.current;
    setActions((previous) => [...previous, { id, kind }]);
    return id;
  }, []);

  const remove = useCallback((id: number) => {
    setActions((previous) => previous.filter((action) => action.id !== id));
  }, []);

  const keepMasks = useCallback((keepIds: ReadonlySet<number>) => {
    setActions((previous) =>
      previous.filter((action) => action.kind === "base" || keepIds.has(action.id)),
    );
  }, []);

  const clear = useCallback(() => setActions([]), []);

  return {
    last: actions[actions.length - 1] ?? null,
    record,
    remove,
    keepMasks,
    clear,
  };
}
