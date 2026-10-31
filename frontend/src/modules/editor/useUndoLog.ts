import { useCallback, useRef, useState } from "react";

export type UndoKind = "mask" | "base";

export interface UndoAction {
  id: number;
  kind: UndoKind;
}

// El id de cada acción la identifica aunque haya otras de su mismo tipo.
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

  const removeOldest = useCallback((kind: UndoKind) => {
    setActions((previous) => {
      const oldest = previous.find((action) => action.kind === kind);
      return previous.filter((action) => action !== oldest);
    });
  }, []);

  const keepMasks = useCallback((keepIds: ReadonlySet<number>) => {
    setActions((previous) =>
      previous.filter((action) => action.kind === "base" || keepIds.has(action.id)),
    );
  }, []);

  const clear = useCallback(() => setActions([]), []);

  return {
    actions,
    last: actions[actions.length - 1] ?? null,
    record,
    remove,
    removeOldest,
    keepMasks,
    clear,
  };
}
