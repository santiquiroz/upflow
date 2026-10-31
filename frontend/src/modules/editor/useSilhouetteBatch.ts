import { useRef, useState, type MutableRefObject } from "react";
import type { InitImageResponse } from "../../lib/apiTypes";
import type { EditorPerson } from "../../services/editor";
import { pngBlobToCropMask, type HistoryEntry } from "./editorMaskCanvas";
import type { Choice } from "./peopleState";
import { cropRect, silhouetteCount, silhouettesFor } from "./silhouettes";

export interface SilhouetteProgress {
  done: number;
  total: number;
}

interface SilhouetteSelection {
  people: EditorPerson[];
  choices: Record<number, Choice>;
  protectKept: boolean;
}

interface SilhouetteBatchOptions {
  baseInfo: InitImageResponse | null;
  selection: SilhouetteSelection;
  sequenceRef: MutableRefObject<number>;
  onDone: (entry: HistoryEntry) => void;
}

export function useSilhouetteBatch({ baseInfo, selection, sequenceRef, onDone }: SilhouetteBatchOptions) {
  const [progress, setProgress] = useState<SilhouetteProgress | null>(null);
  const runningRef = useRef(false);

  async function accept() {
    if (!baseInfo || runningRef.current) return;
    const { initImageToken, width, height } = baseInfo;
    const { people, choices, protectKept } = selection;
    const sequence = sequenceRef.current;
    runningRef.current = true;
    setProgress({ done: 0, total: silhouetteCount(people, choices, protectKept) });
    try {
      const result = await silhouettesFor(initImageToken, people, choices, protectKept, {
        toMask: (blob, box) => pngBlobToCropMask(blob, width, height, cropRect(box, width, height)),
        onProgress: (done, total) => setProgress({ done, total }),
      });
      if (sequence !== sequenceRef.current) return;
      onDone(result.entry);
    } finally {
      runningRef.current = false;
      setProgress(null);
    }
  }

  return { progress, accept };
}
