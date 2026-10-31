import { useState } from "react";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import type { GenerationModelSummary } from "../../lib/apiTypes";
import { NO_EDGE, type EdgeAdjust } from "./editorMaskCanvas";
import { supportsEdgeAdjust } from "./eraseJobParams";
import type { EditMode } from "./InpaintModelSelect";

export const EDGE_PREVIEW_DEBOUNCE_MS = 120;

interface EdgeAdjustInput {
  mode: EditMode;
  modelId: string | null;
  models: GenerationModelSummary[];
  photo: object | null;
}

export function useEdgeAdjust({ mode, modelId, models, photo }: EdgeAdjustInput) {
  const [edge, setEdge] = useState<EdgeAdjust>(NO_EDGE);
  const [seenPhoto, setSeenPhoto] = useState(photo);
  if (seenPhoto !== photo) {
    setSeenPhoto(photo);
    setEdge(NO_EDGE);
  }
  const shownEdge = useDebouncedValue(edge, EDGE_PREVIEW_DEBOUNCE_MS);
  const edgeActive = supportsEdgeAdjust(mode, modelId, models);
  return { edge, setEdge, edgeActive, preview: edgeActive ? shownEdge : null };
}
