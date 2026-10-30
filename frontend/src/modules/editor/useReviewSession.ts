import { useCallback, useEffect, useState } from "react";
import type { EraseZone, GenerationJob } from "../../lib/apiTypes";
import { replaceZone } from "./zones";

export interface ReviewSession {
  eraseMask: ImageData;
  protectMaskToken?: string;
  maskGrowPx: number;
  maskFeatherPx: number;
  zones: EraseZone[] | null;
  resultUrl: string | null;
}

function adoptJob(session: ReviewSession | null, job: GenerationJob | undefined): ReviewSession | null {
  const zones = job?.status === "completed" ? job.eraseZones : null;
  if (!session || session.zones !== null || !zones?.length) return session;
  return { ...session, zones, resultUrl: job?.downloadUrl ?? null };
}

export function useReviewSession(job: GenerationJob | undefined) {
  const [session, setSession] = useState<ReviewSession | null>(null);
  const [hoveredZone, setHoveredZone] = useState<number | null>(null);

  useEffect(() => {
    setSession((current) => adoptJob(current, job));
  }, [job]);

  const start = useCallback((eraseMask: ImageData, protectMaskToken?: string) => {
    setSession({ eraseMask, protectMaskToken, maskGrowPx: 0, maskFeatherPx: 0, zones: null, resultUrl: null });
    setHoveredZone(null);
  }, []);

  const clear = useCallback(() => {
    setSession(null);
    setHoveredZone(null);
  }, []);

  const updateZone = useCallback((zone: EraseZone, resultUrl: string | null) => {
    setSession((current) => current?.zones
      ? { ...current, zones: replaceZone(current.zones, zone), resultUrl: resultUrl ?? current.resultUrl }
      : current);
  }, []);

  return { session, zones: session?.zones ?? null, hoveredZone, setHoveredZone, start, clear, updateZone };
}
