import { useQuery } from "@tanstack/react-query";
import { useDeferredValue, useMemo, useState } from "react";
import type { RestoreAnalysis } from "../../../lib/restoreApiTypes";
import type { BinaryMask, BrushStroke } from "../../editor/maskCanvas";
import { largeHoleCount } from "./damageHoles";
import {
  composeDamageMask,
  maskCoverage,
  maskReviewKey,
  needsMaskReview,
  type MaskSettings,
  type ProbabilityMap,
} from "./damageMask";
import { encodeMaskPng } from "./maskPng";
import { loadProbabilityMap } from "./probabilityMap";
import { versionedUrl } from "./restoreUrls";

export type DamageMapStatus = "none" | "loading" | "ready" | "failed";
export type LoadProbabilityMap = (url: string, signal?: AbortSignal) => Promise<ProbabilityMap>;

export interface DamageMaskInput {
  analysis: RestoreAnalysis;
  revision: number;
  settings: MaskSettings;
  active: boolean;
  loadMap?: LoadProbabilityMap;
}

export interface DamageMaskState {
  map: ProbabilityMap | null;
  mapStatus: DamageMapStatus;
  mask: BinaryMask | null;
  coverage: number;
  largeHoles: number;
  needsReview: boolean;
  reviewed: boolean;
  edited: boolean;
  canUndo: boolean;
  addStroke: (stroke: BrushStroke) => void;
  undo: () => void;
  clear: () => void;
  markReviewed: () => void;
  maskBlob: () => Promise<Blob>;
}

interface EditState {
  photoKey: string;
  strokes: readonly BrushStroke[];
  // Sube con cada trazo, deshacer o limpiar: la revision vale para una mascara exacta.
  editCount: number;
  reviewedKey: string | null;
}

interface MaskStats {
  coverage: number;
  largeHoles: number;
}

function initialEdits(photoKey: string): EditState {
  return { photoKey, strokes: [], editCount: 0, reviewedKey: null };
}

// Otra foto u otra geometria dejan sin sentido los trazos: se empieza de cero.
function currentEdits(state: EditState, photoKey: string): EditState {
  return state.photoKey === photoKey ? state : initialEdits(photoKey);
}

function mapStatusOf(url: string | null, isError: boolean, map: ProbabilityMap | undefined): DamageMapStatus {
  if (url === null) return "none";
  if (isError) return "failed";
  return map === undefined ? "loading" : "ready";
}

// Mientras no hay mapa (cargando o fallo) y nada pintado, vale lo que midio el analisis.
function isMeasurable(status: DamageMapStatus, edited: boolean): boolean {
  return status === "ready" || status === "none" || edited;
}

function measure(mask: BinaryMask | null): MaskStats {
  if (mask === null) {
    return { coverage: 0, largeHoles: 0 };
  }
  return { coverage: maskCoverage(mask), largeHoles: largeHoleCount(mask) };
}

function analysisStats(analysis: RestoreAnalysis): MaskStats {
  return { coverage: analysis.damage.coverage ?? 0, largeHoles: analysis.damage.largeHoles };
}

function mapUrl(analysis: RestoreAnalysis, revision: number, active: boolean): string | null {
  const { probUrl } = analysis.damage;
  return active && probUrl ? versionedUrl(probUrl, revision) : null;
}

export function useDamageMask({
  analysis,
  revision,
  settings,
  active,
  loadMap = loadProbabilityMap,
}: DamageMaskInput): DamageMaskState {
  const photoKey = `${analysis.token}:${revision}`;
  const [stored, setStored] = useState(() => initialEdits(photoKey));
  const edits = currentEdits(stored, photoKey);
  const url = mapUrl(analysis, revision, active);
  const mapQuery = useQuery({
    queryKey: ["restore", "damageMap", url],
    queryFn: ({ signal }) => loadMap(url as string, signal),
    enabled: url !== null,
    staleTime: Infinity,
    gcTime: 0,
    retry: false,
  });
  const map = mapQuery.data ?? null;
  const mapStatus = mapStatusOf(url, mapQuery.isError, mapQuery.data);
  const size = { width: analysis.width, height: analysis.height };
  // Primitivos: el deslizador se mueve sin recomponer 12 MP en cada paso.
  const sensitivity = useDeferredValue(settings.sensitivity);
  const growPx = useDeferredValue(settings.growPx);

  const mask = useMemo(
    () => (active ? composeDamageMask(map, size, { sensitivity, growPx }, edits.strokes) : null),
    [active, map, size.width, size.height, sensitivity, growPx, edits.strokes],
  );
  const edited = edits.strokes.length > 0;
  const measurable = isMeasurable(mapStatus, edited);
  const measured = useMemo(() => (measurable ? measure(mask) : null), [measurable, mask]);
  const stats = measured ?? analysisStats(analysis);
  const currentKey = maskReviewKey(settings, edits.editCount);

  function update(change: (current: EditState) => EditState): void {
    setStored((previous) => change(currentEdits(previous, photoKey)));
  }

  function withStrokes(current: EditState, strokes: readonly BrushStroke[]): EditState {
    return { ...current, strokes, editCount: current.editCount + 1 };
  }

  return {
    map,
    mapStatus,
    mask,
    coverage: stats.coverage,
    largeHoles: stats.largeHoles,
    needsReview: active && needsMaskReview({ ...stats, damageOverFaces: analysis.damageOverFaces }),
    reviewed: edits.reviewedKey === currentKey,
    edited,
    canUndo: edited,
    addStroke: (stroke) => update((current) => withStrokes(current, [...current.strokes, stroke])),
    undo: () => update((current) => withStrokes(current, current.strokes.slice(0, -1))),
    clear: () => update((current) => withStrokes(current, [])),
    markReviewed: () => update((current) => ({ ...current, reviewedKey: maskReviewKey(settings, current.editCount) })),
    // Con los valores de ahora, no los diferidos: se sube exactamente lo que se eligio.
    maskBlob: () => encodeMaskPng(composeDamageMask(map, size, settings, edits.strokes)),
  };
}
