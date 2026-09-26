import type { CctvBox, CctvRedactionRequest, CctvRedactionStyle } from "../../../services/cctv";
import type { TrimRange } from "./cctvFrames";

export interface RedactionKeyframe {
  frame: number;
  box: CctvBox;
}

export interface RedactionTrack {
  id: number;
  firstFrame: number;
  lastFrame: number;
  keyframes: readonly RedactionKeyframe[];
}

export interface RedactionChoice {
  style: CctvRedactionStyle;
  tracks: readonly RedactionTrack[];
  nextId: number;
}

export interface ActiveBox {
  id: number;
  number: number;
  box: CctvBox;
  isKeyframe: boolean;
}

// El relleno va primero y por defecto: es el unico estilo que no deja nada que recuperar.
export const REDACTION_STYLES: readonly CctvRedactionStyle[] = ["fill", "pixelate", "blur"];
// Igual que redaction.MAX_TRACKS y MAX_KEYFRAMES del backend.
export const MAX_REDACTION_TRACKS = 32;
export const MAX_REDACTION_KEYFRAMES = 256;

export const EMPTY_REDACTION: RedactionChoice = { style: "fill", tracks: [], nextId: 1 };

// Una caja nueva tapa todo lo que la copia va a tener: el recorte, o el video entero.
export function redactionSpan(trim: TrimRange | null, frameCount: number): TrimRange {
  return trim ?? [0, Math.max(0, frameCount - 1)];
}

// Mismo redondeo que redaction.round_half_up: la vista previa y la copia ponen la caja en el mismo pixel.
function roundHalfUp(value: number): number {
  return Math.floor(value + 0.5);
}

function lerpBox(start: CctvBox, end: CctvBox, fraction: number): CctvBox {
  const [x, y, w, h] = start.map((value, index) => roundHalfUp(value + (end[index] - value) * fraction));
  return [x, y, w, h];
}

function covers(track: RedactionTrack, frame: number): boolean {
  return track.firstFrame <= frame && frame <= track.lastFrame;
}

function heldOrBetween(keyframes: readonly RedactionKeyframe[], frame: number): CctvBox {
  const first = keyframes[0];
  const last = keyframes[keyframes.length - 1];
  if (frame <= first.frame) {
    return first.box;
  }
  if (frame >= last.frame) {
    return last.box;
  }
  const after = keyframes.findIndex((keyframe) => keyframe.frame > frame);
  const before = keyframes[after - 1];
  const fraction = (frame - before.frame) / (keyframes[after].frame - before.frame);
  return lerpBox(before.box, keyframes[after].box, fraction);
}

export function boxAt(track: RedactionTrack, frame: number): CctvBox | null {
  return covers(track, frame) ? heldOrBetween(track.keyframes, frame) : null;
}

export function hasKeyframeAt(track: RedactionTrack, frame: number): boolean {
  return track.keyframes.some((keyframe) => keyframe.frame === frame);
}

export function activeBoxes(choice: RedactionChoice, frame: number): ActiveBox[] {
  return choice.tracks.flatMap((track, position) => {
    const box = boxAt(track, frame);
    return box ? [{ id: track.id, number: position + 1, box, isKeyframe: hasKeyframeAt(track, frame) }] : [];
  });
}

function mapTrack(choice: RedactionChoice, id: number, update: (track: RedactionTrack) => RedactionTrack): RedactionChoice {
  return { ...choice, tracks: choice.tracks.map((track) => (track.id === id ? update(track) : track)) };
}

export function withTrackAdded(choice: RedactionChoice, box: CctvBox, frame: number, span: TrimRange): RedactionChoice {
  if (choice.tracks.length >= MAX_REDACTION_TRACKS) {
    return choice;
  }
  const track: RedactionTrack = { id: choice.nextId, firstFrame: span[0], lastFrame: span[1], keyframes: [{ frame, box }] };
  return { ...choice, tracks: [...choice.tracks, track], nextId: choice.nextId + 1 };
}

export function withTrackRemoved(choice: RedactionChoice, id: number): RedactionChoice {
  return { ...choice, tracks: choice.tracks.filter((track) => track.id !== id) };
}

function keyframesWith(keyframes: readonly RedactionKeyframe[], keyframe: RedactionKeyframe): RedactionKeyframe[] {
  const others = keyframes.filter((current) => current.frame !== keyframe.frame);
  return [...others, keyframe].sort((a, b) => a.frame - b.frame);
}

function canAddKeyframe(track: RedactionTrack, frame: number): boolean {
  return hasKeyframeAt(track, frame) || track.keyframes.length < MAX_REDACTION_KEYFRAMES;
}

export function withKeyframe(choice: RedactionChoice, id: number, frame: number, box: CctvBox): RedactionChoice {
  return mapTrack(choice, id, (track) =>
    canAddKeyframe(track, frame) ? { ...track, keyframes: keyframesWith(track.keyframes, { frame, box }) } : track,
  );
}

// La ultima keyframe no se quita: una caja sin keyframes no sabria donde tapar.
export function withKeyframeRemoved(choice: RedactionChoice, id: number, frame: number): RedactionChoice {
  return mapTrack(choice, id, (track) => {
    const keyframes = track.keyframes.filter((keyframe) => keyframe.frame !== frame);
    return keyframes.length > 0 ? { ...track, keyframes } : track;
  });
}

// Achicar el tramo descarta las keyframes que quedan afuera; si no queda ninguna, fija la caja que se veia en el borde.
function withSpan(track: RedactionTrack, first: number, last: number): RedactionTrack {
  const inside = track.keyframes.filter((keyframe) => first <= keyframe.frame && keyframe.frame <= last);
  const edge = track.keyframes[0].frame > last ? last : first;
  const keyframes = inside.length > 0 ? inside : [{ frame: edge, box: heldOrBetween(track.keyframes, edge) }];
  return { ...track, firstFrame: first, lastFrame: last, keyframes };
}

export function withTrackStart(choice: RedactionChoice, id: number, frame: number): RedactionChoice {
  return mapTrack(choice, id, (track) => withSpan(track, frame, Math.max(frame, track.lastFrame)));
}

export function withTrackEnd(choice: RedactionChoice, id: number, frame: number): RedactionChoice {
  return mapTrack(choice, id, (track) => withSpan(track, Math.min(frame, track.firstFrame), frame));
}

function followedEdge(value: number, before: number, after: number): number {
  return value === before ? after : value;
}

// Un borde que coincidia con el recorte lo sigue; lo que queda fuera del recorte nuevo se recorta.
function followedTrack(track: RedactionTrack, before: TrimRange, after: TrimRange): RedactionTrack {
  const first = Math.max(after[0], followedEdge(track.firstFrame, before[0], after[0]));
  const last = Math.min(after[1], followedEdge(track.lastFrame, before[1], after[1]));
  return first <= last ? withSpan(track, first, last) : track;
}

export function withSpanFollowed(choice: RedactionChoice, before: TrimRange, after: TrimRange): RedactionChoice {
  return { ...choice, tracks: choice.tracks.map((track) => followedTrack(track, before, after)) };
}

export function overlapsSpan(track: RedactionTrack, span: TrimRange): boolean {
  return track.firstFrame <= span[1] && span[0] <= track.lastFrame;
}

function clippedSpans(choice: RedactionChoice, span: TrimRange): TrimRange[] {
  return choice.tracks
    .map((track): TrimRange => [Math.max(span[0], track.firstFrame), Math.min(span[1], track.lastFrame)])
    .filter(([first, last]) => first <= last)
    .sort((a, b) => a[0] - b[0]);
}

// Mismos tramos que redaction.uncovered_ranges: los cuadros de la copia que ninguna caja tapa.
export function uncoveredRanges(choice: RedactionChoice, span: TrimRange): TrimRange[] {
  const gaps: TrimRange[] = [];
  let following = span[0];
  for (const [first, last] of clippedSpans(choice, span)) {
    if (first > following) {
      gaps.push([following, first - 1]);
    }
    following = Math.max(following, last + 1);
  }
  return following <= span[1] ? [...gaps, [following, span[1]]] : gaps;
}

export function rangesText(ranges: readonly TrimRange[]): string {
  return ranges.map(([first, last]) => (first === last ? `${first}` : `${first}–${last}`)).join(", ");
}

export function withStyle(choice: RedactionChoice, style: CctvRedactionStyle): RedactionChoice {
  return { ...choice, style };
}

function sameBox(a: CctvBox, b: CctvBox): boolean {
  return a.every((value, index) => value === b[index]);
}

function changedIndex(active: readonly ActiveBox[], boxes: readonly CctvBox[]): number {
  return active.findIndex((current, index) => index >= boxes.length || !sameBox(current.box, boxes[index]));
}

// El editor devuelve la lista de cajas visibles: una mas es una caja nueva, una menos se borra y una movida es keyframe.
export function withEditedBoxes(
  choice: RedactionChoice,
  active: readonly ActiveBox[],
  boxes: readonly CctvBox[],
  frame: number,
  span: TrimRange,
): RedactionChoice {
  if (boxes.length > active.length) {
    return withTrackAdded(choice, boxes[boxes.length - 1], frame, span);
  }
  const index = changedIndex(active, boxes);
  if (index < 0) {
    return choice;
  }
  if (boxes.length < active.length) {
    return withTrackRemoved(choice, active[index].id);
  }
  return withKeyframe(choice, active[index].id, frame, boxes[index]);
}

export function redactionBlockerKey(choice: RedactionChoice, span: TrimRange): string | null {
  if (choice.tracks.length === 0) {
    return "cctv.redact.blocked.noBoxes";
  }
  return choice.tracks.every((track) => overlapsSpan(track, span)) ? null : "cctv.redact.blocked.outsideTrim";
}

export function redactionRequest(choice: RedactionChoice): CctvRedactionRequest {
  return {
    style: choice.style,
    tracks: choice.tracks.map((track) => ({
      firstFrame: track.firstFrame,
      lastFrame: track.lastFrame,
      keyframes: track.keyframes.map((keyframe) => ({ frame: keyframe.frame, box: [...keyframe.box] })),
    })),
  };
}
