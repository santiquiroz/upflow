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

export const REDACTION_STYLES: readonly CctvRedactionStyle[] = ["pixelate", "blur"];
// Igual que redaction.MAX_TRACKS y MAX_KEYFRAMES del backend.
export const MAX_REDACTION_TRACKS = 32;
export const MAX_REDACTION_KEYFRAMES = 256;

export const EMPTY_REDACTION: RedactionChoice = { style: "pixelate", tracks: [], nextId: 1 };

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

export function redactionBlockerKey(choice: RedactionChoice): string | null {
  return choice.tracks.length === 0 ? "cctv.redact.blocked.noBoxes" : null;
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
