import type { RestoreOptions, RestoreStepOptions } from "../../../lib/restoreApiTypes";
import { rasterizeStrokes, type BinaryMask, type BrushStroke } from "../../editor/maskCanvas";

// Espejo de app/services/engines/scratch_detect.py: la sensibilidad mueve el
// umbral sin volver a correr el detector.
const LEAST_SENSITIVE_THRESHOLD = 0.6;
const MOST_SENSITIVE_THRESHOLD = 0.2;
const U8_PEAK = 255;
// §2.2 paso 5: con mas de 3% marcado el detector pudo tomar escritura por dano.
export const REVIEW_COVERAGE = 0.03;

const DEFAULT_SENSITIVITY = 0.5;
const MASK_RGB = [239, 68, 68] as const;
const MASK_ALPHA = 150;
// El trazo en vivo se pinta con el mismo color que la mascara ya compuesta.
export const MASK_CSS_COLOR = `rgba(${MASK_RGB.join(", ")}, ${(MASK_ALPHA / 255).toFixed(2)})`;
const HINT_MAX_ALPHA = 80;
const RGBA = 4;

export interface ProbabilityMap {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface MaskSize {
  width: number;
  height: number;
}

export interface MaskSettings {
  sensitivity: number;
  growPx: number;
}

export interface MaskReviewFacts {
  coverage: number;
  largeHoles: number;
  damageOverFaces: boolean;
}

export function maskSettingsFrom(options: RestoreStepOptions): MaskSettings {
  const sensitivity = Number(options.sensitivity ?? DEFAULT_SENSITIVITY);
  const growPx = Number(options.grow_px ?? 0);
  return {
    sensitivity: Number.isFinite(sensitivity) ? sensitivity : DEFAULT_SENSITIVITY,
    growPx: Number.isFinite(growPx) ? growPx : 0,
  };
}

export function sensitivityThreshold(sensitivity: number): number {
  const span = LEAST_SENSITIVE_THRESHOLD - MOST_SENSITIVE_THRESHOLD;
  return LEAST_SENSITIVE_THRESHOLD - sensitivity * span;
}

export function emptyMask(size: MaskSize): BinaryMask {
  return { width: size.width, height: size.height, data: new Uint8Array(size.width * size.height) };
}

// numpy compara el float32 del mapa contra el umbral convertido a float32.
export function thresholdProbability(map: ProbabilityMap, sensitivity: number): BinaryMask {
  const threshold = Math.fround(sensitivityThreshold(sensitivity));
  const data = new Uint8Array(map.data.length);
  for (let index = 0; index < data.length; index += 1) {
    data[index] = Math.fround(map.data[index] / U8_PEAK) > threshold ? 1 : 0;
  }
  return { width: map.width, height: map.height, data };
}

interface LineAccess {
  count: number;
  length: number;
  offset: (line: number, position: number) => number;
}

function rowAccess(mask: BinaryMask): LineAccess {
  return { count: mask.height, length: mask.width, offset: (line, position) => line * mask.width + position };
}

function columnAccess(mask: BinaryMask): LineAccess {
  return { count: mask.width, length: mask.height, offset: (line, position) => position * mask.width + line };
}

function linePrefix(source: Uint8Array, access: LineAccess, line: number, prefix: Int32Array): void {
  for (let position = 0; position < access.length; position += 1) {
    prefix[position + 1] = prefix[position] + source[access.offset(line, position)];
  }
}

// Un paso separable por sumas acumuladas: O(n) sin importar el radio.
// Afuera de la foto cuenta como vacio al agrandar y como marcado al achicar.
function slideLines(source: Uint8Array, access: LineAccess, radius: number, grow: boolean): Uint8Array {
  const result = new Uint8Array(source.length);
  const prefix = new Int32Array(access.length + 1);
  for (let line = 0; line < access.count; line += 1) {
    linePrefix(source, access, line, prefix);
    for (let position = 0; position < access.length; position += 1) {
      const start = Math.max(0, position - radius);
      const end = Math.min(access.length, position + radius + 1);
      const marked = prefix[end] - prefix[start];
      const keep = grow ? marked > 0 : marked === end - start;
      result[access.offset(line, position)] = keep ? 1 : 0;
    }
  }
  return result;
}

export function growMask(mask: BinaryMask, pixels: number): BinaryMask {
  if (pixels === 0) {
    return mask;
  }
  const radius = Math.abs(pixels);
  const grow = pixels > 0;
  const rows = slideLines(mask.data, rowAccess(mask), radius, grow);
  return { ...mask, data: slideLines(rows, columnAccess(mask), radius, grow) };
}

function detectedMask(map: ProbabilityMap | null, size: MaskSize, settings: MaskSettings): BinaryMask {
  // Un mapa de otra geometria no describe esta copia de trabajo.
  if (map === null || map.width !== size.width || map.height !== size.height) {
    return emptyMask(size);
  }
  return growMask(thresholdProbability(map, settings.sensitivity), settings.growPx);
}

// El backend usa la mascara subida tal cual (reemplaza a la automatica), asi
// que la sensibilidad y el agrandado ya tienen que venir aplicados.
export function composeDamageMask(
  map: ProbabilityMap | null,
  size: MaskSize,
  settings: MaskSettings,
  strokes: readonly BrushStroke[],
): BinaryMask {
  return rasterizeStrokes(detectedMask(map, size, settings), strokes);
}

export function maskCoverage(mask: BinaryMask): number {
  if (mask.data.length === 0) {
    return 0;
  }
  let marked = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    marked += mask.data[index];
  }
  return marked / mask.data.length;
}

export function needsMaskReview(facts: MaskReviewFacts): boolean {
  return facts.coverage > REVIEW_COVERAGE || facts.largeHoles > 0 || facts.damageOverFaces;
}

export function maskReviewKey(settings: MaskSettings, editCount: number): string {
  return `${settings.sensitivity}|${settings.growPx}|${editCount}`;
}

// Solo se nombra la mascara pintada si hay una en juego: sin trazos pero con
// una subida antes, hay que decirle al backend que la ignore.
export function withMaskChoice(options: RestoreOptions, useUserMask: boolean, hasSavedMask: boolean): RestoreOptions {
  if (options.repair === undefined || (!useUserMask && !hasSavedMask)) {
    return options;
  }
  return { ...options, repair: { ...options.repair, use_user_mask: useUserMask } };
}

function hintAlpha(map: ProbabilityMap | null, index: number): number {
  return map === null ? 0 : Math.round((map.data[index] / U8_PEAK) * HINT_MAX_ALPHA);
}

// Lo marcado va opaco; debajo, la probabilidad tenue muestra lo que tomaria
// una sensibilidad mas alta.
export function overlayPixels(map: ProbabilityMap | null, mask: BinaryMask): Uint8ClampedArray {
  const usable = map !== null && map.data.length === mask.data.length ? map : null;
  const pixels = new Uint8ClampedArray(mask.data.length * RGBA);
  for (let index = 0; index < mask.data.length; index += 1) {
    const base = index * RGBA;
    pixels[base] = MASK_RGB[0];
    pixels[base + 1] = MASK_RGB[1];
    pixels[base + 2] = MASK_RGB[2];
    pixels[base + 3] = mask.data[index] ? MASK_ALPHA : hintAlpha(usable, index);
  }
  return pixels;
}
