import type { BinaryMask } from "../../editor/maskCanvas";

// Espejo de scratch_fill.LARGE_HOLE_PX: MI-GAN deja manchones en huecos mas anchos.
export const LARGE_HOLE_PX = 24;

interface Region {
  left: number;
  top: number;
  width: number;
  height: number;
}

function rowSpan(mask: BinaryMask, y: number): [number, number] | null {
  const start = y * mask.width;
  let first = -1;
  let last = -1;
  for (let x = 0; x < mask.width; x += 1) {
    if (mask.data[start + x] === 0) continue;
    first = first < 0 ? x : first;
    last = x;
  }
  return first < 0 ? null : [first, last];
}

function markedBounds(mask: BinaryMask): Region | null {
  let minX = mask.width;
  let maxX = -1;
  let minY = -1;
  let maxY = -1;
  for (let y = 0; y < mask.height; y += 1) {
    const span = rowSpan(mask, y);
    if (span === null) continue;
    minX = Math.min(minX, span[0]);
    maxX = Math.max(maxX, span[1]);
    minY = minY < 0 ? y : minY;
    maxY = y;
  }
  return maxY < 0 ? null : { left: minX, top: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

// Recorte con un borde vacio de 1 px: fuera del recorte todo es fondo (o fuera
// de la foto, que el backend tambien rellena con fondo), asi que da lo mismo.
function paddedCrop(mask: BinaryMask, region: Region): BinaryMask {
  const width = region.width + 2;
  const height = region.height + 2;
  const data = new Uint8Array(width * height);
  for (let y = 0; y < region.height; y += 1) {
    const from = (region.top + y) * mask.width + region.left;
    data.set(mask.data.subarray(from, from + region.width), (y + 1) * width + 1);
  }
  return { width, height, data };
}

interface Scratch {
  f: Float64Array;
  d: Float64Array;
  v: Int32Array;
  z: Float64Array;
}

function parabolaCross(f: Float64Array, q: number, p: number): number {
  return (f[q] + q * q - (f[p] + p * p)) / (2 * q - 2 * p);
}

// Felzenszwalb y Huttenlocher: distancia euclidea exacta al cuadrado en 1D.
function distance1d(n: number, scratch: Scratch): void {
  const { f, d, v, z } = scratch;
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q += 1) {
    let s = parabolaCross(f, q, v[k]);
    while (s <= z[k]) {
      k -= 1;
      s = parabolaCross(f, q, v[k]);
    }
    k += 1;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q += 1) {
    while (z[k + 1] < q) k += 1;
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

function makeScratch(length: number): Scratch {
  return {
    f: new Float64Array(length),
    d: new Float64Array(length),
    v: new Int32Array(length),
    z: new Float64Array(length + 1),
  };
}

// Primera pasada exacta y barata para una mascara binaria: distancia vertical
// al fondo, de arriba abajo y de abajo arriba (el recorte empieza y termina en fondo).
function verticalDistances(mask: BinaryMask): Float64Array {
  const { width, data } = mask;
  const grid = new Float64Array(data.length);
  for (let index = width; index < data.length; index += 1) {
    grid[index] = data[index] ? grid[index - width] + 1 : 0;
  }
  for (let index = data.length - width - 1; index >= 0; index -= 1) {
    grid[index] = Math.min(grid[index], grid[index + width] + 1);
  }
  return grid;
}

function isBackgroundRow(grid: Float64Array, start: number, width: number): boolean {
  for (let index = start; index < start + width; index += 1) {
    if (grid[index] !== 0) return false;
  }
  return true;
}

// Igual que cv2.distanceTransform(DIST_L2, PRECISE): distancia al cuadrado de
// cada pixel marcado al pixel de fondo mas cercano.
function squaredDistances(mask: BinaryMask): Float64Array {
  const grid = verticalDistances(mask);
  const scratch = makeScratch(mask.width);
  for (let start = 0; start < grid.length; start += mask.width) {
    if (isBackgroundRow(grid, start, mask.width)) continue;
    for (let x = 0; x < mask.width; x += 1) scratch.f[x] = grid[start + x] * grid[start + x];
    distance1d(mask.width, scratch);
    grid.set(scratch.d, start);
  }
  return grid;
}

function floodLabel(mask: BinaryMask, labels: Int32Array, stack: Int32Array, start: number, label: number): void {
  const offsets = [-1, 1, -mask.width, mask.width];
  let top = 0;
  stack[top++] = start;
  labels[start] = label;
  while (top > 0) {
    const index = stack[--top];
    for (const offset of offsets) {
      const next = index + offset;
      // El borde de 1 px del recorte es fondo: ningun vecino se sale del arreglo.
      if (mask.data[next] === 1 && labels[next] === 0) {
        labels[next] = label;
        stack[top++] = next;
      }
    }
  }
}

// Conexion por lados, como el ndimage.label por defecto del backend.
function labelComponents(mask: BinaryMask): { labels: Int32Array; count: number } {
  const labels = new Int32Array(mask.data.length);
  // Cada pixel entra a la pila una sola vez: alcanza con el tamano del recorte.
  const stack = new Int32Array(mask.data.length);
  let count = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    if (mask.data[index] === 1 && labels[index] === 0) {
      count += 1;
      floodLabel(mask, labels, stack, index, count);
    }
  }
  return { labels, count };
}

// Ancho de cada componente = el doble de su distancia maxima al borde (§3.4.2).
export function holeWidths(mask: BinaryMask): number[] {
  const region = markedBounds(mask);
  if (region === null) {
    return [];
  }
  const crop = paddedCrop(mask, region);
  const { labels, count } = labelComponents(crop);
  const distances = squaredDistances(crop);
  const widest = new Float64Array(count + 1);
  for (let index = 0; index < labels.length; index += 1) {
    widest[labels[index]] = Math.max(widest[labels[index]], distances[index]);
  }
  return Array.from(widest.subarray(1), (squared) => 2 * Math.sqrt(squared));
}

export function largeHoleCount(mask: BinaryMask): number {
  return holeWidths(mask).filter((width) => width > LARGE_HOLE_PX).length;
}
