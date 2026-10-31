// El alfa de la máscara es el peso de la mezcla: 255 mezcla todo el píxel, 0 no toca nada.

function copyImage(source: ImageData): ImageData {
  const data = new Uint8ClampedArray(source.data);
  // putImageData del navegador rechaza objetos planos; jsdom no trae ImageData.
  if (typeof ImageData === "function") return new ImageData(data, source.width, source.height);
  return { width: source.width, height: source.height, data } as ImageData;
}

function mix(from: number, to: number, weight: number): number {
  return Math.round(from + (to - from) * weight);
}

function blendInto(out: ImageData, index: number, from: ArrayLike<number>, fromIndex: number, weight: number): void {
  for (let channel = 0; channel < 4; channel += 1) {
    out.data[index + channel] = mix(out.data[index + channel], from[fromIndex + channel], weight);
  }
}

function weightAt(mask: ImageData, index: number): number {
  return mask.data[index + 3] / 255;
}

export function restoreWithMask(current: ImageData, original: ImageData, mask: ImageData): ImageData {
  const out = copyImage(current);
  for (let index = 0; index < out.data.length; index += 4) {
    const weight = weightAt(mask, index);
    if (weight > 0) blendInto(out, index, original.data, index, weight);
  }
  return out;
}

function sourceIndex(width: number, height: number, x: number, y: number): number | null {
  if (x < 0 || y < 0 || x >= width || y >= height) return null;
  return (y * width + x) * 4;
}

export function cloneWithMask(current: ImageData, mask: ImageData, rawDx: number, rawDy: number): ImageData {
  // El desplazamiento llega en píxeles de foto con decimales; un índice fraccionario lee undefined.
  const dx = Math.round(rawDx);
  const dy = Math.round(rawDy);
  const out = copyImage(current);
  for (let y = 0; y < out.height; y += 1) {
    for (let x = 0; x < out.width; x += 1) {
      const index = (y * out.width + x) * 4;
      const weight = weightAt(mask, index);
      const from = sourceIndex(out.width, out.height, x + dx, y + dy);
      if (weight > 0 && from !== null) blendInto(out, index, current.data, from, weight);
    }
  }
  return out;
}
