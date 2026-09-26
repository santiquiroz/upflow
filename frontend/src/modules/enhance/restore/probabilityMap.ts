import type { ProbabilityMap } from "./damageMask";

const RGBA = 4;

export function redChannel(rgba: Uint8ClampedArray): Uint8Array {
  const out = new Uint8Array(rgba.length / RGBA);
  for (let index = 0; index < out.length; index += 1) {
    out[index] = rgba[index * RGBA];
  }
  return out;
}

function readPixels(bitmap: ImageBitmap): ProbabilityMap {
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (context === null) {
    throw new Error("Canvas 2D is not available");
  }
  context.drawImage(bitmap, 0, 0);
  const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  return { width: bitmap.width, height: bitmap.height, data: redChannel(pixels) };
}

// Sin conversion de color: el PNG guarda probabilidades, no un tono a corregir.
export async function loadProbabilityMap(url: string, signal?: AbortSignal): Promise<ProbabilityMap> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Damage map request failed with ${response.status}`);
  }
  const bitmap = await createImageBitmap(await response.blob(), {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  try {
    return readPixels(bitmap);
  } finally {
    bitmap.close();
  }
}
