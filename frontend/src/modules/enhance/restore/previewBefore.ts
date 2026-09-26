import type { CropBox } from "../../../lib/restoreApiTypes";
import type { Size } from "./geometryMath";
import { previewSourceRect } from "./previewCrop";

export interface PreviewBeforeDeps {
  cut: (previewUrl: string, crop: CropBox, working: Size) => Promise<string>;
  release: (url: string) => void;
}

const BEFORE_TYPE = "image/png";

function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not encode the preview area"))), BEFORE_TYPE);
  });
}

function drawArea(bitmap: ImageBitmap, crop: CropBox, working: Size): HTMLCanvasElement {
  const rect = previewSourceRect(crop, working, { width: bitmap.width, height: bitmap.height });
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("Canvas 2D is not available");
  }
  context.drawImage(bitmap, rect.x, rect.y, rect.width, rect.height, 0, 0, canvas.width, canvas.height);
  return canvas;
}

// El "antes" sale de la vista previa de la sesion: el backend solo guarda el "despues".
async function cutPreviewArea(previewUrl: string, crop: CropBox, working: Size): Promise<string> {
  const response = await fetch(previewUrl);
  if (!response.ok) {
    throw new Error(`Preview request failed with ${response.status}`);
  }
  const bitmap = await createImageBitmap(await response.blob());
  try {
    return URL.createObjectURL(await canvasBlob(drawArea(bitmap, crop, working)));
  } finally {
    bitmap.close();
  }
}

export const DEFAULT_PREVIEW_BEFORE: PreviewBeforeDeps = {
  cut: cutPreviewArea,
  release: (url) => URL.revokeObjectURL(url),
};
