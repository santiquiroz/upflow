// Mensajes internos del Editor (en inglés, de canvas y decodificación) traducidos en el borde de la UI.
const INTERNAL_MESSAGE_KEYS: Record<string, string> = {
  "could not decode image": "editor.error.decodeImage",
  "could not decode mask": "editor.error.decodeMask",
  "canvas 2d context unavailable": "editor.error.canvasContext",
  "Could not load the image": "editor.error.loadImage",
  "could not export the mask": "editor.error.exportMask",
  "could not encode the image": "editor.error.encodeImage",
  "Segmentation failed": "editor.error.segmentation",
};

export function localizedErrorMessage(error: unknown, t: (key: string) => string): string {
  const message = error instanceof Error ? error.message : String(error);
  const key = INTERNAL_MESSAGE_KEYS[message];
  return key ? t(key) : message;
}
