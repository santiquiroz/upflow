// El identificador interno de un tipo de modelo no es un nombre. Son nombres de
// tecnologia, no copia de la app: no se traducen.
const MODEL_KIND_LABELS: Record<string, string> = {
  "builtin-ncnn": "NCNN (Vulkan)",
  onnx: "ONNX",
  "asr-onnx": "Whisper (ONNX)",
  "generation-onnx": "Diffusion (ONNX)",
};

export function modelKindLabel(kind: string): string {
  return MODEL_KIND_LABELS[kind] ?? kind;
}
