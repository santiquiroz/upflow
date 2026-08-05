import type { ModelResponse, SupportedModelResponse } from "../../lib/apiTypes";

// El backend valida la escala contra el modelo y rechaza el job
// (`job_manager._resolve_builtin_model`). Como /engine ya manda `supportedModels`
// con las escalas de cada uno, la pantalla puede no ofrecer nunca la combinacion
// que va a fallar, en vez de dejar subir el archivo para recien ahi avisar.
//
// Un modelo que el catalogo no describe (un ONNX bajado de Hugging Face) no
// declara escalas, y ahi el rango del motor es lo unico que se sabe.
export function scalesForModel(
  model: ModelResponse | null,
  supportedModels: readonly SupportedModelResponse[],
  allowedScales: readonly number[],
): number[] {
  if (!model) {
    return [...allowedScales];
  }
  const described = supportedModels.find((candidate) => candidate.key === model.id);
  if (!described) {
    return [...allowedScales];
  }
  return allowedScales.filter((scale) => described.scales.includes(scale));
}

// Cambiar de modelo puede dejar seleccionada una escala que el nuevo no soporta.
// Se mueve a la mas cercana en vez de dejarla invalida.
export function correctScaleFor(current: number, offered: readonly number[]): number | null {
  if (offered.length === 0) {
    return null;
  }
  if (offered.includes(current)) {
    return current;
  }
  return offered.reduce((closest, scale) =>
    Math.abs(scale - current) < Math.abs(closest - current) ? scale : closest,
  );
}
