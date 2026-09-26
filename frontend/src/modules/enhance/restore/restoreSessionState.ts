import type { RestoreAnalysis, RestoreFace } from "../../../lib/restoreApiTypes";

export type RestoreSessionPhase = "idle" | "analyzing" | "updating" | "ready" | "failed";

export interface RestoreSessionState {
  phase: RestoreSessionPhase;
  fileName: string | null;
  analysis: RestoreAnalysis | null;
  faces: RestoreFace[];
  maskCoverage: number | null;
  // Sube con cada analisis: la vista previa conserva su URL y sin esto el
  // navegador mostraria la copia vieja despues de girar o recortar.
  revision: number;
  uploadPercent: number | null;
  errorMessage: string | null;
}

export type FacePatch = Partial<Pick<RestoreFace, "enabled" | "blend">>;

export type RestoreSessionAction =
  | { type: "analyzeStarted"; fileName: string }
  | { type: "uploadProgress"; percent: number | null }
  | { type: "analysisReceived"; analysis: RestoreAnalysis }
  | { type: "geometryStarted" }
  | { type: "maskSaved"; coverage: number }
  | { type: "faceChanged"; index: number; patch: FacePatch }
  | { type: "failed"; message: string }
  | { type: "reset" };

export const INITIAL_RESTORE_SESSION: RestoreSessionState = {
  phase: "idle",
  fileName: null,
  analysis: null,
  faces: [],
  maskCoverage: null,
  revision: 0,
  uploadPercent: null,
  errorMessage: null,
};

function startAnalysis(state: RestoreSessionState, fileName: string): RestoreSessionState {
  return {
    ...INITIAL_RESTORE_SESSION,
    revision: state.revision,
    phase: "analyzing",
    fileName,
  };
}

// La geometria nueva borra la mascara pintada en el backend, asi que la
// cobertura guardada deja de valer.
function receiveAnalysis(state: RestoreSessionState, analysis: RestoreAnalysis): RestoreSessionState {
  return {
    ...state,
    phase: "ready",
    analysis,
    faces: analysis.faces,
    maskCoverage: null,
    revision: state.revision + 1,
    uploadPercent: null,
    errorMessage: null,
  };
}

function patchFace(faces: RestoreFace[], index: number, patch: FacePatch): RestoreFace[] {
  return faces.map((face) => (face.index === index ? { ...face, ...patch } : face));
}

// Un fallo al girar o recortar deja el analisis anterior en pantalla: sigue
// siendo valido y el usuario puede reintentar.
function fail(state: RestoreSessionState, message: string): RestoreSessionState {
  return {
    ...state,
    phase: state.analysis === null ? "failed" : "ready",
    uploadPercent: null,
    errorMessage: message,
  };
}

export function restoreSessionReducer(
  state: RestoreSessionState,
  action: RestoreSessionAction,
): RestoreSessionState {
  switch (action.type) {
    case "analyzeStarted":
      return startAnalysis(state, action.fileName);
    case "uploadProgress":
      return { ...state, uploadPercent: action.percent };
    case "analysisReceived":
      return receiveAnalysis(state, action.analysis);
    case "geometryStarted":
      return { ...state, phase: "updating", errorMessage: null };
    case "maskSaved":
      return { ...state, maskCoverage: action.coverage, errorMessage: null };
    case "faceChanged":
      return { ...state, faces: patchFace(state.faces, action.index, action.patch) };
    case "failed":
      return fail(state, action.message);
    case "reset":
      return { ...INITIAL_RESTORE_SESSION, revision: state.revision };
  }
}

export function isSessionBusy(phase: RestoreSessionPhase): boolean {
  return phase === "analyzing" || phase === "updating";
}
