import { useCallback, useEffect, useReducer, useRef } from "react";
import type { RestoreGeometry } from "../../../lib/restoreApiTypes";
import { analyzePhoto, setPhotoGeometry, uploadDamageMask } from "../../../services/restore";
import {
  INITIAL_RESTORE_SESSION,
  restoreSessionReducer,
  type FacePatch,
  type RestoreSessionState,
} from "./restoreSessionState";

export interface RestoreSessionServices {
  analyzePhoto: typeof analyzePhoto;
  setPhotoGeometry: typeof setPhotoGeometry;
  uploadDamageMask: typeof uploadDamageMask;
}

export interface RestoreSession extends RestoreSessionState {
  analyze: (file: File) => Promise<void>;
  applyGeometry: (geometry: RestoreGeometry) => Promise<void>;
  saveMask: (mask: Blob) => Promise<boolean>;
  updateFace: (index: number, patch: FacePatch) => void;
  reset: () => void;
}

const DEFAULT_SERVICES: RestoreSessionServices = { analyzePhoto, setPhotoGeometry, uploadDamageMask };

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function useRestoreSession(services: RestoreSessionServices = DEFAULT_SERVICES): RestoreSession {
  const [state, dispatch] = useReducer(restoreSessionReducer, INITIAL_RESTORE_SESSION);
  // Cada pedido lleva un numero: la respuesta de una foto anterior (o de un
  // giro ya reemplazado) no debe pisar lo que el usuario tiene en pantalla.
  const requestRef = useRef(0);
  const uploadRef = useRef<AbortController | null>(null);
  const token = state.analysis?.token ?? null;

  const nextRequest = useCallback(() => {
    uploadRef.current?.abort();
    uploadRef.current = null;
    requestRef.current += 1;
    return requestRef.current;
  }, []);

  useEffect(() => () => uploadRef.current?.abort(), []);

  const analyze = useCallback(
    async (file: File) => {
      const request = nextRequest();
      const controller = new AbortController();
      uploadRef.current = controller;
      dispatch({ type: "analyzeStarted", fileName: file.name });
      try {
        const analysis = await services.analyzePhoto(file, {
          signal: controller.signal,
          onProgress: (percent) => {
            if (request === requestRef.current) dispatch({ type: "uploadProgress", percent });
          },
        });
        if (request === requestRef.current) dispatch({ type: "analysisReceived", analysis });
      } catch (error) {
        if (request === requestRef.current) dispatch({ type: "failed", message: errorText(error) });
      }
    },
    [nextRequest, services],
  );

  const applyGeometry = useCallback(
    async (geometry: RestoreGeometry) => {
      if (token === null) return;
      const request = nextRequest();
      dispatch({ type: "geometryStarted" });
      try {
        const analysis = await services.setPhotoGeometry(token, geometry);
        if (request === requestRef.current) dispatch({ type: "analysisReceived", analysis });
      } catch (error) {
        if (request === requestRef.current) dispatch({ type: "failed", message: errorText(error) });
      }
    },
    [nextRequest, services, token],
  );

  const saveMask = useCallback(
    async (mask: Blob) => {
      if (token === null) return false;
      const request = requestRef.current;
      try {
        const saved = await services.uploadDamageMask(token, mask);
        if (request !== requestRef.current) return false;
        dispatch({ type: "maskSaved", coverage: saved.coverage });
        return true;
      } catch (error) {
        if (request === requestRef.current) dispatch({ type: "failed", message: errorText(error) });
        return false;
      }
    },
    [services, token],
  );

  const updateFace = useCallback((index: number, patch: FacePatch) => {
    dispatch({ type: "faceChanged", index, patch });
  }, []);

  const reset = useCallback(() => {
    nextRequest();
    dispatch({ type: "reset" });
  }, [nextRequest]);

  return { ...state, analyze, applyGeometry, saveMask, updateFace, reset };
}
