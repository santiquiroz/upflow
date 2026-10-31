import { useCallback, useRef, useState, type MutableRefObject } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { InitImageResponse } from "../../lib/apiTypes";
import { fetchInitImage, healEditorImage } from "../../services/editor";
import { uploadGenerationInitImage } from "../../services/generation";
import { createMaskCanvas, exportMaskBlob } from "./editorMaskCanvas";
import { drawStrokes, type BrushStroke } from "./maskCanvas";
import { isRetouchTool } from "./useRetouchTool";

// Mismo límite que el backend (SMALL_ZONE_SIDE): más grande, la zona es de Quitar.
export const HEAL_MAX_SIDE = 700;

// Corrector, Restaurar y Clonar escriben la base: mientras uno está en vuelo, los demás no arrancan.
export function editsBase(tool: string): boolean {
  return tool === "heal" || isRetouchTool(tool);
}

// Caja del trazo en píxeles de la foto: los puntos más el radio del pincel en cada borde.
// Un solo recorrido: Math.max(...xs) desborda la pila con un trazo largo.
export function strokeSides(stroke: BrushStroke): { width: number; height: number } {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const { x, y } of stroke.points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  return {
    width: maxX - minX + 2 * stroke.radius,
    height: maxY - minY + 2 * stroke.radius,
  };
}

export function isHealStrokeTooLarge(stroke: BrushStroke): boolean {
  const { width, height } = strokeSides(stroke);
  return Math.max(width, height) > HEAL_MAX_SIDE;
}

// La máscara del corrector es blanco=zona, negro=resto, con el tamaño de la foto.
function healMaskBlob(stroke: BrushStroke, width: number, height: number): Promise<Blob> {
  const canvas = createMaskCanvas(width, height);
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.reject(new Error("canvas 2d context unavailable"));
  drawStrokes(ctx, [stroke], "#ffffff");
  return exportMaskBlob(canvas);
}

interface HealToolOptions {
  baseInfo: InitImageResponse | null;
  // Cada foto nueva sube este contador: una corrección que termina después ya no pertenece a la imagen que se ve.
  sequenceRef: MutableRefObject<number>;
  onAdopt: (info: InitImageResponse, blob: Blob) => void;
  onError: (message: string) => void;
}

export function useHealTool({ baseInfo, sequenceRef, onAdopt, onError }: HealToolOptions) {
  const { t } = useTranslation();
  const [healing, setHealing] = useState(false);
  const healingRef = useRef(false);

  // Al soltar el trazo: sube su máscara, pide la corrección y adopta la imagen nueva como base. No toca el historial de máscaras.
  const commit = useCallback(async (stroke: BrushStroke) => {
    if (!baseInfo || healingRef.current) return;
    if (isHealStrokeTooLarge(stroke)) {
      onError(t("editor.heal.tooLarge"));
      return;
    }
    const sequence = sequenceRef.current;
    healingRef.current = true;
    setHealing(true);
    try {
      const maskBlob = await healMaskBlob(stroke, baseInfo.width, baseInfo.height);
      const maskUpload = await uploadGenerationInitImage(new File([maskBlob], "heal-mask.png", { type: "image/png" }));
      const healed = await healEditorImage(baseInfo.initImageToken, maskUpload.initImageToken);
      const blob = await fetchInitImage(healed.initImageToken);
      if (sequence === sequenceRef.current) onAdopt(healed, blob);
    } catch (error) {
      if (sequence === sequenceRef.current) onError(error instanceof Error ? error.message : String(error));
    } finally {
      healingRef.current = false;
      setHealing(false);
    }
  }, [baseInfo, sequenceRef, onAdopt, onError, t]);

  return { healing, commit };
}
