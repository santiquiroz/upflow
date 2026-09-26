import { useEffect, useState } from "react";
import type { CropBox } from "../../../lib/restoreApiTypes";
import type { Size } from "./geometryMath";
import type { PreviewBeforeDeps } from "./previewBefore";

export function usePreviewBefore(
  previewUrl: string,
  crop: CropBox | null,
  working: Size,
  deps: PreviewBeforeDeps,
): string | null {
  const [url, setUrl] = useState<string | null>(null);
  const cropKey = crop === null ? null : crop.join(",");

  useEffect(() => {
    setUrl(null);
    if (crop === null) {
      return undefined;
    }
    let active = true;
    let made: string | null = null;
    deps
      .cut(previewUrl, crop, working)
      .then((cut) => {
        made = cut;
        if (active) {
          setUrl(cut);
        } else {
          deps.release(cut);
        }
      })
      // Sin "antes" la tarjeta muestra solo el resultado: no hay nada que avisar.
      .catch(() => undefined);
    return () => {
      active = false;
      if (made !== null) {
        deps.release(made);
      }
    };
    // cropKey resume el recorte: el arreglo cambia de identidad en cada lectura del job.
  }, [previewUrl, cropKey, working.width, working.height, deps]);

  return url;
}
