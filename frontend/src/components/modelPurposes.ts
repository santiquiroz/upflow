export const SUPER_RESOLUTION = "SR";
export const RESTORATION = "Restoration";
export const UPSCALE_ONLY: readonly string[] = [SUPER_RESOLUTION];

// Imagen y video corren cualquier modelo RGB 3->3 que el instalador acepto: un 1x de
// limpieza (DeJPEG, denoise) limpia y el resto de la escala es un reescalado comun.
export const IMAGE_PIPELINE_PURPOSES: readonly string[] = [SUPER_RESOLUTION, RESTORATION];
