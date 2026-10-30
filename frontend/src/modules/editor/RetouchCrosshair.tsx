import type { BrushPoint } from "./maskCanvas";

interface RetouchCrosshairProps {
  point: BrushPoint;
  imageWidth: number;
  imageHeight: number;
}

// Vive dentro del mismo transform que la imagen: sus porcentajes son de la imagen, no de la pantalla.
export function RetouchCrosshair({ point, imageWidth, imageHeight }: RetouchCrosshairProps) {
  return (
    <div
      aria-hidden="true"
      data-testid="retouch-crosshair"
      className="pointer-events-none absolute h-6 w-6 -translate-x-1/2 -translate-y-1/2"
      style={{ left: `${(point.x / imageWidth) * 100}%`, top: `${(point.y / imageHeight) * 100}%` }}
    >
      <span className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-black/70" />
      <span className="absolute left-0 top-1/2 h-px w-full -translate-y-1/2 bg-black/70" />
      <span className="absolute inset-1/4 rounded-full border border-white/90" />
    </div>
  );
}
