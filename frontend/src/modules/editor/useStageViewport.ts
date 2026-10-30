import { useEffect, type Dispatch, type MutableRefObject, type SetStateAction } from "react";
import { clampPan, zoomAtPoint, type Viewport } from "./viewport";

type SetViewport = Dispatch<SetStateAction<Viewport>>;

export function useStageViewport(
  stageRef: MutableRefObject<HTMLDivElement | null>,
  setViewport: SetViewport,
  resetKey: unknown,
): void {
  // La rueda se escucha a mano con passive:false — React registra onWheel como
  // pasivo y preventDefault ahí no frena el scroll de la página.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }
    function handleWheel(event: WheelEvent) {
      event.preventDefault();
      const rect = stage!.getBoundingClientRect();
      // El origen del transform es la caja interior: descontar el borde o el
      // punto anclado se corre un píxel por paso y el error se acumula.
      const focus = {
        x: event.clientX - rect.left - stage!.clientLeft,
        y: event.clientY - rect.top - stage!.clientTop,
      };
      const container = { width: stage!.clientWidth, height: stage!.clientHeight };
      setViewport((current) =>
        zoomAtPoint(current, current.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), focus, container),
      );
    }
    stage.addEventListener("wheel", handleWheel, { passive: false });
    return () => stage.removeEventListener("wheel", handleWheel);
  }, [resetKey]);

  // Al achicar la ventana el lienzo encoge y un desplazamiento que era legal
  // deja la foto entera fuera de la vista: el usuario ve un recuadro vacío sin
  // ninguna pista. Se vuelve a acotar cada vez que cambia el tamaño.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      setViewport((current) =>
        clampPan(current, { width: stage.clientWidth, height: stage.clientHeight }),
      );
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, [resetKey]);
}
