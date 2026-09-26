import { fireEvent } from "@testing-library/react";

// jsdom no trae PointerEvent: sin esto fireEvent crea un Event plano y se pierden clientX/clientY.
class TestPointerEvent extends MouseEvent {
  readonly pointerId: number;

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 0;
  }
}

export function installPointerEvent(): void {
  if (typeof window.PointerEvent === "undefined") {
    window.PointerEvent = TestPointerEvent as unknown as typeof PointerEvent;
  }
}

// Un "1080p Lite" se muestra al doble de ancho: 1920 px en pantalla son 960 px guardados.
export const DISPLAYED_LITE = { left: 0, top: 0, width: 1920, height: 1080, right: 1920, bottom: 1080, x: 0, y: 0 };

export function drag(surface: HTMLElement, from: [number, number], to: [number, number], release = true): void {
  fireEvent.pointerDown(surface, { clientX: from[0], clientY: from[1], pointerId: 1 });
  fireEvent.pointerMove(surface, { clientX: to[0], clientY: to[1], pointerId: 1 });
  if (release) {
    fireEvent.pointerUp(surface, { pointerId: 1 });
  }
}
