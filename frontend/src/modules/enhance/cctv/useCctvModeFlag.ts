import { useState } from "react";

export function readCctvFlag(search: string): boolean {
  return new URLSearchParams(search).get("cctv") === "1";
}

// Se lee de window.location y no del router: /enhance/video?cctv=1 es solo el punto de entrada
// (arbol de capacidades) y el panel sigue siendo testeable sin router.
export function useCctvModeFlag(): [boolean, (enabled: boolean) => void] {
  const [enabled, setEnabled] = useState(() => readCctvFlag(window.location.search));
  return [enabled, setEnabled];
}
