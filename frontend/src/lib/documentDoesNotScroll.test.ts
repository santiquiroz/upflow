import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

// El documento NUNCA debe scrollear: scrollean los paneles de adentro.
//
// Reportado dos veces el 2026-08-06. La primera vez se arreglo solo el
// contenedor de la app (`h-dvh` + columnas con overflow) y el problema siguió:
// eso evita que las COLUMNAS estiren la app, pero no impide que el DOCUMENTO
// crezca si algo se escapa del contenedor. Ahí la página se corre fuera de la
// ventana y queda una franja vacía abajo.
//
// Se chequea sobre el ARCHIVO y no sobre el DOM porque jsdom no aplica CSS: un
// test que renderice no vería la diferencia. Es la misma lección que `h-dvh`,
// que se verificó contra el CSS compilado — una regla que no llega al build se
// ve bien en el código y no hace nada.
const CSS_GLOBAL = join(__dirname, "..", "index.css");

describe("el documento no scrollea", () => {
  const css = readFileSync(CSS_GLOBAL, "utf-8").replace(/\s+/g, " ");

  it("html, body y #root quedan clavados a la ventana", () => {
    expect(css).toMatch(/html,\s*body,\s*#root\s*\{[^}]*height:\s*100dvh/);
  });

  it("y no dejan desbordar", () => {
    expect(css).toMatch(/html,\s*body,\s*#root\s*\{[^}]*overflow:\s*hidden/);
  });
});
