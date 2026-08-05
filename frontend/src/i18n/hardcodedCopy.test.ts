import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// La copia de la app se escribe en ingles y se traduce por catalogo. Agregar
// texto en espanol directo en un componente rompe eso en silencio: el usuario
// termina con una pantalla en dos idiomas (paso de verdad — la barra lateral en
// ingles llevaba a una pagina en espanol) y ademas ese texto ya no se puede
// traducir. Este test es la red que lo impide.
//
// Los COMENTARIOS del repo se escriben en espanol a proposito, asi que se
// eliminan antes de mirar. Lo que se revisa es solo el texto que ve el usuario.

const SRC_DIR = path.resolve(__dirname, "..");

// Palabras funcionales que no son palabras del ingles. Se piden DOS distintas
// en la misma cadena para que un identificador suelto no dispare el test; una
// frase de verdad en espanol siempre trae varias, con tildes o sin ellas.
const SPANISH_MARKERS = [
  "que", "para", "los", "las", "una", "unos", "unas", "del", "por", "sin",
  "pero", "cuando", "donde", "desde", "hasta", "sobre", "entre", "cada",
  "todo", "toda", "todos", "todas", "esta", "este", "estos", "estas", "mas",
  "más", "muy", "asi", "así", "tambien", "también", "porque", "aunque",
  "segun", "según", "puede", "pueden", "tiene", "tienen", "hace", "hacen",
  "estan", "están", "ningun", "ningún", "ninguna", "algun", "algún", "alguna",
  "archivo", "archivos", "pantalla", "tamano", "tamaño", "idioma", "volumen",
  "cuadros", "elegi", "elegí", "elige", "abrir", "nivel", "ruido", "voz",
];

const MARKER_PATTERN = new RegExp(`\\b(${SPANISH_MARKERS.join("|")})\\b`, "gi");

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/(^|[^:])\/\/.*$/gm, "$1");
}

// Caracteres que el ingles no usa. Alcanzan solos porque no aparecen por
// accidente: atrapan etiquetas cortas ("Sin limite") que no llegan a juntar dos
// palabras marcadoras.
const SPANISH_CHARS = /[áéíóúñ¿¡]/i;

// Solo el texto que llega a la pantalla: cadenas entre comillas y texto suelto
// dentro de JSX. Deja fuera imports, clases de tailwind y nombres de props.
// El texto JSX puede ocupar varias lineas, asi que se aplasta el salto de linea
// antes de mirarlo — si no, una frase larga se escapa por estar partida.
function userFacingText(source: string): string[] {
  const literals = source.match(/"[^"\n]{6,}"|'[^'\n]{6,}'/g) ?? [];
  const jsxText = (source.match(/>[^<>{}]{6,}</g) ?? []).map((t) =>
    t.replace(/\s+/g, " ").trim(),
  );
  return [...literals, ...jsxText];
}

function looksSpanish(text: string): boolean {
  if (SPANISH_CHARS.test(text)) return true;
  const found = new Set((text.match(MARKER_PATTERN) ?? []).map((w) => w.toLowerCase()));
  return found.size >= 2;
}

// El nombre de un idioma se escribe SIEMPRE en ese idioma: alguien que cayo en
// una pantalla que no entiende reconoce "Español", no "Spanish". Traducirlo
// romperia justo la unica etiqueta que tiene que funcionar sin saber el idioma.
const EXEMPT = new Set([path.join("modules", "settings", "LanguageSection.tsx")]);

function sourceFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === "i18n" ? [] : sourceFiles(full);
    if (!/\.tsx?$/.test(entry.name) || /\.test\.tsx?$/.test(entry.name)) return [];
    return [full];
  });
}

describe("la copia de la interfaz vive en el catalogo, no en los componentes", () => {
  it("no deja texto en espanol hardcodeado fuera de src/i18n", () => {
    const offenders = sourceFiles(SRC_DIR).flatMap((file) => {
      const relative = path.relative(SRC_DIR, file);
      if (EXEMPT.has(relative)) return [];
      const spanish = userFacingText(stripComments(fs.readFileSync(file, "utf8"))).filter(
        looksSpanish,
      );
      return spanish.map((text) => `${relative}: ${text.trim().slice(0, 70)}`);
    });

    expect(offenders).toEqual([]);
  });

});
