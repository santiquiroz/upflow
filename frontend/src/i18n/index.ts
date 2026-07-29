import { en } from "./en";
import { es } from "./es";

export type Locale = "es" | "en";
export type Catalog = Record<string, string>;
export type TranslationParams = Record<string, string | number>;

const CATALOGS: Record<Locale, Catalog> = { en, es };
const STORAGE_KEY = "upflow.locale";

export const LOCALES: readonly Locale[] = ["es", "en"];

export function isLocale(value: unknown): value is Locale {
  return value === "es" || value === "en";
}

export function detectLocale(): Locale {
  // Los tests fijan el locale en español (ver vitest.setup.ts): la app se
  // escribió en español, así que el catálogo `es` reproduce los textos
  // originales VERBATIM y los 521 tests que buscan texto literal siguen siendo
  // la red que detecta una extracción mal hecha.
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (isLocale(stored)) return stored;
  } catch {
    // localStorage puede no existir (SSR, modo privado): no es motivo para
    // que la app no arranque.
  }
  const preferred = typeof navigator !== "undefined" ? navigator.language : "";
  return preferred.toLowerCase().startsWith("es") ? "es" : "en";
}

export function persistLocale(locale: Locale): void {
  try {
    localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    // Si no se puede persistir, el idioma vale para esta sesión y nada más.
  }
}

function interpolate(template: string, params?: TranslationParams): string {
  if (!params) return template;
  return template.replace(/\{\{(\w+)\}\}/g, (match, name: string) => {
    const value = params[name];
    return value === undefined ? match : String(value);
  });
}

export function translate(
  locale: Locale,
  key: string,
  params?: TranslationParams,
): string {
  const template = CATALOGS[locale][key];
  if (template === undefined) {
    // Devolver la clave la hace visible en pantalla en vez de mostrar un hueco
    // vacío. El test de paridad de catálogos es lo que impide que llegue a
    // produccion.
    return key;
  }
  return interpolate(template, params);
}

export function catalogFor(locale: Locale): Catalog {
  return CATALOGS[locale];
}
