import { describe, expect, it } from "vitest";
import { LOCALES, catalogFor, isLocale, translate } from ".";
import { en } from "./en";
import { es } from "./es";

describe("paridad de catálogos", () => {
  it("los dos idiomas tienen exactamente las mismas claves", () => {
    // ESTE es el test que hace viable i18n a mano. El único fallo real del
    // enfoque es agregar una clave en un idioma y olvidarla en el otro, y una
    // igualdad de conjuntos lo mata.
    const enKeys = Object.keys(en).sort();
    const esKeys = Object.keys(es).sort();

    const missingInEs = enKeys.filter((k) => !(k in es));
    const missingInEn = esKeys.filter((k) => !(k in en));

    expect(missingInEs, "claves que faltan en español").toEqual([]);
    expect(missingInEn, "claves que faltan en inglés").toEqual([]);
  });

  it("ninguna traducción queda vacía", () => {
    for (const locale of LOCALES) {
      for (const [key, value] of Object.entries(catalogFor(locale))) {
        expect(value.trim(), `${locale}:${key}`).not.toBe("");
      }
    }
  });

  it("los parámetros de una clave son los mismos en los dos idiomas", () => {
    // Un {{param}} presente en un idioma y ausente en el otro deja un hueco en
    // la oración traducida, que es un fallo silencioso.
    const paramsOf = (template: string) =>
      [...template.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]).sort();

    for (const key of Object.keys(en)) {
      expect(paramsOf(es[key as keyof typeof es]), `parámetros de ${key}`).toEqual(
        paramsOf(en[key as keyof typeof en]),
      );
    }
  });
});

describe("translate", () => {
  it("interpola los parámetros que recibe", () => {
    const message = translate("en", "generation.warning.diskLow", {
      free: "6.2 GB",
      path: "D:\\temp",
      needed: "2.6 GB",
    });
    expect(message).toContain("6.2 GB");
    expect(message).toContain("D:\\temp");
    expect(message).toContain("2.6 GB");
  });

  it("deja el marcador intacto si falta un parámetro", () => {
    // Preferible a mostrar "undefined": el hueco es evidente al mirarlo.
    const message = translate("en", "generation.warning.diskLow", { free: "6.2 GB" });
    expect(message).toContain("{{path}}");
  });

  it("devuelve la clave cuando no existe la traducción", () => {
    expect(translate("en", "clave.inexistente")).toBe("clave.inexistente");
  });

  it("no toca un texto sin parámetros", () => {
    expect(translate("es", "generation.warning.cpuSlow")).toBe(
      "En CPU tarda varios minutos por imagen.",
    );
  });

  it("traduce la misma clave distinto en cada idioma", () => {
    expect(translate("es", "voice.step.deesser.label")).not.toBe(
      translate("en", "voice.step.deesser.label"),
    );
  });
});

describe("isLocale", () => {
  it("acepta solo los idiomas soportados", () => {
    expect(isLocale("es")).toBe(true);
    expect(isLocale("en")).toBe(true);
    expect(isLocale("pt")).toBe(false);
    expect(isLocale(null)).toBe(false);
    expect(isLocale(undefined)).toBe(false);
  });
});
