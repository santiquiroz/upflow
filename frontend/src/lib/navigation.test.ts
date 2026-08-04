import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { es } from "../i18n/es";
import { NAV_ENTRIES } from "./navigation";

describe("NAV_ENTRIES", () => {
  it("exposes the transcription page in primary navigation", () => {
    expect(NAV_ENTRIES).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          labelKey: "nav.transcribe",
          path: "/transcribe",
        }),
      ]),
    );
  });

  // La barra lateral quedo en ingles mientras las pantallas se iban escribiendo
  // en espanol: navegar cambiaba de idioma a mitad de camino. Una clave que no
  // este en los dos catalogos vuelve a dejar media barra sin traducir, y en
  // pantalla eso se ve como el nombre de la clave.
  it("has every label in both catalogs", () => {
    const missing = NAV_ENTRIES.flatMap((entry) => [
      ...(entry.labelKey in en ? [] : [`en: ${entry.labelKey}`]),
      ...(entry.labelKey in es ? [] : [`es: ${entry.labelKey}`]),
    ]);

    expect(missing).toEqual([]);
  });
});
