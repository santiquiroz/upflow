import { describe, expect, it } from "vitest";
import { isMultiUrl, parseUrlList } from "./urlList";

describe("parseUrlList", () => {
  it("una sola URL sigue siendo una", () => {
    expect(parseUrlList("https://ejemplo.com/v")).toEqual(["https://ejemplo.com/v"]);
    expect(isMultiUrl("https://ejemplo.com/v")).toBe(false);
  });

  it("separa por líneas y respeta el orden pegado", () => {
    const texto = "https://a.com/1\nhttps://b.com/2\nhttps://c.com/3";
    expect(parseUrlList(texto)).toEqual(["https://a.com/1", "https://b.com/2", "https://c.com/3"]);
    expect(isMultiUrl(texto)).toBe(true);
  });

  it("limpia espacios y líneas en blanco", () => {
    // Copiar de un chat o de un bloc de notas arrastra basura; una línea vacía
    // no es una descarga y encolarla sería un trabajo que falla solo.
    const texto = "  https://a.com/1  \n\n\n   \nhttps://b.com/2\n";
    expect(parseUrlList(texto)).toEqual(["https://a.com/1", "https://b.com/2"]);
  });

  it("los saltos de Windows cuentan igual", () => {
    expect(parseUrlList("https://a.com/1\r\nhttps://b.com/2")).toHaveLength(2);
  });

  it("texto vacío no es un lote", () => {
    expect(parseUrlList("")).toEqual([]);
    expect(isMultiUrl("   \n  ")).toBe(false);
  });
});
