import { describe, expect, it } from "vitest";
import { exceedsUploadLimit, formatMegabytes } from "./uploadLimit";

// El servidor corta la subida MIENTRAS la recibe (`storage.save_upload`), asi
// que sin este chequeo un archivo de 3 GB se sube entero antes de que alguien
// diga que no entra. El limite se compara con la MISMA cuenta que usa el
// servidor: `limit_mb * 1024 * 1024`, y el que da justo entra.

const MB = 1024 * 1024;

describe("exceedsUploadLimit", () => {
  it("accepts a file below the limit", () => {
    expect(exceedsUploadLimit(50 * MB, 2048)).toBe(false);
  });

  it("accepts a file exactly at the limit, like the server does", () => {
    // El servidor compara con `>`, asi que el que da justo pasa. Si aca se
    // usara `>=` la app rechazaria archivos que el servidor si acepta.
    expect(exceedsUploadLimit(2048 * MB, 2048)).toBe(false);
  });

  it("rejects a file one byte over the limit", () => {
    expect(exceedsUploadLimit(2048 * MB + 1, 2048)).toBe(true);
  });

  it("does not reject anything when the limit is unknown", () => {
    // Antes de que responda /engine no hay limite que aplicar. Rechazar por las
    // dudas seria peor que dejar que el servidor decida.
    expect(exceedsUploadLimit(9999 * MB, null)).toBe(false);
  });
});

describe("formatMegabytes", () => {
  it("keeps whole megabytes readable", () => {
    expect(formatMegabytes(50 * MB)).toBe("50 MB");
  });

  it("switches to gigabytes when megabytes stop being readable", () => {
    expect(formatMegabytes(3 * 1024 * MB)).toBe("3 GB");
  });

  it("keeps one decimal for a size between whole gigabytes", () => {
    expect(formatMegabytes(2560 * MB)).toBe("2.5 GB");
  });
});
