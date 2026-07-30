import { describe, expect, it } from "vitest";
import type { UpscalerPreflightResponse } from "../../lib/apiTypes";
import {
  UPSCALER_CONVERSION_DISK_FACTOR,
  UPSCALER_DIRECT_DISK_FACTOR,
  buildUpscalerWarnings,
} from "./upscalerWarnings";

const GB = 1024 ** 3;

const base: UpscalerPreflightResponse = {
  repoId: "owner/model",
  compat: "ready_onnx",
  compatReasonKey: "compat.upscaler.readyOnnx",
  compatReasonParams: {},
  degraded: false,
  downloadBytes: 2 * GB,
  devices: [
    {
      id: "dml:0",
      name: "RX 7800 XT",
      kind: "gpu",
      freeVramBytes: 12 * GB,
    },
  ],
  disk: { targetPath: "D:\\models", freeBytes: 20 * GB },
  freeRamBytes: 24 * GB,
};

describe("buildUpscalerWarnings", () => {
  it("returns only the degraded notice when evaluation failed", () => {
    expect(
      buildUpscalerWarnings({
        ...base,
        degraded: true,
        compat: null,
      }),
    ).toEqual([
      {
        code: "degraded",
        key: "upscaler.warning.degraded",
        params: {},
      },
    ]);
  });

  it("warns when the repository is gated", () => {
    expect(
      buildUpscalerWarnings({ ...base, compat: "gated" }),
    ).toContainEqual({
      code: "gated",
      key: "upscaler.warning.gated",
      params: {},
    });
  });

  it("uses the backend incompatibility key and params unchanged", () => {
    const reasonParams = { filename: "weights.bin" };
    expect(
      buildUpscalerWarnings({
        ...base,
        compat: "incompatible",
        compatReasonKey: "compat.upscaler.noWeights",
        compatReasonParams: reasonParams,
      }),
    ).toContainEqual({
      code: "incompatible",
      key: "compat.upscaler.noWeights",
      params: reasonParams,
    });
  });

  it("falls back only when an incompatible response has no reason key", () => {
    expect(
      buildUpscalerWarnings({
        ...base,
        compat: "incompatible",
        compatReasonKey: null,
      }),
    ).toContainEqual({
      code: "incompatible",
      key: "upscaler.warning.incompatibleFallback",
      params: {},
    });
  });

  it("warns when disk cannot cover a conversion, which needs twice the download", () => {
    // Convertir deja el archivo descargado y el .onnx resultante en disco a la vez.
    const warnings = buildUpscalerWarnings({
      ...base,
      compat: "needs_conversion",
      compatReasonKey: "compat.upscaler.needsConversion",
      disk: { targetPath: "D:\models", freeBytes: 3 * GB },
    });

    expect(UPSCALER_CONVERSION_DISK_FACTOR).toBe(2);
    expect(warnings).toContainEqual({
      code: "disk_low",
      key: "upscaler.warning.diskLow",
      params: {
        free: "3.0 GB",
        path: "D:\models",
        needed: "4.0 GB",
      },
    });
  });

  it("does not warn about the conversion margin for a repo that ships onnx", () => {
    // Un .onnx no se convierte: se baja a un nombre de staging y se renombra, que
    // es UNA copia. Pedir el doble avisaria de un espacio que nunca se usa.
    const warnings = buildUpscalerWarnings({
      ...base,
      compat: "ready_onnx",
      disk: { targetPath: "D:\models", freeBytes: 3 * GB },
    });

    expect(UPSCALER_DIRECT_DISK_FACTOR).toBe(1);
    expect(warnings.map((warning) => warning.code)).not.toContain("disk_low");
  });

  it("still warns about a repo that ships onnx when even one copy does not fit", () => {
    const warnings = buildUpscalerWarnings({
      ...base,
      compat: "ready_onnx",
      disk: { targetPath: "D:\models", freeBytes: 1 * GB },
    });

    expect(warnings.map((warning) => warning.code)).toContain("disk_low");
  });

  it("does not warn when disk covers the conversion margin", () => {
    expect(
      buildUpscalerWarnings({
        ...base,
        disk: { targetPath: "D:\\models", freeBytes: 4 * GB },
      }).map((warning) => warning.code),
    ).not.toContain("disk_low");
  });

  it("never warns about disk when disk is null", () => {
    expect(
      buildUpscalerWarnings({ ...base, disk: null }).map(
        (warning) => warning.code,
      ),
    ).not.toContain("disk_low");
  });

  it("never warns about disk when downloadBytes is null", () => {
    expect(
      buildUpscalerWarnings({ ...base, downloadBytes: null }).map(
        (warning) => warning.code,
      ),
    ).not.toContain("disk_low");
  });

  it("never warns when freeVramBytes is null", () => {
    const warnings = buildUpscalerWarnings({
      ...base,
      devices: [{ ...base.devices[0], freeVramBytes: null }],
    });

    expect(warnings).toEqual([]);
  });

  it("never warns when freeRamBytes is null", () => {
    expect(
      buildUpscalerWarnings({ ...base, freeRamBytes: null }),
    ).toEqual([]);
  });

  it("does not invent a VRAM warning from a measured low value", () => {
    const warnings = buildUpscalerWarnings({
      ...base,
      devices: [{ ...base.devices[0], freeVramBytes: 1 }],
    });

    expect(warnings).toEqual([]);
  });
});
