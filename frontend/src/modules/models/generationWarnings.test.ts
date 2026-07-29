import { expect, it } from "vitest";
import { translate } from "../../i18n";
import { SINGLE_FILE_DISK_PEAK_FACTOR, buildWarnings } from "./generationWarnings";

const GB = 1024 ** 3;

const base = {
  repoId: "owner/name",
  compat: "needs_conversion" as const,
  compatReason: "Sin ONNX propio para unet",
  degraded: false,
  referenceWidth: 512,
  referenceHeight: 512,
  precisions: [{ precision: "fp16" as const, downloadBytes: 3 * GB, estimatedPeakBytes: 4 * GB }],
  devices: [
    { id: "dml:0", name: "RX 7900 XTX", kind: "gpu", freeVramBytes: 22 * GB },
    { id: "dml:1", name: "RX 6600", kind: "gpu", freeVramBytes: 3 * GB },
    { id: "cpu", name: "CPU", kind: "cpu", freeVramBytes: null },
  ],
  disk: { targetPath: "D:\\temp", freeBytes: 50 * GB },
  checkpoints: [],
  freeRamBytes: null,
};

it("warns about devices where the estimate does not fit", () => {
  const codes = buildWarnings(base, "fp16").map((w) => w.code);
  expect(codes).toContain("device_wont_fit");
});

it("does not warn about a device with room to spare", () => {
  const fits = { ...base, devices: [base.devices[0]] };
  expect(buildWarnings(fits, "fp16").map((w) => w.code)).not.toContain("device_wont_fit");
});

it("warns when free disk is below the download size", () => {
  const tight = { ...base, disk: { targetPath: "D:\\temp", freeBytes: 1 * GB } };
  expect(buildWarnings(tight, "fp16").map((w) => w.code)).toContain("disk_low");
});

it("never warns about disk when it could not be measured", () => {
  const noDisk = { ...base, disk: null };
  expect(buildWarnings(noDisk, "fp16").map((w) => w.code)).not.toContain("disk_low");
});

it("warns when free RAM is below the chosen checkpoint size", () => {
  const checkpoint = {
    path: "pony.safetensors",
    sizeBytes: 7 * GB,
    architecture: "xl_base",
    installable: true,
    reasonKey: "checkpoint.ready",
    reasonParams: { architecture: "xl_base" },
  };
  const tightRam = { ...base, freeRamBytes: 6 * GB };
  expect(buildWarnings(tightRam, "fp16", checkpoint).map((w) => w.code)).toContain("ram_low");
});

it("never warns about RAM when it could not be measured", () => {
  const checkpoint = {
    path: "pony.safetensors",
    sizeBytes: 7 * GB,
    architecture: "xl_base",
    installable: true,
    reasonKey: "checkpoint.ready",
    reasonParams: { architecture: "xl_base" },
  };
  expect(buildWarnings(base, "fp16", checkpoint).map((w) => w.code)).not.toContain("ram_low");
});

it("never warns about a device whose VRAM could not be measured", () => {
  const unmeasured = {
    ...base,
    devices: [{ id: "dml:0", name: "GPU", kind: "gpu", freeVramBytes: null }],
  };
  expect(buildWarnings(unmeasured, "fp16").map((w) => w.code)).not.toContain("device_wont_fit");
});

it("warns that CPU generation is slow", () => {
  expect(buildWarnings(base, "fp16").map((w) => w.code)).toContain("cpu_slow");
});

it("warns when the repo is gated", () => {
  const gated = { ...base, compat: "gated" as const };
  expect(buildWarnings(gated, "fp16").map((w) => w.code)).toContain("gated");
});

it("returns a degraded notice and nothing else when preflight failed", () => {
  const degraded = { ...base, degraded: true, precisions: [], compat: null };
  expect(buildWarnings(degraded, "fp16").map((w) => w.code)).toEqual(["degraded"]);
});

describe("aviso de disco para checkpoints single-file", () => {
  const checkpoint = {
    path: "pony.safetensors",
    sizeBytes: 6 * GB,
    architecture: "xl_base",
    installable: true,
    reasonKey: "checkpoint.ready",
    reasonParams: { architecture: "xl_base" },
  };
  const singleFile = {
    ...base,
    compat: "single_file" as const,
    precisions: [],
    checkpoints: [checkpoint],
  };

  it("avisa cuando el pico de 4x no entra, aunque el checkpoint solo si entre", () => {
    // Medido: el pico es ~4x el checkpoint, no su tamano. Con 20 GB libres un
    // checkpoint de 6 GB "entra" pero su conversion necesita ~24 GB.
    const tight = { ...singleFile, disk: { targetPath: "D:\temp", freeBytes: 20 * GB } };
    const codes = buildWarnings(tight, "fp16", checkpoint).map((w) => w.code);
    expect(codes).toContain("disk_low");
  });

  it("no avisa cuando el pico entra holgado", () => {
    const roomy = { ...singleFile, disk: { targetPath: "D:\temp", freeBytes: 200 * GB } };
    const codes = buildWarnings(roomy, "fp16", checkpoint).map((w) => w.code);
    expect(codes).not.toContain("disk_low");
  });

  it("nunca avisa de disco si no se pudo medir", () => {
    const noDisk = { ...singleFile, disk: null };
    const codes = buildWarnings(noDisk, "fp16", checkpoint).map((w) => w.code);
    expect(codes).not.toContain("disk_low");
  });

  it("el mensaje nombra el pico, no el tamano del checkpoint", () => {
    const tight = { ...singleFile, disk: { targetPath: "D:\temp", freeBytes: 20 * GB } };
    const warning = buildWarnings(tight, "fp16", checkpoint).find((w) => w.code === "disk_low");
    const message = warning
      ? translate("en", warning.key, warning.params)
      : undefined;
    expect(message).toContain("about 24.0 GB of peak disk space");
    expect(message).not.toContain("6.0 GB required");
  });

  it("el factor medido esta expuesto como constante revisable", () => {
    expect(SINGLE_FILE_DISK_PEAK_FACTOR).toBe(4);
  });
});
