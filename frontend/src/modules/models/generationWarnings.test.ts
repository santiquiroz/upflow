import { expect, it } from "vitest";
import { buildWarnings } from "./generationWarnings";

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
    reason: "Checkpoint SDXL completo.",
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
    reason: "Checkpoint SDXL completo.",
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
