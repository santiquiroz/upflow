import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchLicenses, fetchPackLicense, type LicensesResponse } from "./licenses";

const BODY: LicensesResponse = {
  packs: [],
  thirdParty: [{ title: "Real-ESRGAN ONNX exports", section: "Bundled", fields: {}, licenseText: null }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchLicenses", () => {
  it("reads the installed packs and third-party notices from the licenses endpoint", async () => {
    const response = new Response(JSON.stringify(BODY), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    const licenses = await fetchLicenses();

    expect(String(vi.mocked(fetch).mock.calls[0][0])).toMatch(/\/api\/v1\/licenses$/);
    expect(licenses).toEqual(BODY);
  });
});

describe("fetchPackLicense", () => {
  it("reads the license gate of one pack", async () => {
    const body = { pack: "restore-faces-nc", gated: true, licenseText: "S-Lab License 1.0" };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } }),
      ),
    );

    const gate = await fetchPackLicense("restore-faces-nc");

    expect(String(vi.mocked(fetch).mock.calls[0][0])).toMatch(/\/api\/v1\/packs\/restore-faces-nc\/license$/);
    expect(gate).toEqual(body);
  });
});
