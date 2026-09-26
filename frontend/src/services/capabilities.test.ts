import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  CapabilityDomainId,
  CapabilityTreeResponse,
} from "../lib/apiTypes";
import { fetchCapabilityTree, provisionPack, provisionQuery } from "./capabilities";

function mockFetchOnce(body: unknown, init: ResponseInit = { status: 200 }) {
  const response = new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchCapabilityTree", () => {
  it("issues a GET to /api/v1/capabilities/tree and returns the typed payload", async () => {
    const domains: CapabilityDomainId[] = [
      "video",
      "image",
      "audio",
      "generate",
    ];
    const payload: CapabilityTreeResponse = {
      domains: domains.map((domain) => ({
        domain,
        labelKey: `capability.domain.${domain}`,
        capabilities: [],
        roadmap: [],
      })),
    };
    mockFetchOnce(payload);

    const result = await fetchCapabilityTree();

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/capabilities/tree",
      expect.objectContaining({ method: "GET" }),
    );
    expect(result).toEqual(payload);
  });
});

describe("provisionQuery", () => {
  it("is empty when there is no variant and the license is not accepted", () => {
    expect(provisionQuery(undefined, false)).toBe("");
  });

  it("carries the variant encoded", () => {
    expect(provisionQuery("es-en", false)).toBe("?variant=es-en");
  });

  it("carries the license acceptance by the name the backend reads", () => {
    expect(provisionQuery(undefined, true)).toBe("?acceptLicense=true");
    expect(provisionQuery("fp16", true)).toBe("?variant=fp16&acceptLicense=true");
  });
});

describe("provisionPack", () => {
  it("posts the acceptance flag only when the license was accepted", async () => {
    mockFetchOnce({ jobId: "j", pack: "restore-faces-nc", status: "queued", error: null, statusUrl: "/x" });

    await provisionPack("restore-faces-nc", undefined, true);

    expect(String(vi.mocked(fetch).mock.calls[0][0])).toMatch(
      /\/api\/v1\/packs\/restore-faces-nc\/provision\?acceptLicense=true$/,
    );
  });
});
