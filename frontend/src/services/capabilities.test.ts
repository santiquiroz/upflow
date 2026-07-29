import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  CapabilityDomainId,
  CapabilityTreeResponse,
} from "../lib/apiTypes";
import { fetchCapabilityTree } from "./capabilities";

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
