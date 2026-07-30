import { describe, expect, it } from "vitest";
import type { CapabilityResponse } from "../../lib/apiTypes";
import { surfaceFor } from "./capabilityRoutes";

function capability(overrides: Partial<CapabilityResponse> = {}): CapabilityResponse {
  return {
    id: "video.upscale",
    domain: "video",
    labelKey: "capability.video.upscale",
    status: "available",
    provisioning: "registry",
    jobKind: "video",
    strategies: ["model"],
    missingPacks: [],
    unavailableReasonKey: null,
    setupReasonKey: null,
    ...overrides,
  };
}

describe("surfaceFor", () => {
  it("sends both video capabilities to the video surface", () => {
    // Reescalar e interpolar viven en el mismo panel: es el mismo job de video.
    for (const id of ["video.upscale", "video.interpolate"]) {
      expect(surfaceFor(capability({ id, domain: "video" }))).toBe("/enhance/video");
    }
  });

  it("sends image upscaling to the image surface", () => {
    expect(surfaceFor(capability({ id: "image.upscale", domain: "image" }))).toBe(
      "/enhance/image",
    );
  });

  it("sends audio enhancement capabilities to the audio surface", () => {
    for (const id of ["audio.denoise", "audio.restore", "audio.voice"]) {
      expect(surfaceFor(capability({ id, domain: "audio", jobKind: "audio" }))).toBe(
        "/audio",
      );
    }
  });

  it("sends audio transcription to the transcribe surface", () => {
    expect(
      surfaceFor(
        capability({
          id: "audio.transcribe",
          domain: "audio",
          jobKind: "transcribe",
        }),
      ),
    ).toBe("/transcribe");
  });

  it("sends text to image to the generate surface", () => {
    expect(
      surfaceFor(
        capability({ id: "generate.textToImage", domain: "generate", jobKind: "generation" }),
      ),
    ).toBe("/generate");
  });

  it("has no surface for a capability without a job kind", () => {
    // Sin job_kind no hay a donde mandar trabajo, asi que abrir una pantalla
    // seria llevar al usuario a algo que no hace nada.
    expect(
      surfaceFor(capability({ id: "audio.stems", domain: "audio", jobKind: null })),
    ).toBeNull();
  });

  it("ignores the map when the job kind is missing", () => {
    // Doble gate a proposito: agregar una entrada al mapa por error no puede
    // abrir la pantalla de una capacidad que no existe todavia.
    expect(surfaceFor(capability({ id: "video.upscale", jobKind: null }))).toBeNull();
  });

  it("has no surface for an unmapped id", () => {
    expect(surfaceFor(capability({ id: "video.inventado" }))).toBeNull();
  });
});
