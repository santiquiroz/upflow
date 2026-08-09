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
    for (const id of ["audio.denoise", "audio.restore", "audio.restoreSr", "audio.voice"]) {
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

  it("solo lleva a rutas que existen de verdad", () => {
    // Antes habia un doble porton por `jobKind === null`. Servia de sustituto de
    // "la pantalla no existe", pero era falso para las capacidades sincronicas
    // —la voz devuelve el audio en la misma peticion— que quedaban listadas sin
    // llevar a ningun lado. Esto chequea lo que ese porton queria proteger, y lo
    // chequea de verdad: que el destino sea una ruta real de la app.
    const RUTAS_REALES = [
      "/enhance/video",
      "/enhance/image",
      "/audio",
      "/transcribe",
      "/generate",
      "/voice",
    ];
    const IDS = [
      "video.upscale",
      "video.interpolate",
      "image.upscale",
      "audio.denoise",
      "audio.restore",
      "audio.restoreSr",
      "audio.voice",
      "audio.transcribe",
      "audio.speak",
      "audio.voiceConvert",
      "generate.textToImage",
      "generate.textToVideo",
    ];
    for (const id of IDS) {
      const destino = surfaceFor(capability({ id }));
      expect(RUTAS_REALES, `${id} -> ${destino}`).toContain(destino);
    }
  });

  it("has no surface for an unmapped id", () => {
    expect(surfaceFor(capability({ id: "video.inventado" }))).toBeNull();
  });
});

describe("capacidades sincronicas", () => {
  // La sintesis de voz no encola: devuelve el audio en la misma peticion. Pero
  // SI tiene pantalla. El porton usaba `jobKind === null` como sustituto de "no
  // tiene a donde ir", y para estas eso es falso — quedaban listadas en Tareas
  // sin llevar a ningun lado.
  it("la sintesis de voz lleva a la pantalla de Voz", () => {
    expect(
      surfaceFor(capability({ id: "audio.speak", domain: "audio", jobKind: null })),
    ).toBe("/voice");
  });

  it("la conversion de voz lleva a la pantalla de Voz", () => {
    expect(
      surfaceFor(capability({ id: "audio.voiceConvert", domain: "audio", jobKind: null })),
    ).toBe("/voice");
  });

  it("una capacidad que no esta en el mapa sigue sin llevar a ningun lado", () => {
    expect(surfaceFor(capability({ id: "print.slice", domain: "print", jobKind: null }))).toBeNull();
  });
});
