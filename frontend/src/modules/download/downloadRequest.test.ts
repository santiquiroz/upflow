import { describe, expect, it } from "vitest";
import type { MediaProbe } from "../../lib/apiTypes";
import {
  DEFAULT_HEIGHT,
  HEIGHT_OPTIONS,
  MAX_PLAYLIST_ITEMS,
  clampPlaylistLimit,
  formatBytes,
  formatDuration,
  isProbablyUrl,
  offeredHeights,
  playlistNotice,
} from "./downloadRequest";

function makeProbe(overrides: Partial<MediaProbe> = {}): MediaProbe {
  return {
    title: "Un video",
    durationSeconds: 120,
    uploader: "alguien",
    extractor: "Youtube",
    isPlaylist: false,
    entryCount: 1,
    availableHeights: [360, 720, 1080, 2160],
    ...overrides,
  };
}

describe("el techo de calidad por defecto", () => {
  it("no es el más caro de los ofrecidos", () => {
    // Un default en 4K es cómo se llega a esperar horas por algo que nadie pidió.
    expect(DEFAULT_HEIGHT).toBe(1080);
    expect(DEFAULT_HEIGHT).toBeLessThan(Math.max(...HEIGHT_OPTIONS));
  });
});

describe("reconocer una URL", () => {
  it("acepta http y https con dominio", () => {
    expect(isProbablyUrl("https://youtube.com/watch?v=x")).toBe(true);
    expect(isProbablyUrl("  http://vimeo.com/1  ")).toBe(true);
  });

  it("rechaza lo que no es una URL web", () => {
    // El chequeo es de FORMA nada más: el rechazo real de esquemas internos vive en el
    // backend, porque la API es alcanzable sin pasar por esta pantalla.
    expect(isProbablyUrl("file:///C:/secreto.txt")).toBe(false);
    expect(isProbablyUrl("solo texto")).toBe(false);
    expect(isProbablyUrl("")).toBe(false);
  });
});

describe("el aviso de playlist", () => {
  it("no dice nada cuando la URL es un video suelto", () => {
    expect(playlistNotice(makeProbe(), false, 10)).toBeNull();
  });

  it("avisa que la URL es una playlist aunque se baje un solo item", () => {
    // Saberlo importa: explica por qué se bajó uno de doscientos.
    const notice = playlistNotice(makeProbe({ isPlaylist: true, entryCount: 200 }), false, 10);

    expect(notice?.entryCount).toBe(200);
    expect(notice?.willDownload).toBe(1);
    expect(notice?.needsConfirmation).toBe(false);
  });

  it("pide confirmación cuando de verdad van a bajar muchos", () => {
    const notice = playlistNotice(makeProbe({ isPlaylist: true, entryCount: 200 }), true, 50);

    expect(notice?.willDownload).toBe(50);
    expect(notice?.needsConfirmation).toBe(true);
  });

  it("no pide confirmación por unos pocos", () => {
    const notice = playlistNotice(makeProbe({ isPlaylist: true, entryCount: 5 }), true, 10);

    expect(notice?.willDownload).toBe(5);
    expect(notice?.needsConfirmation).toBe(false);
  });

  it("nunca promete más items de los que la playlist tiene", () => {
    const notice = playlistNotice(makeProbe({ isPlaylist: true, entryCount: 3 }), true, 50);

    expect(notice?.willDownload).toBe(3);
  });
});

describe("el límite de playlist", () => {
  it("se recorta al techo que el backend acepta", () => {
    // Dejar mandar 9999 sería un 400 seguro después de completar el formulario.
    expect(clampPlaylistLimit(9999)).toBe(MAX_PLAYLIST_ITEMS);
  });

  it("nunca baja de uno", () => {
    expect(clampPlaylistLimit(0)).toBe(1);
    expect(clampPlaylistLimit(-5)).toBe(1);
  });

  it("sobrevive a un valor no numérico", () => {
    // Un input vacío llega como NaN y dejaría el pedido en un estado imposible.
    expect(clampPlaylistLimit(Number.NaN)).toBe(1);
  });
});

describe("las alturas ofrecidas", () => {
  it("no ofrece más de lo que el video tiene", () => {
    // Ofrecer 4K sobre un video que solo existe en 720p crea una expectativa que no se
    // puede cumplir.
    expect(offeredHeights(makeProbe({ availableHeights: [360, 720] }))).toEqual([360, 480, 720]);
  });

  it("ofrece todo mientras no se sepa nada", () => {
    expect(offeredHeights(null)).toEqual(HEIGHT_OPTIONS);
  });

  it("sigue ofreciendo algo con un video más chico que la opción más baja", () => {
    const offered = offeredHeights(makeProbe({ availableHeights: [144] }));

    expect(offered.length).toBeGreaterThan(0);
  });

  it("ignora una lista de alturas vacía en vez de quedarse sin opciones", () => {
    expect(offeredHeights(makeProbe({ availableHeights: [] }))).toEqual(HEIGHT_OPTIONS);
  });
});

describe("formatos legibles", () => {
  it("muestra tamaños en la unidad que se entiende", () => {
    expect(formatBytes(500)).toBe("500 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });

  it("deja de mostrar decimales cuando el número ya es grande", () => {
    // "118.6 MB" no dice más que "119 MB" y una barra de progreso con el decimal
    // bailando en cada evento se lee peor.
    expect(formatBytes(124386876)).toBe("119 MB");
    expect(formatBytes(50 * 1024 * 1024)).toBe("50.0 MB");
  });

  it("no inventa un tamaño cuando no se conoce", () => {
    // Algunos sitios no publican el total; mostrar 0 sería una mentira.
    expect(formatBytes(null)).toBe("—");
    expect(formatBytes(0)).toBe("—");
  });

  it("muestra duraciones con horas solo cuando las hay", () => {
    expect(formatDuration(95)).toBe("1:35");
    expect(formatDuration(3725)).toBe("1:02:05");
  });

  it("no inventa una duración desconocida", () => {
    expect(formatDuration(null)).toBe("—");
  });
});
