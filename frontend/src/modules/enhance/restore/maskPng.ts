import type { BinaryMask } from "../../editor/maskCanvas";

// PNG de 1 bit en escala de grises: PIL lo abre en modo "1" (blanco = reparar,
// negro = conservar), que es lo que acepta restore_session.mask_from_png. Un
// canvas.toBlob daria RGBA y el backend lo rechazaria.
const SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const BIT_DEPTH = 1;
const GRAYSCALE = 0;
const NO_FILTER = 0;
const BITS_PER_BYTE = 8;

export type Deflate = (bytes: Uint8Array<ArrayBuffer>) => Promise<Uint8Array>;

const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < BITS_PER_BYTE; bit += 1) {
    value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  }
  return value >>> 0;
});

export function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function uint32(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function chunk(type: string, body: Uint8Array): Uint8Array {
  const typed = new Uint8Array([...Array.from(type, (char) => char.charCodeAt(0)), ...body]);
  const out = new Uint8Array(4 + typed.length + 4);
  out.set(uint32(body.length), 0);
  out.set(typed, 4);
  out.set(uint32(crc32(typed)), 4 + typed.length);
  return out;
}

function header(mask: BinaryMask): Uint8Array {
  return new Uint8Array([...uint32(mask.width), ...uint32(mask.height), BIT_DEPTH, GRAYSCALE, 0, 0, 0]);
}

export function packedRows(mask: BinaryMask): Uint8Array<ArrayBuffer> {
  const rowBytes = Math.ceil(mask.width / BITS_PER_BYTE);
  const stride = rowBytes + 1;
  const out = new Uint8Array(stride * mask.height);
  for (let y = 0; y < mask.height; y += 1) {
    out[y * stride] = NO_FILTER;
    for (let x = 0; x < mask.width; x += 1) {
      if (mask.data[y * mask.width + x]) {
        out[y * stride + 1 + (x >> 3)] |= 0x80 >> (x & 7);
      }
    }
  }
  return out;
}

function concat(parts: Uint8Array[]): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

// "deflate" de CompressionStream es zlib (RFC 1950), justo lo que pide IDAT.
export async function deflateBytes(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array> {
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  const compressed = source.pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(compressed).arrayBuffer());
}

export async function maskPngBytes(
  mask: BinaryMask,
  deflate: Deflate = deflateBytes,
): Promise<Uint8Array<ArrayBuffer>> {
  const compressed = await deflate(packedRows(mask));
  return concat([
    new Uint8Array(SIGNATURE),
    chunk("IHDR", header(mask)),
    chunk("IDAT", compressed),
    chunk("IEND", new Uint8Array(0)),
  ]);
}

export async function encodeMaskPng(mask: BinaryMask, deflate: Deflate = deflateBytes): Promise<Blob> {
  return new Blob([await maskPngBytes(mask, deflate)], { type: "image/png" });
}
