import { describe, expect, it } from "vitest";
import type { BinaryMask } from "../../editor/maskCanvas";
import { crc32, deflateBytes, encodeMaskPng, maskPngBytes, packedRows } from "./maskPng";

async function inflate(bytes: Uint8Array<ArrayBuffer>): Promise<Uint8Array> {
  const source = new ReadableStream<BufferSource>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
  return new Uint8Array(await new Response(source.pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
}

interface PngChunk {
  type: string;
  body: Uint8Array<ArrayBuffer>;
  crc: number;
}

function readChunks(bytes: Uint8Array<ArrayBuffer>): PngChunk[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunks: PngChunk[] = [];
  for (let offset = 8; offset < bytes.length; ) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(...bytes.subarray(offset + 4, offset + 8));
    const body = bytes.subarray(offset + 8, offset + 8 + length);
    chunks.push({ type, body, crc: view.getUint32(offset + 8 + length) });
    offset += 12 + length;
  }
  return chunks;
}

const MASK: BinaryMask = { width: 10, height: 2, data: Uint8Array.from([1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) };

describe("crc32", () => {
  it("matches the reference value of the PNG spec", () => {
    expect(crc32(new TextEncoder().encode("IEND"))).toBe(0xae426082);
  });
});

describe("packedRows", () => {
  it("packs each row MSB first behind a no-filter byte", () => {
    expect(Array.from(packedRows(MASK))).toEqual([0, 0b10000000, 0b11000000, 0, 0, 0b01000000]);
  });
});

describe("maskPngBytes", () => {
  it("writes a 1-bit grayscale PNG whose pixels are the mask", async () => {
    const bytes = await maskPngBytes(MASK);

    expect(Array.from(bytes.subarray(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    const chunks = readChunks(bytes);
    expect(chunks.map((chunk) => chunk.type)).toEqual(["IHDR", "IDAT", "IEND"]);
    expect(Array.from(chunks[0].body)).toEqual([0, 0, 0, 10, 0, 0, 0, 2, 1, 0, 0, 0, 0]);
    for (const chunk of chunks) {
      const typed = new Uint8Array([...new TextEncoder().encode(chunk.type), ...chunk.body]);
      expect(chunk.crc).toBe(crc32(typed));
    }
    expect(Array.from(await inflate(chunks[1].body))).toEqual(Array.from(packedRows(MASK)));
  });

  it("uses the deflate it is given", async () => {
    const bytes = await maskPngBytes(MASK, async () => new Uint8Array([7, 7]));

    expect(Array.from(readChunks(bytes)[1].body)).toEqual([7, 7]);
  });
});

describe("encodeMaskPng", () => {
  it("returns a PNG blob", async () => {
    const blob = await encodeMaskPng(MASK, deflateBytes);

    expect(blob.type).toBe("image/png");
    expect(blob.size).toBeGreaterThan(8);
  });
});
