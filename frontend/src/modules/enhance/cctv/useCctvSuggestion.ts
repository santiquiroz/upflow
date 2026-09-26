import { useEffect, useState } from "react";
import type { VideoDimensions } from "../videoOutputResolution";

// Espejo de sniff_container (app/services/media_signature.py): solo los formatos que delatan un DVR.
const SNIFF_BYTES = 16;
const H264_NAL_TYPES: ReadonlySet<number> = new Set([1, 5, 6, 7, 8, 9]);
const HEVC_NAL_TYPES: ReadonlySet<number> = new Set([32, 33, 34, 35, 39, 40]);
const HEVC_LAYER0_TID1 = 0x01;
const FORBIDDEN_ZERO_BIT = 0x80;
const LITE_STORAGE_SIZES: ReadonlyArray<readonly [number, number]> = [
  [960, 1080],
  [640, 720],
];

function startsWithAscii(bytes: Uint8Array, text: string): boolean {
  return [...text].every((char, index) => bytes[index] === char.charCodeAt(0));
}

function startsWithBytes(bytes: Uint8Array, prefix: readonly number[]): boolean {
  return prefix.every((value, index) => bytes[index] === value);
}

function firstNalHeader(bytes: Uint8Array): Uint8Array {
  if (startsWithBytes(bytes, [0, 0, 0, 1])) {
    return bytes.subarray(4, 6);
  }
  if (startsWithBytes(bytes, [0, 0, 1])) {
    return bytes.subarray(3, 5);
  }
  return new Uint8Array();
}

function isHevcNal(header: Uint8Array): boolean {
  if (header.length < 2 || header[0] & FORBIDDEN_ZERO_BIT) {
    return false;
  }
  return HEVC_NAL_TYPES.has((header[0] >> 1) & 0x3f) && header[1] === HEVC_LAYER0_TID1;
}

function isH264Nal(header: Uint8Array): boolean {
  return header.length > 0 && !(header[0] & FORBIDDEN_ZERO_BIT) && H264_NAL_TYPES.has(header[0] & 0x1f);
}

function isMpegPs(bytes: Uint8Array): boolean {
  return startsWithBytes(bytes, [0, 0, 1, 0xba]);
}

export function looksLikeDvrExport(bytes: Uint8Array): boolean {
  if (startsWithAscii(bytes, "IMKH") || startsWithAscii(bytes, "DHAV")) {
    return true;
  }
  const header = firstNalHeader(bytes);
  return !isMpegPs(bytes) && (isHevcNal(header) || isH264Nal(header));
}

export function isLiteStorageSize(dimensions: VideoDimensions | null): boolean {
  return LITE_STORAGE_SIZES.some(([width, height]) => dimensions?.width === width && dimensions?.height === height);
}

export function readFileHead(file: Blob, size: number = SNIFF_BYTES): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file.slice(0, size));
  });
}

interface SniffResult {
  file: File;
  looksLikeDvr: boolean;
}

export function useCctvSuggestion(file: File | null, dimensions: VideoDimensions | null): boolean {
  const [sniffed, setSniffed] = useState<SniffResult | null>(null);

  useEffect(() => {
    if (!file) {
      return;
    }
    let active = true;
    readFileHead(file)
      .then((bytes) => {
        if (active) {
          setSniffed({ file, looksLikeDvr: looksLikeDvrExport(bytes) });
        }
      })
      .catch(() => {
        // Sin cabecera no hay sugerencia: el interruptor sigue a mano.
      });
    return () => {
      active = false;
    };
  }, [file]);

  if (!file) {
    return false;
  }
  const dvrSignature = sniffed?.file === file && sniffed.looksLikeDvr;
  return dvrSignature || isLiteStorageSize(dimensions);
}
