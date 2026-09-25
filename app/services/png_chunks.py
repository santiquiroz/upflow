from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXIF_JPEG_PREFIX = b"Exif\x00\x00"
MAX_KEYWORD_BYTES = 79
IHDR_BIT_DEPTH_OFFSET = 8
# iCCP and sRGB are mutually exclusive in a valid PNG.
CONFLICTING_TYPES = {b"iCCP": {b"iCCP", b"sRGB"}, b"eXIf": {b"eXIf"}}


@dataclass(frozen=True)
class PngChunk:
    type: bytes
    data: bytes

    def encode(self) -> bytes:
        return build_chunk(self.type, self.data)


def build_chunk(chunk_type: bytes, data: bytes) -> bytes:
    if len(chunk_type) != 4 or not chunk_type.isalpha():
        raise ValueError(f"Invalid PNG chunk type: {chunk_type!r}")
    crc = zlib.crc32(chunk_type + data)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def build_iccp_chunk(icc: bytes, name: str = "ICC Profile") -> bytes:
    deflate_method = b"\x00"
    data = _keyword(name) + deflate_method + zlib.compress(icc)
    return build_chunk(b"iCCP", data)


def build_itxt_chunk(keyword: str, text: str) -> bytes:
    uncompressed_without_language = b"\x00\x00\x00\x00"
    data = _keyword(keyword) + uncompressed_without_language + text.encode("utf-8")
    return build_chunk(b"iTXt", data)


def build_exif_chunk(exif: bytes) -> bytes:
    return build_chunk(b"eXIf", exif.removeprefix(EXIF_JPEG_PREFIX))


def parse_chunks(png: bytes) -> list[PngChunk]:
    if not png.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file")
    chunks: list[PngChunk] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(png):
        chunk, offset = _read_chunk(png, offset)
        chunks.append(chunk)
    return chunks


def png_bit_depth(png: bytes) -> int:
    if not png.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file")
    ihdr, _ = _read_chunk(png, len(PNG_SIGNATURE))
    if ihdr.type != b"IHDR":
        raise ValueError("PNG does not start with IHDR")
    return ihdr.data[IHDR_BIT_DEPTH_OFFSET]


def insert_after_ihdr(png: bytes, new_chunks: list[bytes]) -> bytes:
    chunks = parse_chunks(png)
    inserted = [_decode_single_chunk(raw) for raw in new_chunks]
    kept = [chunk for chunk in chunks[1:] if not _is_replaced(chunk, inserted)]
    ordered = [chunks[0], *inserted, *kept]
    return PNG_SIGNATURE + b"".join(chunk.encode() for chunk in ordered)


def _keyword(value: str) -> bytes:
    encoded = value.encode("latin-1")
    if not 1 <= len(encoded) <= MAX_KEYWORD_BYTES or b"\x00" in encoded:
        raise ValueError(f"Invalid PNG keyword: {value!r}")
    return encoded + b"\x00"


def _read_chunk(png: bytes, offset: int) -> tuple[PngChunk, int]:
    if offset + 12 > len(png):
        raise ValueError("Truncated PNG chunk")
    (length,) = struct.unpack(">I", png[offset : offset + 4])
    chunk_type = png[offset + 4 : offset + 8]
    data_end = offset + 8 + length
    data = png[offset + 8 : data_end]
    (crc,) = struct.unpack(">I", png[data_end : data_end + 4])
    if len(data) != length or zlib.crc32(chunk_type + data) != crc:
        raise ValueError(f"Bad CRC in PNG chunk {chunk_type!r}")
    return PngChunk(chunk_type, data), data_end + 4


def _decode_single_chunk(raw: bytes) -> PngChunk:
    chunk, end = _read_chunk(raw, 0)
    if end != len(raw):
        raise ValueError("Expected exactly one PNG chunk")
    return chunk


def _is_replaced(existing: PngChunk, inserted: list[PngChunk]) -> bool:
    return any(_replaces(new, existing) for new in inserted)


def _replaces(new: PngChunk, existing: PngChunk) -> bool:
    if new.type == b"iTXt" and existing.type == b"iTXt":
        return _itxt_keyword(new) == _itxt_keyword(existing)
    return existing.type in CONFLICTING_TYPES.get(new.type, set())


def _itxt_keyword(chunk: PngChunk) -> bytes:
    return chunk.data.split(b"\x00", 1)[0]
