from __future__ import annotations

import io
import struct
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image

from app.services.png_chunks import (
    PNG_SIGNATURE,
    build_chunk,
    build_exif_chunk,
    build_iccp_chunk,
    build_itxt_chunk,
    insert_after_ihdr,
    parse_chunks,
    png_bit_depth,
)


def encoded_png(depth: int = 8, size: tuple[int, int] = (6, 4)) -> bytes:
    dtype = np.uint16 if depth == 16 else np.uint8
    pixels = np.full((size[1], size[0], 3), 7, dtype=dtype)
    ok, data = cv2.imencode(".png", pixels)
    assert ok
    return data.tobytes()


def test_build_chunk_writes_length_type_data_and_crc() -> None:
    chunk = build_chunk(b"tEXt", b"abc")

    assert chunk[:4] == struct.pack(">I", 3)
    assert chunk[4:8] == b"tEXt"
    assert chunk[8:11] == b"abc"
    assert chunk[11:] == struct.pack(">I", zlib.crc32(b"tEXtabc"))


def test_build_chunk_rejects_invalid_type() -> None:
    with pytest.raises(ValueError):
        build_chunk(b"toolong", b"")


def test_parse_chunks_round_trips_an_encoded_png() -> None:
    chunks = parse_chunks(encoded_png())

    assert chunks[0].type == b"IHDR"
    assert chunks[-1].type == b"IEND"
    assert any(chunk.type == b"IDAT" for chunk in chunks)


def test_parse_chunks_rejects_a_bad_crc() -> None:
    data = bytearray(encoded_png())
    data[len(PNG_SIGNATURE) + 8] ^= 0xFF

    with pytest.raises(ValueError, match="CRC"):
        parse_chunks(bytes(data))


def test_parse_chunks_rejects_non_png() -> None:
    with pytest.raises(ValueError):
        parse_chunks(b"not a png")


@pytest.mark.parametrize("depth", [8, 16])
def test_png_bit_depth_reads_ihdr(depth: int) -> None:
    assert png_bit_depth(encoded_png(depth)) == depth


def test_inserted_chunks_follow_ihdr_and_precede_idat() -> None:
    new = [build_itxt_chunk("Comment", "hola"), build_iccp_chunk(b"fake-icc")]

    result = insert_after_ihdr(encoded_png(), new)
    types = [chunk.type for chunk in parse_chunks(result)]

    assert types[0] == b"IHDR"
    assert types[1:3] == [b"iTXt", b"iCCP"]
    assert types.index(b"iCCP") < types.index(b"IDAT")


def test_inserting_iccp_drops_existing_srgb_and_iccp() -> None:
    with_srgb = insert_after_ihdr(encoded_png(), [build_chunk(b"sRGB", b"\x00")])

    result = insert_after_ihdr(with_srgb, [build_iccp_chunk(b"icc")])
    types = [chunk.type for chunk in parse_chunks(result)]

    assert b"sRGB" not in types
    assert types.count(b"iCCP") == 1


def test_inserting_an_itxt_replaces_only_the_same_keyword() -> None:
    first = insert_after_ihdr(
        encoded_png(),
        [build_itxt_chunk("Comment", "keep"), build_itxt_chunk("XML:com.adobe.xmp", "old")],
    )

    result = insert_after_ihdr(first, [build_itxt_chunk("XML:com.adobe.xmp", "new")])
    image = Image.open(io.BytesIO(result))
    image.load()

    assert image.info["Comment"] == "keep"
    assert image.info["XML:com.adobe.xmp"] == "new"


@pytest.mark.parametrize("depth", [8, 16])
def test_pillow_reads_the_injected_icc_exif_and_xmp(depth: int) -> None:
    icc = b"\x00\x01icc-profile-bytes" * 20
    exif = Image.Exif()
    exif[0x010E] = "restaurada"
    new = [
        build_iccp_chunk(icc),
        build_exif_chunk(exif.tobytes()),
        build_itxt_chunk("XML:com.adobe.xmp", "<x:xmpmeta/>"),
    ]

    image = Image.open(io.BytesIO(insert_after_ihdr(encoded_png(depth), new)))
    image.load()

    assert image.info["icc_profile"] == icc
    assert image.getexif()[0x010E] == "restaurada"
    assert image.info["XML:com.adobe.xmp"] == "<x:xmpmeta/>"


def test_exif_chunk_strips_the_jpeg_style_prefix() -> None:
    chunk = build_exif_chunk(b"Exif\x00\x00MM\x00\x2a")

    assert chunk[8:12] == b"MM\x00\x2a"


def test_iccp_rejects_a_long_profile_name() -> None:
    with pytest.raises(ValueError):
        build_iccp_chunk(b"icc", name="x" * 80)
