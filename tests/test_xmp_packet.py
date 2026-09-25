from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app.services.png_chunks import parse_chunks
from app.services.xmp_packet import (
    DIGITAL_SOURCE_ENHANCED,
    DIGITAL_SOURCE_COMPOSITE,
    XmpFields,
    build_xmp_packet,
    extract_xmp,
    insert_xmp_into_jpeg,
    insert_xmp_into_png,
    normalize_photo_date,
    read_xmp_properties,
)


def fields(**overrides: object) -> XmpFields:
    base = {
        "digital_source_type": DIGITAL_SOURCE_COMPOSITE,
        "description": "Restored: scratch fill <MI-GAN> & faces",
        "creator_tool": "Upflow 0.80.0",
        "date_created": None,
    }
    return XmpFields(**{**base, **overrides})


def jpeg_bytes(exif: bool = False) -> bytes:
    buffer = io.BytesIO()
    options = {"exif": Image.Exif().tobytes()} if exif else {}
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, "JPEG", **options)
    return buffer.getvalue()


def test_packet_round_trips_every_field() -> None:
    packet = build_xmp_packet(fields(date_created="1974-06"))

    props = read_xmp_properties(packet)

    assert props["Iptc4xmpExt:DigitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert props["dc:description"] == "Restored: scratch fill <MI-GAN> & faces"
    assert props["xmp:CreatorTool"] == "Upflow 0.80.0"
    assert props["photoshop:DateCreated"] == "1974-06"


def test_packet_omits_date_created_when_not_given() -> None:
    props = read_xmp_properties(build_xmp_packet(fields()))

    assert "photoshop:DateCreated" not in props


def test_packet_is_wrapped_in_xpacket_markers() -> None:
    packet = build_xmp_packet(fields())

    assert packet.startswith("<?xpacket begin=")
    assert packet.rstrip().endswith('<?xpacket end="w"?>')


def test_reading_a_packet_with_a_dtd_is_refused() -> None:
    hostile = '<!DOCTYPE x [<!ENTITY a "aaaa">]>' + build_xmp_packet(fields())

    with pytest.raises(ValueError, match="DTD"):
        read_xmp_properties(hostile)


def test_unknown_digital_source_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_xmp_packet(fields(digital_source_type="http://example.com/other"))


@pytest.mark.parametrize("value", ["1974", "1974-06", "1974-06-30", " 1974-06 "])
def test_partial_iso_dates_are_accepted(value: str) -> None:
    assert normalize_photo_date(value) == value.strip()


@pytest.mark.parametrize("value", ["74", "1974-13", "1974-02-30", "June 1974", "1974/06", ""])
def test_malformed_photo_dates_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_photo_date(value)


@pytest.mark.parametrize("with_exif", [False, True])
def test_jpeg_injection_is_readable_by_pillow(with_exif: bool) -> None:
    packet = build_xmp_packet(fields(digital_source_type=DIGITAL_SOURCE_ENHANCED))

    result = insert_xmp_into_jpeg(jpeg_bytes(exif=with_exif), packet)
    image = Image.open(io.BytesIO(result))
    image.load()

    assert read_xmp_properties(extract_xmp(image))["Iptc4xmpExt:DigitalSourceType"] == (
        DIGITAL_SOURCE_ENHANCED
    )
    assert image.size == (8, 8)


def test_jpeg_injection_replaces_a_previous_packet() -> None:
    first = insert_xmp_into_jpeg(jpeg_bytes(), build_xmp_packet(fields()))

    result = insert_xmp_into_jpeg(first, build_xmp_packet(fields(creator_tool="Upflow 9")))

    assert result.count(b"http://ns.adobe.com/xap/1.0/\x00") == 1
    assert b"Upflow 9" in result


def test_jpeg_injection_keeps_exif_before_xmp() -> None:
    result = insert_xmp_into_jpeg(jpeg_bytes(exif=True), build_xmp_packet(fields()))

    assert result.index(b"Exif\x00\x00") < result.index(b"http://ns.adobe.com/xap/1.0/")


def test_jpeg_injection_rejects_non_jpeg() -> None:
    with pytest.raises(ValueError):
        insert_xmp_into_jpeg(b"\x89PNG", build_xmp_packet(fields()))


def test_png_injection_is_readable_by_pillow() -> None:
    ok, encoded = cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint16))
    assert ok

    result = insert_xmp_into_png(encoded.tobytes(), build_xmp_packet(fields()))
    image = Image.open(io.BytesIO(result))
    image.load()

    assert [chunk.type for chunk in parse_chunks(result)][1] == b"iTXt"
    assert read_xmp_properties(extract_xmp(image))["Iptc4xmpExt:DigitalSourceType"] == (
        DIGITAL_SOURCE_COMPOSITE
    )


def test_webp_packet_saved_by_pillow_is_readable() -> None:
    buffer = io.BytesIO()
    packet = build_xmp_packet(fields())
    Image.new("RGB", (8, 8)).save(buffer, "WEBP", xmp=packet.encode("utf-8"))

    image = Image.open(io.BytesIO(buffer.getvalue()))

    assert read_xmp_properties(extract_xmp(image))["xmp:CreatorTool"] == "Upflow 0.80.0"


def test_extract_xmp_returns_none_without_packet() -> None:
    assert extract_xmp(Image.new("RGB", (2, 2))) is None
