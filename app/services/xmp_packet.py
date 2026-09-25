from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from xml.sax.saxutils import escape

from PIL import Image

from app.services.png_chunks import build_itxt_chunk, insert_after_ihdr

DIGITAL_SOURCE_BASE = "http://cv.iptc.org/newscodes/digitalsourcetype/"
DIGITAL_SOURCE_COMPOSITE = DIGITAL_SOURCE_BASE + "compositeWithTrainedAlgorithmicMedia"
DIGITAL_SOURCE_ENHANCED = DIGITAL_SOURCE_BASE + "algorithmicallyEnhanced"
DIGITAL_SOURCE_TYPES = frozenset({DIGITAL_SOURCE_COMPOSITE, DIGITAL_SOURCE_ENHANCED})

PNG_XMP_KEYWORD = "XML:com.adobe.xmp"
JPEG_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
JPEG_SOI = b"\xff\xd8"
JPEG_APP0 = 0xE0
JPEG_APP1 = 0xE1
JPEG_APP15 = 0xEF
JPEG_MAX_SEGMENT_PAYLOAD = 0xFFFF - 2

NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "Iptc4xmpExt": "http://iptc.org/std/Iptc4xmpExt/2008-02-29/",
}
SIMPLE_PROPERTIES = ("Iptc4xmpExt:DigitalSourceType", "xmp:CreatorTool", "photoshop:DateCreated")
PARTIAL_DATE_PATTERN = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


@dataclass(frozen=True)
class XmpFields:
    digital_source_type: str
    description: str
    creator_tool: str
    date_created: str | None = None


def normalize_photo_date(value: str) -> str:
    candidate = value.strip()
    match = PARTIAL_DATE_PATTERN.match(candidate)
    if match is None:
        raise ValueError(f"Photo date must be YYYY, YYYY-MM or YYYY-MM-DD: {value!r}")
    year, month, day = (int(part) if part else 1 for part in match.groups())
    date(year, month, day)
    return candidate


def build_xmp_packet(fields: XmpFields) -> str:
    if fields.digital_source_type not in DIGITAL_SOURCE_TYPES:
        raise ValueError(f"Unknown DigitalSourceType: {fields.digital_source_type!r}")
    properties = "".join(_property_elements(fields))
    namespaces = " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in NAMESPACES.items())
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        f'<x:xmpmeta {namespaces}>'
        '<rdf:RDF><rdf:Description rdf:about="">'
        f"{properties}"
        "</rdf:Description></rdf:RDF></x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )


def read_xmp_properties(packet: str) -> dict[str, str]:
    description = _rdf_description(packet)
    values = {name: _simple_value(description, name) for name in SIMPLE_PROPERTIES}
    values["dc:description"] = _lang_alt_value(description, "dc:description")
    return {name: value for name, value in values.items() if value is not None}


def extract_xmp(image: Image.Image) -> str | None:
    raw = image.info.get("xmp") or image.info.get(PNG_XMP_KEYWORD)
    if raw is None:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


def insert_xmp_into_png(png: bytes, packet: str) -> bytes:
    return insert_after_ihdr(png, [build_itxt_chunk(PNG_XMP_KEYWORD, packet)])


def insert_xmp_into_jpeg(jpeg: bytes, packet: str) -> bytes:
    if not jpeg.startswith(JPEG_SOI):
        raise ValueError("Not a JPEG file")
    segments, body_offset = _leading_app_segments(jpeg)
    kept = [segment for segment in segments if not _is_xmp_segment(segment)]
    split = _after_last_app0_or_app1(kept)
    ordered = [*kept[:split], _xmp_segment(packet), *kept[split:]]
    return JPEG_SOI + b"".join(ordered) + jpeg[body_offset:]


def _property_elements(fields: XmpFields) -> list[str]:
    elements = [
        _simple_element("Iptc4xmpExt:DigitalSourceType", fields.digital_source_type),
        _lang_alt_element("dc:description", fields.description),
        _simple_element("xmp:CreatorTool", fields.creator_tool),
    ]
    if fields.date_created is not None:
        elements.append(
            _simple_element("photoshop:DateCreated", normalize_photo_date(fields.date_created))
        )
    return elements


def _simple_element(name: str, value: str) -> str:
    return f"<{name}>{escape(value)}</{name}>"


def _lang_alt_element(name: str, value: str) -> str:
    item = f'<rdf:li xml:lang="x-default">{escape(value)}</rdf:li>'
    return f"<{name}><rdf:Alt>{item}</rdf:Alt></{name}>"


def _rdf_description(packet: str) -> ET.Element:
    # XMP never carries a DTD; refusing one closes entity expansion without defusedxml.
    if "<!DOCTYPE" in packet or "<!ENTITY" in packet:
        raise ValueError("XMP packet must not declare a DTD")
    root = ET.fromstring(_strip_xpacket(packet))
    description = root.find("rdf:RDF/rdf:Description", NAMESPACES)
    if description is None:
        raise ValueError("XMP packet has no rdf:Description")
    return description


def _strip_xpacket(packet: str) -> str:
    return re.sub(r"<\?xpacket[^>]*\?>", "", packet).strip()


def _simple_value(description: ET.Element, name: str) -> str | None:
    element = description.find(name, NAMESPACES)
    return None if element is None else (element.text or "")


def _lang_alt_value(description: ET.Element, name: str) -> str | None:
    element = description.find(f"{name}/rdf:Alt/rdf:li", NAMESPACES)
    return None if element is None else (element.text or "")


def _leading_app_segments(jpeg: bytes) -> tuple[list[bytes], int]:
    segments: list[bytes] = []
    offset = len(JPEG_SOI)
    while _is_app_marker(jpeg, offset):
        (length,) = struct.unpack(">H", jpeg[offset + 2 : offset + 4])
        segments.append(jpeg[offset : offset + 2 + length])
        offset += 2 + length
    return segments, offset


def _is_app_marker(jpeg: bytes, offset: int) -> bool:
    return (
        offset + 4 <= len(jpeg)
        and jpeg[offset] == 0xFF
        and JPEG_APP0 <= jpeg[offset + 1] <= JPEG_APP15
    )


def _is_xmp_segment(segment: bytes) -> bool:
    return segment[1] == JPEG_APP1 and segment[4:].startswith(JPEG_XMP_HEADER)


def _after_last_app0_or_app1(segments: list[bytes]) -> int:
    positions = [
        index + 1 for index, segment in enumerate(segments) if segment[1] in (JPEG_APP0, JPEG_APP1)
    ]
    return max(positions, default=0)


def _xmp_segment(packet: str) -> bytes:
    payload = JPEG_XMP_HEADER + packet.encode("utf-8")
    if len(payload) > JPEG_MAX_SEGMENT_PAYLOAD:
        raise ValueError("XMP packet too large for a single JPEG APP1 segment")
    return bytes((0xFF, JPEG_APP1)) + struct.pack(">H", len(payload) + 2) + payload
