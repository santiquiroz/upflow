"""Identificacion del contenedor de un video por sus primeros bytes (spec §4.2 paso 3).

Ignora la extension: los DVR Hikvision/HiLook exportan MPEG-PS con cabecera
`IMKH` y la guardan como `.mp4`. Lo usa la ingesta CCTV y la sugerencia del
modo CCTV en Enhance -> Video.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ContainerKind = Literal[
    "hikvision_ps",
    "dahua_dav",
    "mpeg_ps",
    "mp4",
    "avi",
    "matroska",
    "raw_h264",
    "raw_hevc",
    "unknown",
]

SNIFF_SIZE = 64

_START_CODE_4 = b"\x00\x00\x00\x01"
_START_CODE_3 = b"\x00\x00\x01"
_H264_NAL_TYPES = frozenset({1, 5, 6, 7, 8, 9})
_HEVC_NAL_TYPES = frozenset({32, 33, 34, 35, 39, 40})
_HEVC_LAYER0_TID1 = 0x01
_SUGGESTS_CCTV = frozenset({"hikvision_ps", "dahua_dav", "raw_h264", "raw_hevc"})


@dataclass(frozen=True)
class ContainerGuess:
    kind: ContainerKind
    label: str
    ffmpeg_format: str | None

    @property
    def suggests_cctv(self) -> bool:
        return self.kind in _SUGGESTS_CCTV


HIKVISION_PS = ContainerGuess("hikvision_ps", "Hikvision PS", "mpeg")
DAHUA_DAV = ContainerGuess("dahua_dav", "Dahua DAV", "dhav")
MPEG_PS = ContainerGuess("mpeg_ps", "MPEG-PS", "mpeg")
MP4 = ContainerGuess("mp4", "MP4/MOV", None)
AVI = ContainerGuess("avi", "AVI", None)
MATROSKA = ContainerGuess("matroska", "Matroska", None)
RAW_H264 = ContainerGuess("raw_h264", "Raw H.264", "h264")
RAW_HEVC = ContainerGuess("raw_hevc", "Raw H.265", "hevc")
UNKNOWN = ContainerGuess("unknown", "Unknown", None)

CONTAINER_GUESSES: tuple[ContainerGuess, ...] = (
    HIKVISION_PS,
    DAHUA_DAV,
    MPEG_PS,
    MP4,
    AVI,
    MATROSKA,
    RAW_H264,
    RAW_HEVC,
    UNKNOWN,
)


def is_hikvision_ps(data: bytes) -> bool:
    return data.startswith(b"IMKH")


def is_dahua_dav(data: bytes) -> bool:
    return data.startswith(b"DHAV")


def is_mpeg_ps(data: bytes) -> bool:
    return data.startswith(b"\x00\x00\x01\xba")


def is_mp4(data: bytes) -> bool:
    return data[4:8] == b"ftyp"


def is_avi(data: bytes) -> bool:
    return data.startswith(b"RIFF") and data[8:12] == b"AVI "


def is_matroska(data: bytes) -> bool:
    return data.startswith(b"\x1a\x45\xdf\xa3")


def first_nal_header(data: bytes) -> bytes:
    if data.startswith(_START_CODE_4):
        return data[4:6]
    if data.startswith(_START_CODE_3):
        return data[3:5]
    return b""


def is_hevc_nal(header: bytes) -> bool:
    if len(header) < 2 or header[0] & 0x80:
        return False
    return ((header[0] >> 1) & 0x3F) in _HEVC_NAL_TYPES and header[1] == _HEVC_LAYER0_TID1


def is_h264_nal(header: bytes) -> bool:
    if not header or header[0] & 0x80:
        return False
    return (header[0] & 0x1F) in _H264_NAL_TYPES


def is_raw_hevc(data: bytes) -> bool:
    return is_hevc_nal(first_nal_header(data))


def is_raw_h264(data: bytes) -> bool:
    return is_h264_nal(first_nal_header(data))


# Orden importante: MPEG-PS empieza con un start code (00 00 01 BA) y HEVC antes
# que H.264 porque algunos encabezados HEVC tambien son NAL H.264 validos.
_RULES: tuple[tuple[Callable[[bytes], bool], ContainerGuess], ...] = (
    (is_hikvision_ps, HIKVISION_PS),
    (is_dahua_dav, DAHUA_DAV),
    (is_mpeg_ps, MPEG_PS),
    (is_mp4, MP4),
    (is_avi, AVI),
    (is_matroska, MATROSKA),
    (is_raw_hevc, RAW_HEVC),
    (is_raw_h264, RAW_H264),
)


def sniff_container(first_bytes: bytes) -> ContainerGuess:
    return next((guess for matches, guess in _RULES if matches(first_bytes)), UNKNOWN)


def sniff_file(path: Path) -> ContainerGuess:
    with path.open("rb") as handle:
        return sniff_container(handle.read(SNIFF_SIZE))


def container_guess(kind: str) -> ContainerGuess:
    for guess in CONTAINER_GUESSES:
        if guess.kind == kind:
            return guess
    raise ValueError(f"Unknown container kind: {kind!r}")
