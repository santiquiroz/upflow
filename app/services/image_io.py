from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import IFD

from app.services.png_chunks import (
    build_exif_chunk,
    build_iccp_chunk,
    insert_after_ihdr,
    png_bit_depth,
)
from app.services.xmp_packet import insert_xmp_into_jpeg, insert_xmp_into_png

ORIENTATION_TAG = 0x0112
UPRIGHT = 1
LOSSY_QUALITY = 95
PNG_COMPRESSION = 6
MAX_VALUE = {8: 255.0, 16: 65535.0}
SUPPORTED_INPUT_FORMATS = frozenset({"JPEG", "MPO", "PNG", "WEBP", "BMP", "TIFF"})
OUTPUT_FORMATS = {
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "webp": "WEBP",
    "tif": "TIFF",
    "tiff": "TIFF",
}
SIXTEEN_BIT_FORMATS = frozenset({"PNG", "TIFF"})
TIFF_XMP_TAG = 700
ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
ALPHA_DROPPED = "alpha_dropped"
CMYK_ICC_DROPPED = "cmyk_icc_dropped"
# The Interop pointer is a raw offset that would dangle once the IFD is rewritten, and
# PixelXDimension/PixelYDimension describe the input size, not the restored one.
STALE_CAPTURE_TAGS = frozenset({0xA005, 0xA002, 0xA003})
# DateTimeOriginal/OffsetTimeOriginal/SubSecTimeOriginal describe when the print was
# photographed or scanned, not when the photo was taken: they become *Digitized.
CAPTURE_TO_DIGITIZED = {0x9003: 0x9004, 0x9011: 0x9012, 0x9291: 0x9292}
DATE_TIME_ORIGINAL = 0x9003
# TIFF layout tags and embedded blobs that describe the input file, not the output.
NON_DESCRIPTIVE_TAGS = frozenset(
    {
        254, 255, 256, 257, 258, 259, 262, 273, 277, 278, 279, 284, 317, 320, 322, 323,
        324, 325, 330, 338, 339, 347, 513, 514, 530, 532, 700, 33723, 34377, 34675,
        ORIENTATION_TAG, IFD.Exif, IFD.GPSInfo, IFD.Interop,
    }
)

ORIENTATION_TRANSFORMS = {
    1: lambda a: a,
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1],
    5: lambda a: a.swapaxes(0, 1),
    6: lambda a: np.rot90(a, k=-1),
    7: lambda a: a[::-1, ::-1].swapaxes(0, 1),
    8: lambda a: np.rot90(a, k=1),
}


@dataclass(frozen=True)
class LoadedImage:
    rgb: np.ndarray
    bit_depth: int
    icc: bytes | None
    exif: bytes | None
    orientation_applied: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetadataPrivacy:
    gps_removed: bool
    date_time_original_moved: bool
    metadata_not_embedded: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputExif:
    exif: bytes | None
    privacy: MetadataPrivacy


@dataclass(frozen=True)
class _Header:
    format: str
    mode: str
    icc: bytes | None
    exif: bytes | None
    orientation: int


@dataclass(frozen=True)
class _Pixels:
    array: np.ndarray
    bit_depth: int
    warnings: tuple[str, ...]


def load_image_for_restore(path: Path | str) -> LoadedImage:
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        header = _read_header(image)
        pixels = _decode_with_cv2(data) if _needs_cv2(header, data) else _decode_with_pillow(image)
    oriented = apply_exif_orientation(pixels.array, header.orientation)
    icc, icc_warnings = _usable_icc(header)
    return LoadedImage(
        rgb=_to_unit_float(oriented, pixels.bit_depth),
        bit_depth=pixels.bit_depth,
        icc=icc,
        exif=header.exif,
        orientation_applied=header.orientation,
        warnings=pixels.warnings + icc_warnings,
    )


def apply_exif_orientation(pixels: np.ndarray, orientation: int) -> np.ndarray:
    return ORIENTATION_TRANSFORMS.get(orientation, ORIENTATION_TRANSFORMS[UPRIGHT])(pixels)


def prepare_output_exif(exif_bytes: bytes | None, keep_gps: bool = False) -> OutputExif:
    if not exif_bytes:
        nothing_changed = MetadataPrivacy(gps_removed=False, date_time_original_moved=False)
        return OutputExif(None, nothing_changed)
    source = Image.Exif()
    source.load(exif_bytes)
    capture_ifd = _without_stale_tags(source.get_ifd(IFD.Exif))
    gps_ifd = dict(source.get_ifd(IFD.GPSInfo))
    output = _output_exif(
        _descriptive_tags(source),
        _capture_dates_as_digitized(capture_ifd),
        gps_ifd if keep_gps else {},
    )
    privacy = MetadataPrivacy(
        gps_removed=bool(gps_ifd) and not keep_gps,
        date_time_original_moved=DATE_TIME_ORIGINAL in capture_ifd,
    )
    return OutputExif(output.tobytes(), privacy)


def save_restored(
    rgb: np.ndarray,
    path: Path | str,
    fmt: str,
    bit_depth: int,
    icc: bytes | None,
    exif: bytes | None,
    xmp: str,
    keep_gps: bool = False,
) -> MetadataPrivacy:
    pillow_format = _output_format(fmt, bit_depth)
    prepared = prepare_output_exif(exif, keep_gps)
    pixels = _quantize(rgb, bit_depth)
    encoders = {"PNG": _encode_png, "JPEG": _encode_jpeg, "WEBP": _encode_webp, "TIFF": _encode_tiff}
    encoded = encoders[pillow_format](pixels, icc, prepared.exif, xmp)
    _write_atomically(Path(path), encoded)
    not_embedded = _metadata_not_embedded(pillow_format, bit_depth, icc, prepared.exif, xmp)
    return replace(prepared.privacy, metadata_not_embedded=not_embedded)


def _read_header(image: Image.Image) -> _Header:
    if image.format not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported image format for restore: {image.format}")
    exif = image.getexif()
    orientation = exif.get(ORIENTATION_TAG, UPRIGHT)
    return _Header(
        format=image.format,
        mode=image.mode,
        icc=image.info.get("icc_profile") or None,
        exif=exif.tobytes() if len(exif) else None,
        orientation=orientation if orientation in ORIENTATION_TRANSFORMS else UPRIGHT,
    )


def _needs_cv2(header: _Header, data: bytes) -> bool:
    # Pillow truncates 16-bit RGB to 8 bits, so TIFF and 16-bit PNG go through cv2.
    return header.format == "TIFF" or (header.format == "PNG" and png_bit_depth(data) == 16)


def _decode_with_cv2(data: bytes) -> _Pixels:
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype not in (np.uint8, np.uint16):
        raise ValueError("Unsupported pixel layout: only 8 and 16-bit integer images")
    bit_depth = 16 if decoded.dtype == np.uint16 else 8
    rgb, warnings = _cv2_to_rgb(decoded)
    return _Pixels(rgb, bit_depth, warnings)


def _cv2_to_rgb(decoded: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    if decoded.ndim == 2:
        return np.repeat(decoded[..., None], 3, axis=2), ()
    if decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB), (ALPHA_DROPPED,)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), ()


def _decode_with_pillow(image: Image.Image) -> _Pixels:
    has_alpha = image.mode in ALPHA_MODES or "transparency" in image.info
    rgb = np.asarray(image.convert("RGB"))
    return _Pixels(rgb, 8, (ALPHA_DROPPED,) if has_alpha else ())


def _usable_icc(header: _Header) -> tuple[bytes | None, tuple[str, ...]]:
    # A CMYK profile cannot describe the RGB pixels Pillow converts to.
    if header.mode == "CMYK" and header.icc is not None:
        return None, (CMYK_ICC_DROPPED,)
    return header.icc, ()


def _to_unit_float(pixels: np.ndarray, bit_depth: int) -> np.ndarray:
    unit = pixels.astype(np.float32, order="C")
    unit /= np.float32(MAX_VALUE[bit_depth])
    return unit


def _descriptive_tags(source: Image.Exif) -> dict[int, object]:
    return {tag: value for tag, value in source.items() if tag not in NON_DESCRIPTIVE_TAGS}


def _without_stale_tags(capture_ifd: dict[int, object]) -> dict[int, object]:
    return {tag: value for tag, value in capture_ifd.items() if tag not in STALE_CAPTURE_TAGS}


def _capture_dates_as_digitized(capture_ifd: dict[int, object]) -> dict[int, object]:
    kept = {tag: value for tag, value in capture_ifd.items() if tag not in CAPTURE_TO_DIGITIZED}
    moved = {
        digitized: capture_ifd[original]
        for original, digitized in CAPTURE_TO_DIGITIZED.items()
        if original in capture_ifd
    }
    return {**kept, **moved}


def _output_exif(
    base: dict[int, object], capture_ifd: dict[int, object], gps_ifd: dict[int, object]
) -> Image.Exif:
    output = Image.Exif()
    for tag, value in {**base, ORIENTATION_TAG: UPRIGHT}.items():
        output[tag] = value
    if capture_ifd:
        output[IFD.Exif] = capture_ifd
    if gps_ifd:
        output[IFD.GPSInfo] = gps_ifd
    return output


def _output_format(fmt: str, bit_depth: int) -> str:
    pillow_format = OUTPUT_FORMATS.get(fmt.lower())
    if pillow_format is None:
        raise ValueError(f"Restore output format must be png, jpg, jpeg, webp, tif or tiff: {fmt!r}")
    if bit_depth not in MAX_VALUE or (bit_depth == 16 and pillow_format not in SIXTEEN_BIT_FORMATS):
        raise ValueError(f"{bit_depth}-bit output is not available for {fmt}")
    return pillow_format


def _quantize(rgb: np.ndarray, bit_depth: int) -> np.ndarray:
    scaled = np.clip(rgb, 0.0, 1.0) * np.float32(MAX_VALUE[bit_depth])
    np.rint(scaled, out=scaled)
    return scaled.astype(np.uint16 if bit_depth == 16 else np.uint8)


def _encode_png(pixels: np.ndarray, icc: bytes | None, exif: bytes | None, xmp: str) -> bytes:
    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise ValueError("PNG encoding failed")
    metadata = [build_iccp_chunk(icc)] if icc else []
    metadata += [build_exif_chunk(exif)] if exif else []
    return insert_xmp_into_png(insert_after_ihdr(encoded.tobytes(), metadata), xmp)


def _encode_jpeg(pixels: np.ndarray, icc: bytes | None, exif: bytes | None, xmp: str) -> bytes:
    # Pillow 10.4 has no xmp= for JPEG, so the APP1 packet is inserted by hand.
    return insert_xmp_into_jpeg(_encode_with_pillow(pixels, "JPEG", icc, exif), xmp)


def _encode_webp(pixels: np.ndarray, icc: bytes | None, exif: bytes | None, xmp: str) -> bytes:
    return _encode_with_pillow(pixels, "WEBP", icc, exif, xmp=xmp.encode("utf-8"))


def _encode_tiff(pixels: np.ndarray, icc: bytes | None, exif: bytes | None, xmp: str) -> bytes:
    if pixels.dtype == np.uint16:
        return _encode_tiff16(pixels)
    buffer = io.BytesIO()
    options = {"icc_profile": icc} if icc else {}
    # Uncompressed on purpose: Pillow's libtiff path (any compression) cannot write the EXIF sub-IFD.
    Image.fromarray(pixels, "RGB").save(buffer, "TIFF", tiffinfo=_tiff_tags(exif, xmp), **options)
    return buffer.getvalue()


def _tiff_tags(exif: bytes | None, xmp: str) -> Image.Exif:
    tags = Image.Exif()
    if exif:
        tags.load(exif)
    tags[TIFF_XMP_TAG] = xmp.encode("utf-8")
    return tags


def _encode_tiff16(pixels: np.ndarray) -> bytes:
    # Pillow cannot write 16-bit RGB, and cv2 writes no ICC, EXIF or XMP: save_restored declares it.
    ok, encoded = cv2.imencode(".tiff", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("TIFF encoding failed")
    return encoded.tobytes()


def _metadata_not_embedded(
    pillow_format: str, bit_depth: int, icc: bytes | None, exif: bytes | None, xmp: str
) -> tuple[str, ...]:
    if pillow_format != "TIFF" or bit_depth != 16:
        return ()
    present = {"icc": bool(icc), "exif": bool(exif), "xmp": bool(xmp)}
    return tuple(name for name, is_present in present.items() if is_present)


def _encode_with_pillow(
    pixels: np.ndarray, pillow_format: str, icc: bytes | None, exif: bytes | None, **extra: object
) -> bytes:
    options = {"quality": LOSSY_QUALITY, "icc_profile": icc, "exif": exif, **extra}
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(
        buffer, pillow_format, **{key: value for key, value in options.items() if value is not None}
    )
    return buffer.getvalue()


def _write_atomically(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".part"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
