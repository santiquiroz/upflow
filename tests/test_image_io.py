from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps
from PIL.ExifTags import IFD

from app.services.image_io import (
    apply_exif_orientation,
    load_image_for_restore,
    prepare_output_exif,
    save_restored,
)
from app.services.png_chunks import parse_chunks
from app.services.xmp_packet import (
    DIGITAL_SOURCE_COMPOSITE,
    XmpFields,
    build_xmp_packet,
    extract_xmp,
    read_xmp_properties,
)

ORIENTATION = 0x0112
MAKE = 0x010F
DATE_TIME_ORIGINAL = 0x9003
DATE_TIME_DIGITIZED = 0x9004
OFFSET_TIME_ORIGINAL = 0x9011
OFFSET_TIME_DIGITIZED = 0x9012
GPS_LATITUDE_REF = 0x0001
REPRODUCTION_DATE = "2026:09:20 18:30:00"


def srgb_icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def phone_exif(orientation: int = 1, gps: bool = True) -> Image.Exif:
    exif = Image.Exif()
    exif[ORIENTATION] = orientation
    exif[MAKE] = "PhoneMaker"
    exif[IFD.Exif] = {DATE_TIME_ORIGINAL: REPRODUCTION_DATE, OFFSET_TIME_ORIGINAL: "-05:00"}
    if gps:
        exif[IFD.GPSInfo] = {GPS_LATITUDE_REF: "N", 2: (4.0, 36.0, 0.0)}
    return exif


def asymmetric_rgb(height: int = 4, width: int = 6) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3) * 3
    return values


def xmp(date_created: str | None = None) -> str:
    return build_xmp_packet(
        XmpFields(
            digital_source_type=DIGITAL_SOURCE_COMPOSITE,
            description="Restored",
            creator_tool="Upflow test",
            date_created=date_created,
        )
    )


def reopen(path: Path) -> Image.Image:
    image = Image.open(path)
    image.load()
    return image


# --- carga ------------------------------------------------------------------


def test_jpeg_with_orientation_6_is_rotated_upright(tmp_path: Path) -> None:
    path = tmp_path / "phone.jpg"
    Image.new("RGB", (60, 20), (200, 10, 10)).save(path, exif=phone_exif(orientation=6))

    loaded = load_image_for_restore(path)

    assert loaded.rgb.shape == (60, 20, 3)
    assert loaded.orientation_applied == 6
    assert loaded.bit_depth == 8
    assert loaded.rgb.dtype == np.float32


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orientation_matches_pillow_exif_transpose(orientation: int) -> None:
    pixels = asymmetric_rgb()
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[ORIENTATION] = orientation
    buffer = io.BytesIO()
    image.save(buffer, "PNG", exif=exif)
    expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(buffer.getvalue()))))

    np.testing.assert_array_equal(apply_exif_orientation(pixels, orientation), expected)


def test_invalid_orientation_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "odd.jpg"
    Image.new("RGB", (8, 4)).save(path, exif=phone_exif(orientation=42))

    loaded = load_image_for_restore(path)

    assert loaded.orientation_applied == 1
    assert loaded.rgb.shape == (4, 8, 3)


def test_icc_and_exif_are_read_from_a_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "with-icc.jpg"
    Image.new("RGB", (8, 8)).save(path, icc_profile=srgb_icc(), exif=phone_exif())

    loaded = load_image_for_restore(path)

    assert loaded.icc == srgb_icc()
    exif = Image.Exif()
    exif.load(loaded.exif)
    assert exif[MAKE] == "PhoneMaker"


def test_sixteen_bit_tiff_is_not_saturated(tmp_path: Path) -> None:
    path = tmp_path / "scan.tif"
    pixels = np.zeros((5, 7, 3), dtype=np.uint16)
    pixels[..., 0] = 40000
    pixels[..., 2] = 1000
    assert cv2.imwrite(str(path), pixels)

    loaded = load_image_for_restore(path)

    assert loaded.bit_depth == 16
    np.testing.assert_allclose(loaded.rgb[0, 0], [1000 / 65535, 0.0, 40000 / 65535], rtol=1e-6)


def test_sixteen_bit_grayscale_png_becomes_three_channels(tmp_path: Path) -> None:
    path = tmp_path / "gray16.png"
    assert cv2.imwrite(str(path), np.full((3, 4), 30000, dtype=np.uint16))

    loaded = load_image_for_restore(path)

    assert loaded.bit_depth == 16
    assert loaded.rgb.shape == (3, 4, 3)
    np.testing.assert_allclose(loaded.rgb, 30000 / 65535, rtol=1e-6)


def test_tiff_icc_is_read_with_pillow(tmp_path: Path) -> None:
    path = tmp_path / "scan8.tif"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, icc_profile=srgb_icc())

    loaded = load_image_for_restore(path)

    assert loaded.icc == srgb_icc()
    assert loaded.bit_depth == 8
    np.testing.assert_allclose(loaded.rgb[0, 0], np.array([1, 2, 3]) / 255, rtol=1e-6)


@pytest.mark.parametrize("name,depth", [("alpha.png", 8), ("alpha16.png", 16)])
def test_alpha_is_dropped_with_a_warning(tmp_path: Path, name: str, depth: int) -> None:
    path = tmp_path / name
    dtype = np.uint16 if depth == 16 else np.uint8
    assert cv2.imwrite(str(path), np.full((2, 2, 4), 9, dtype=dtype))

    loaded = load_image_for_restore(path)

    assert loaded.rgb.shape == (2, 2, 3)
    assert "alpha_dropped" in loaded.warnings


def test_cmyk_icc_is_dropped_because_the_pixels_become_rgb(tmp_path: Path) -> None:
    path = tmp_path / "print.jpg"
    Image.new("CMYK", (4, 4), (0, 0, 0, 0)).save(path, icc_profile=b"cmyk-profile")

    loaded = load_image_for_restore(path)

    assert loaded.icc is None
    assert "cmyk_icc_dropped" in loaded.warnings
    assert loaded.rgb.shape == (4, 4, 3)


def test_opaque_image_has_no_warnings(tmp_path: Path) -> None:
    path = tmp_path / "plain.webp"
    Image.new("RGB", (4, 4)).save(path)

    assert load_image_for_restore(path).warnings == ()


def test_unicode_paths_are_supported(tmp_path: Path) -> None:
    path = tmp_path / "foto de la abuela ñ.tif"
    ok, encoded = cv2.imencode(".tif", np.zeros((2, 2, 3), dtype=np.uint16))
    assert ok
    encoded.tofile(str(path))

    loaded = load_image_for_restore(path)
    save_restored(loaded.rgb, tmp_path / "salida ñ.png", "png", 16, None, None, xmp())

    assert loaded.bit_depth == 16
    assert (tmp_path / "salida ñ.png").exists()


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "anim.gif"
    Image.new("RGB", (2, 2)).save(path)

    with pytest.raises(ValueError):
        load_image_for_restore(path)


# --- política de EXIF --------------------------------------------------------


def test_gps_is_removed_by_default() -> None:
    result = prepare_output_exif(phone_exif().tobytes())

    exif = Image.Exif()
    exif.load(result.exif)
    assert exif.get_ifd(IFD.GPSInfo) == {}
    assert result.privacy.gps_removed is True


def test_gps_is_kept_on_request() -> None:
    result = prepare_output_exif(phone_exif().tobytes(), keep_gps=True)

    exif = Image.Exif()
    exif.load(result.exif)
    assert exif.get_ifd(IFD.GPSInfo)[GPS_LATITUDE_REF] == "N"
    assert result.privacy.gps_removed is False


def test_date_time_original_moves_to_digitized() -> None:
    result = prepare_output_exif(phone_exif().tobytes())

    exif = Image.Exif()
    exif.load(result.exif)
    exif_ifd = exif.get_ifd(IFD.Exif)
    assert DATE_TIME_ORIGINAL not in exif_ifd
    assert exif_ifd[DATE_TIME_DIGITIZED] == REPRODUCTION_DATE
    assert OFFSET_TIME_ORIGINAL not in exif_ifd
    assert exif_ifd[OFFSET_TIME_DIGITIZED] == "-05:00"
    assert result.privacy.date_time_original_moved is True


def test_input_pixel_dimensions_are_not_copied() -> None:
    exif = phone_exif()
    exif[IFD.Exif] = {**exif[IFD.Exif], 0xA002: 4000, 0xA003: 3000}

    result = prepare_output_exif(exif.tobytes())

    output = Image.Exif()
    output.load(result.exif)
    assert 0xA002 not in output.get_ifd(IFD.Exif)
    assert output.get_ifd(IFD.Exif)[DATE_TIME_DIGITIZED] == REPRODUCTION_DATE


def test_orientation_is_reset_and_descriptive_tags_survive() -> None:
    result = prepare_output_exif(phone_exif(orientation=6).tobytes())

    exif = Image.Exif()
    exif.load(result.exif)
    assert exif[ORIENTATION] == 1
    assert exif[MAKE] == "PhoneMaker"


def test_tiff_structure_tags_are_not_copied(tmp_path: Path) -> None:
    path = tmp_path / "scan.tif"
    Image.new("RGB", (4, 4)).save(path, icc_profile=srgb_icc())

    result = prepare_output_exif(load_image_for_restore(path).exif)

    exif = Image.Exif()
    exif.load(result.exif)
    assert 273 not in exif  # StripOffsets
    assert 34675 not in exif  # ICC: travels on its own


def test_missing_exif_stays_missing() -> None:
    result = prepare_output_exif(None)

    assert result.exif is None
    assert result.privacy.gps_removed is False
    assert result.privacy.date_time_original_moved is False


# --- guardado ---------------------------------------------------------------


def float_image(height: int = 6, width: int = 5) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.random((height, width, 3), dtype=np.float32)


@pytest.mark.parametrize("fmt,suffix", [("jpg", "jpg"), ("webp", "webp"), ("png", "png")])
def test_icc_exif_and_xmp_survive_an_eight_bit_save(tmp_path: Path, fmt: str, suffix: str) -> None:
    path = tmp_path / f"out.{suffix}"

    privacy = save_restored(
        float_image(), path, fmt, 8, srgb_icc(), phone_exif(orientation=6).tobytes(), xmp("1974")
    )

    image = reopen(path)
    assert image.info["icc_profile"] == srgb_icc()
    assert image.getexif()[ORIENTATION] == 1
    assert image.getexif().get_ifd(IFD.GPSInfo) == {}
    props = read_xmp_properties(extract_xmp(image))
    assert props["Iptc4xmpExt:DigitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert props["photoshop:DateCreated"] == "1974"
    assert privacy.gps_removed and privacy.date_time_original_moved


def test_sixteen_bit_png_round_trips_exactly_with_readable_chunks(tmp_path: Path) -> None:
    source = tmp_path / "scan16.png"
    pixels = np.random.default_rng(5).integers(0, 65536, (6, 5, 3), dtype=np.uint16)
    assert cv2.imwrite(str(source), pixels)
    loaded = load_image_for_restore(source)
    output = tmp_path / "restored.png"

    save_restored(loaded.rgb, output, "png", 16, srgb_icc(), phone_exif().tobytes(), xmp())

    reloaded = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert reloaded.dtype == np.uint16
    np.testing.assert_array_equal(reloaded, pixels)
    types = [chunk.type for chunk in parse_chunks(output.read_bytes())]
    assert types.index(b"iCCP") < types.index(b"IDAT")
    assert b"iTXt" in types and b"eXIf" in types
    image = reopen(output)
    assert image.info["icc_profile"] == srgb_icc()
    assert read_xmp_properties(extract_xmp(image))["xmp:CreatorTool"] == "Upflow test"


def test_sixteen_bit_load_and_save_keeps_the_embedded_icc(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    save_restored(float_image(), first, "png", 16, srgb_icc(), None, xmp())

    loaded = load_image_for_restore(first)

    assert loaded.bit_depth == 16
    assert loaded.icc == srgb_icc()


def test_out_of_range_values_are_clipped(tmp_path: Path) -> None:
    path = tmp_path / "clip.png"
    rgb = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)

    save_restored(rgb, path, "png", 8, None, None, xmp())

    np.testing.assert_array_equal(np.asarray(reopen(path))[0, 0], [0, 128, 255])


def test_sixteen_bit_is_only_for_png(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_restored(float_image(), tmp_path / "x.jpg", "jpg", 16, None, None, xmp())


def test_unknown_output_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_restored(float_image(), tmp_path / "x.tif", "tiff", 8, None, None, xmp())


def test_no_exif_in_means_no_exif_out(tmp_path: Path) -> None:
    path = tmp_path / "plain.jpg"

    privacy = save_restored(float_image(), path, "jpeg", 8, None, None, xmp())

    assert len(reopen(path).getexif()) == 0
    assert privacy.gps_removed is False


def test_failed_encode_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    huge_xmp = xmp() + " " * 70_000

    with pytest.raises(ValueError):
        save_restored(float_image(), path, "jpg", 8, None, None, huge_xmp)

    assert list(tmp_path.iterdir()) == []


def test_save_does_not_modify_the_input_array(tmp_path: Path) -> None:
    rgb = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    before = rgb.copy()

    save_restored(rgb, tmp_path / "x.png", "png", 16, None, None, xmp())

    np.testing.assert_array_equal(rgb, before)
