from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services import label_band as lb
from app.services.frame_export import StillRequest, StillSource, export_still_pairs
from app.services.cctv_clarify_runner import ClarifyTools
from app.services.xmp_packet import (
    DIGITAL_SOURCE_COMPOSITE,
    DIGITAL_SOURCE_ENHANCED,
    extract_xmp,
    read_xmp_properties,
)
from ffmpeg_support import needs_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.9.9"
JOB_ID = "3f2a9c1e-7b44-4d0e-9a51-0c8f2e6d1b77"


# --- Fuente OFL bundleada ---


def test_the_bundled_font_is_the_pinned_ofl_source_code_pro() -> None:
    assert lb.FONT_PATH == ROOT / "app" / "assets" / "fonts" / "SourceCodePro-Regular.ttf"
    assert hashlib.sha256(lb.FONT_PATH.read_bytes()).hexdigest() == lb.FONT_SHA256
    assert lb.verified_font() == lb.FONT_PATH


def test_the_ofl_license_travels_next_to_the_font_and_in_the_notices() -> None:
    license_text = lb.FONT_LICENSE_PATH.read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert lb.FONT_LICENSE_PATH.parent == lb.FONT_PATH.parent
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_text
    assert "Reserved Font Name 'Source'" in license_text
    assert "Component: `app/assets/fonts/SourceCodePro-Regular.ttf`" in notices
    assert "License: OFL-1.1" in notices and lb.FONT_SHA256 in notices


def test_a_font_with_another_sha256_is_refused(tmp_path: Path) -> None:
    tampered = tmp_path / "SourceCodePro-Regular.ttf"
    tampered.write_bytes(lb.FONT_PATH.read_bytes() + b"\0")

    with pytest.raises(lb.FontIntegrityError) as error:
        lb.verified_font(tampered)

    assert error.value.key == "cctv.error.fontUnverified"


def test_a_missing_font_is_refused(tmp_path: Path) -> None:
    with pytest.raises(lb.FontIntegrityError, match="missing"):
        lb.verified_font(tmp_path / "nope.ttf")


def test_writing_the_label_never_falls_back_to_an_unverified_font(tmp_path: Path) -> None:
    other = tmp_path / "other.ttf"
    other.write_bytes(b"not a font")

    with pytest.raises(lb.FontIntegrityError):
        lb.write_label_assets(tmp_path, 352, 288, VERSION, JOB_ID, font_path=other)


# --- Textos ---


def test_the_band_text_is_always_bilingual_with_version_and_job() -> None:
    text = lb.band_text(VERSION, JOB_ID)

    assert text == (
        "AI-ENHANCED VISUALIZATION — NOT ORIGINAL FOOTAGE / "
        "VISUALIZACIÓN CON IA — NO ES LA GRABACIÓN ORIGINAL — "
        f"Upflow {VERSION} — {JOB_ID}"
    )
    assert lb.band_lines(VERSION, JOB_ID) == (lb.BAND_EN, lb.BAND_ES, f"Upflow {VERSION} — {JOB_ID}")


def test_the_in_image_mark_names_ai_and_the_short_job_id() -> None:
    assert lb.mark_text(JOB_ID) == "AI · 3f2a9c1e"


def test_the_metadata_comment_says_it_is_not_original_footage() -> None:
    assert lb.metadata_comment(VERSION, JOB_ID) == (
        f"AI-enhanced visualization by Upflow {VERSION} (job {JOB_ID}); not original footage"
    )


# --- PNG de la banda y de la marca ---


def test_the_band_png_spans_the_frame_width_with_an_even_height(tmp_path: Path) -> None:
    assets = lb.write_label_assets(tmp_path, 352, 288, VERSION, JOB_ID)
    band = Image.open(assets.band)

    assert band.size == (352, assets.band_height)
    assert assets.band_height % 2 == 0 and assets.band_height > 0
    assert band.mode == "RGB"
    assert np.asarray(band).max() > 200


def test_the_band_lines_fit_the_frame_width() -> None:
    block = lb.fitted_block(lb.FONT_PATH, lb.band_lines(VERSION, JOB_ID), 352, lb.band_font_cap(288))

    assert block.lines == lb.band_lines(VERSION, JOB_ID)
    font = lb.load_font(lb.FONT_PATH, block.size)
    assert lb.widest_line(font, block.lines) <= 352 * lb.TEXT_FILL_RATIO


def test_a_narrow_frame_splits_the_band_at_the_dashes() -> None:
    block = lb.fitted_block(lb.FONT_PATH, lb.band_lines(VERSION, JOB_ID), 200, 20)

    assert block.lines[:4] == (
        "AI-ENHANCED VISUALIZATION",
        "NOT ORIGINAL FOOTAGE",
        "VISUALIZACIÓN CON IA",
        "NO ES LA GRABACIÓN ORIGINAL",
    )


def test_the_font_grows_with_the_frame_height_up_to_its_cap() -> None:
    small = lb.fitted_block(lb.FONT_PATH, lb.band_lines(VERSION, JOB_ID), 1920, lb.band_font_cap(288))
    large = lb.fitted_block(lb.FONT_PATH, lb.band_lines(VERSION, JOB_ID), 1920, lb.band_font_cap(1080))

    assert small.size == 12 and large.size == 30


def test_the_mark_is_small_translucent_and_even(tmp_path: Path) -> None:
    assets = lb.write_label_assets(tmp_path, 1920, 1080, VERSION, JOB_ID)
    mark = Image.open(assets.mark)

    assert mark.mode == "RGBA"
    assert mark.width % 2 == 0 and mark.height % 2 == 0
    assert mark.width < 1920 // 4 and mark.height < 1080 // 10
    assert np.asarray(mark)[0, 0, 3] == lb.MARK_BACKGROUND[3]


def test_rendering_the_label_twice_gives_identical_files(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()

    a = lb.write_label_assets(first, 704, 576, VERSION, JOB_ID)
    b = lb.write_label_assets(second, 704, 576, VERSION, JOB_ID)

    assert a.band.read_bytes() == b.band.read_bytes()
    assert a.mark.read_bytes() == b.mark.read_bytes()


# --- Filtros ---


def assets_at(tmp_path: Path) -> lb.LabelAssets:
    return lb.LabelAssets(tmp_path / "band.png", tmp_path / "mark.png", band_height=48, margin=4)


def test_the_graph_puts_the_mark_inside_the_image_before_padding_the_band_below(tmp_path: Path) -> None:
    graph = lb.label_band_graph(assets_at(tmp_path))

    mark = graph.index("overlay=x=4:y=main_h-overlay_h-4")
    pad = graph.index("pad=w=iw:h=ih+48:x=0:y=0:color=black")
    band = graph.index("overlay=x=0:y=main_h-overlay_h:eval=init")
    assert mark < pad < band
    assert graph.startswith("null[lb_image];")


def test_the_graph_reads_both_pngs_through_escaped_movie_sources(tmp_path: Path) -> None:
    graph = lb.label_band_graph(lb.LabelAssets(Path("C:/cases/a, [b]/band.png"), Path("C:/m.png"), 48, 4))

    assert "movie=C\\\\:/cases/a\\, \\[b\\]/band.png[lb_band]" in graph
    assert "movie=C\\\\:/m.png[lb_mark]" in graph


def test_the_graph_keeps_a_prefix_chain_in_front(tmp_path: Path) -> None:
    graph = lb.label_band_graph(assets_at(tmp_path), prefix=("select=eq(n\\,7)",))

    assert graph.startswith("select=eq(n\\,7),null[lb_image];")


def test_label_band_args_carry_one_vf_and_the_comment(tmp_path: Path) -> None:
    args = lb.label_band_args(assets_at(tmp_path), VERSION, JOB_ID)

    assert args[0] == "-vf" and args[2:] == ["-metadata", f"comment={lb.metadata_comment(VERSION, JOB_ID)}"]


# --- XMP ---


@pytest.mark.parametrize(
    ("generative", "expected"), [(True, DIGITAL_SOURCE_COMPOSITE), (False, DIGITAL_SOURCE_ENHANCED)]
)
def test_the_digital_source_type_depends_on_a_generative_model(generative: bool, expected: str) -> None:
    properties = read_xmp_properties(lb.still_xmp_packet(VERSION, JOB_ID, generative))

    assert properties["Iptc4xmpExt:DigitalSourceType"] == expected
    assert properties["dc:description"] == lb.band_text(VERSION, JOB_ID)
    assert properties["xmp:CreatorTool"] == f"Upflow {VERSION}"


# --- Con ffmpeg real ---


def ffmpeg_path() -> Path:
    return Settings().ffmpeg_binary_path


def make_clip(path: Path, frames: int = 25) -> Path:
    command = [
        str(ffmpeg_path()), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=352x288:rate=25", "-frames:v", str(frames),
        "-c:v", "libx264", "-threads", "1", "-x264-params", "threads=1", "-g", "25", "-bf", "2", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path


def decode_rgb(path: Path, width: int, height: int) -> np.ndarray:
    command = [str(ffmpeg_path()), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def odd_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "case 7, [cam 2]; o'neil"
    directory.mkdir()
    return directory


@needs_ffmpeg
def test_real_encode_burns_the_band_below_and_the_mark_inside_every_frame(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "clip.mkv")
    assets = lb.write_label_assets(odd_dir(tmp_path), 352, 288, VERSION, JOB_ID)
    out = tmp_path / "labeled.mkv"
    command = [
        str(ffmpeg_path()), "-hide_banner", "-v", "error", "-y", "-i", str(clip),
        *lb.label_band_args(assets, VERSION, JOB_ID), "-c:v", "ffv1", str(out),
    ]  # fmt: skip

    subprocess.run(command, check=True, capture_output=True)

    height = 288 + assets.band_height
    labeled = decode_rgb(out, 352, height)
    plain = decode_rgb(clip, 352, 288)
    assert labeled.shape == (25, height, 352, 3)
    band = np.asarray(Image.open(assets.band), dtype=np.int16)
    assert np.abs(labeled[:, 288:].astype(np.int16) - band).max() <= 24
    mark_w, mark_h = Image.open(assets.mark).size
    top = 288 - assets.margin - mark_h
    corner = (slice(top, 288 - assets.margin), slice(assets.margin, assets.margin + mark_w))
    assert all(np.abs(labeled[i][corner].astype(int) - plain[i][corner]).mean() > 20 for i in range(25))
    assert np.array_equal(labeled[:, : top - 2], plain[:, : top - 2])
    probe = subprocess.run(
        [str(Settings().ffprobe_binary_path), "-v", "error", "-show_entries", "format_tags=comment", str(out)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    assert "not original footage" in probe


@needs_ffmpeg
async def test_real_ai_still_png_carries_the_band_and_the_xmp(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "clip.mkv")
    out_dir = odd_dir(tmp_path)
    assets = lb.write_label_assets(out_dir, 352, 288, VERSION, JOB_ID)
    times = tuple(n / 25 for n in range(25))
    request = StillRequest(
        original=StillSource(clip, times),
        processed=StillSource(clip, times),
        frames=(12,),
        output_dir=out_dir,
        label=assets,
        xmp_packet=lb.still_xmp_packet(VERSION, JOB_ID, generative=True),
    )
    settings = Settings()

    (pair,) = await export_still_pairs(ClarifyTools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path), request)

    processed = Image.open(pair.processed.path)
    assert processed.size == (352, 288 + assets.band_height)
    properties = read_xmp_properties(extract_xmp(processed))
    assert properties["Iptc4xmpExt:DigitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    original = Image.open(pair.original.path)
    assert original.size == (352, 288) and extract_xmp(original) is None
