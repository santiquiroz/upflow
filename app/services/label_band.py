"""Rotulo obligatorio del carril IA de CCTV (spec §4.7 puntos 8 y 10, §4.12) y de la copia anonimizada.

Banda bilingue fuera del area de imagen (`pad` + `overlay` de un PNG hecho con
Pillow y la fuente OFL bundleada) y una marca chica dentro de la imagen, para
que recortar la banda no deje el video sin rotulo. Es un rotulo, no una
proteccion contra manipulacion.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.services.ffmpeg_filters import FILTER_SEPARATOR, escape_filter_path
from app.services.xmp_packet import (
    DIGITAL_SOURCE_COMPOSITE,
    DIGITAL_SOURCE_ENHANCED,
    XmpFields,
    build_xmp_packet,
    insert_xmp_into_png,
)

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_PATH = FONTS_DIR / "SourceCodePro-Regular.ttf"
FONT_LICENSE_PATH = FONTS_DIR / "OFL.txt"
FONT_SHA256 = "74bd80d3e42a08517cd7e1108ba3d86f2da29ac0f3065be95e0357956ab9db37"
FONT_UNVERIFIED = "cctv.error.fontUnverified"

BAND_EN = "AI-ENHANCED VISUALIZATION — NOT ORIGINAL FOOTAGE"
BAND_ES = "VISUALIZACIÓN CON IA — NO ES LA GRABACIÓN ORIGINAL"
SEPARATOR = " — "
LANGUAGE_SEPARATOR = " / "
MARK_PREFIX = "AI · "
SHORT_JOB_ID_LENGTH = 8

BAND_NAME = "ai_label_band.png"
MARK_NAME = "ai_label_mark.png"

Rgba = tuple[int, int, int, int]
WHITE: Rgba = (255, 255, 255, 255)
BLACK: Rgba = (0, 0, 0, 255)
MARK_BACKGROUND: Rgba = (0, 0, 0, 176)

TEXT_FILL_RATIO = 0.94
MIN_FONT_SIZE = 8
BAND_FONT_DIVISOR = 36
BAND_FONT_FLOOR = 12
MARK_FONT_DIVISOR = 48
MARK_FONT_FLOOR = 10
MARK_MARGIN_DIVISOR = 90
MARK_MARGIN_FLOOR = 4
BAND_PADDING_RATIO = 0.4
MARK_PADDING_RATIO = 0.3


@dataclass(frozen=True, slots=True)
class LabelTexts:
    band_en: str
    band_es: str
    mark_prefix: str
    comment: str


AI_LABEL = LabelTexts(
    BAND_EN,
    BAND_ES,
    MARK_PREFIX,
    "AI-enhanced visualization by Upflow {version} (job {job_id}); not original footage",
)


class FontIntegrityError(RuntimeError):
    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f"The bundled font {path.name} can't be used: {detail}")
        self.key = FONT_UNVERIFIED
        self.path = path


# --- Fuente bundleada ---


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_font(path: Path = FONT_PATH, expected_sha256: str = FONT_SHA256) -> Path:
    if not path.is_file():
        raise FontIntegrityError(path, "the file is missing")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise FontIntegrityError(path, f"sha256 {actual} does not match the pinned {expected_sha256}")
    return path


def load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


# --- Textos del rotulo ---


def band_lines(version: str, job_id: str, texts: LabelTexts = AI_LABEL) -> tuple[str, str, str]:
    return (texts.band_en, texts.band_es, f"Upflow {version}{SEPARATOR}{job_id}")


def band_text(version: str, job_id: str, texts: LabelTexts = AI_LABEL) -> str:
    return f"{texts.band_en}{LANGUAGE_SEPARATOR}{texts.band_es}{SEPARATOR}Upflow {version}{SEPARATOR}{job_id}"


def short_job_id(job_id: str) -> str:
    return job_id[:SHORT_JOB_ID_LENGTH]


def mark_text(job_id: str, texts: LabelTexts = AI_LABEL) -> str:
    return f"{texts.mark_prefix}{short_job_id(job_id)}"


def metadata_comment(version: str, job_id: str, texts: LabelTexts = AI_LABEL) -> str:
    return texts.comment.format(version=version, job_id=job_id)


# --- Bloques de texto con Pillow ---


@dataclass(frozen=True, slots=True)
class TextBlock:
    lines: tuple[str, ...]
    size: int


def even_up(value: int) -> int:
    return value + (value % 2)


def widest_line(font: ImageFont.FreeTypeFont, lines: Sequence[str]) -> float:
    return max(font.getlength(line) for line in lines)


def fits(font_path: Path, lines: Sequence[str], size: int, max_width: float) -> bool:
    return widest_line(load_font(font_path, size), lines) <= max_width


def largest_fitting_size(font_path: Path, lines: Sequence[str], max_width: float, max_size: int) -> int | None:
    sizes = range(max(max_size, MIN_FONT_SIZE), MIN_FONT_SIZE - 1, -1)
    return next((size for size in sizes if fits(font_path, lines, size, max_width)), None)


def split_at_separator(lines: Sequence[str]) -> tuple[str, ...]:
    return tuple(part for line in lines for part in line.split(SEPARATOR))


def fitted_block(font_path: Path, lines: Sequence[str], width: int, max_size: int) -> TextBlock:
    max_width = width * TEXT_FILL_RATIO
    size = largest_fitting_size(font_path, lines, max_width, max_size)
    if size is not None:
        return TextBlock(tuple(lines), size)
    split = split_at_separator(lines)
    return TextBlock(split, largest_fitting_size(font_path, split, max_width, max_size) or MIN_FONT_SIZE)


def line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def block_padding(size: int, ratio: float) -> int:
    return max(2, round(size * ratio))


def block_height(font: ImageFont.FreeTypeFont, lines: Sequence[str], padding: int) -> int:
    return even_up(2 * padding + len(lines) * line_height(font))


def render_block(
    block: TextBlock, width: int, font_path: Path, background: Rgba, padding_ratio: float
) -> Image.Image:
    font = load_font(font_path, block.size)
    padding = block_padding(block.size, padding_ratio)
    image = Image.new("RGBA", (width, block_height(font, block.lines, padding)), background)
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(block.lines):
        draw.text((width // 2, padding + row * line_height(font)), line, font=font, fill=WHITE, anchor="ma")
    return image


def text_width(font_path: Path, block: TextBlock) -> int:
    return round(widest_line(load_font(font_path, block.size), block.lines))


# --- Banda y marca ---


def band_font_cap(frame_height: int) -> int:
    return max(BAND_FONT_FLOOR, frame_height // BAND_FONT_DIVISOR)


def mark_font_size(frame_height: int) -> int:
    return max(MARK_FONT_FLOOR, frame_height // MARK_FONT_DIVISOR)


def mark_margin(frame_height: int) -> int:
    return even_up(max(MARK_MARGIN_FLOOR, frame_height // MARK_MARGIN_DIVISOR))


def render_band(
    width: int, frame_height: int, version: str, job_id: str, font_path: Path, texts: LabelTexts = AI_LABEL
) -> Image.Image:
    block = fitted_block(font_path, band_lines(version, job_id, texts), width, band_font_cap(frame_height))
    return render_block(block, width, font_path, BLACK, BAND_PADDING_RATIO).convert("RGB")


def render_mark(frame_height: int, job_id: str, font_path: Path, texts: LabelTexts = AI_LABEL) -> Image.Image:
    block = TextBlock((mark_text(job_id, texts),), mark_font_size(frame_height))
    padding = block_padding(block.size, MARK_PADDING_RATIO)
    width = even_up(text_width(font_path, block) + 2 * padding)
    return render_block(block, width, font_path, MARK_BACKGROUND, MARK_PADDING_RATIO)


@dataclass(frozen=True, slots=True)
class LabelAssets:
    band: Path
    mark: Path
    band_height: int
    margin: int


def write_label_assets(
    directory: Path,
    width: int,
    height: int,
    version: str,
    job_id: str,
    font_path: Path = FONT_PATH,
    texts: LabelTexts = AI_LABEL,
) -> LabelAssets:
    font = verified_font(font_path)
    band = render_band(width, height, version, job_id, font, texts)
    band_path, mark_path = directory / BAND_NAME, directory / MARK_NAME
    band.save(band_path, format="PNG")
    render_mark(height, job_id, font, texts).save(mark_path, format="PNG")
    return LabelAssets(band_path, mark_path, band.height, mark_margin(height))


# --- Filtros de ffmpeg ---


def movie_source(path: Path, label: str) -> str:
    return f"movie={escape_filter_path(path)}[{label}]"


def label_band_graph(assets: LabelAssets, prefix: Sequence[str] = (), head_graph: str | None = None) -> str:
    # head_graph: un grafo que ya termina en [lb_image] (p. ej. el que protege el OSD del carril IA).
    m = assets.margin
    head = head_graph or FILTER_SEPARATOR.join([*prefix, "null"]) + "[lb_image]"
    return ";".join(
        [
            head,
            movie_source(assets.mark, "lb_mark"),
            f"[lb_image][lb_mark]overlay=x={m}:y=main_h-overlay_h-{m}:eval=init,"
            f"pad=w=iw:h=ih+{assets.band_height}:x=0:y=0:color=black[lb_padded]",
            movie_source(assets.band, "lb_band"),
            "[lb_padded][lb_band]overlay=x=0:y=main_h-overlay_h:eval=init",
        ]
    )


def label_band_args(
    assets: LabelAssets,
    version: str,
    job_id: str,
    prefix: Sequence[str] = (),
    head_graph: str | None = None,
    texts: LabelTexts = AI_LABEL,
) -> list[str]:
    graph = label_band_graph(assets, prefix, head_graph)
    return ["-vf", graph, "-metadata", f"comment={metadata_comment(version, job_id, texts)}"]


# --- XMP de los PNG exportados ---


def digital_source_type(generative: bool) -> str:
    return DIGITAL_SOURCE_COMPOSITE if generative else DIGITAL_SOURCE_ENHANCED


def still_xmp_packet(version: str, job_id: str, generative: bool) -> str:
    fields = XmpFields(digital_source_type(generative), band_text(version, job_id), f"Upflow {version}")
    return build_xmp_packet(fields)


def tag_png_with_xmp(path: Path, packet: str) -> None:
    path.write_bytes(insert_xmp_into_png(path.read_bytes(), packet))
