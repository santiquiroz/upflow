from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.turnaround import (
    Box,
    EmptySheetError,
    column_runs,
    ink_bounds,
    panel_boxes,
    split_sheet,
    square_on_white,
)


def make_sheet(
    size: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> Image.Image:
    sheet = Image.new("RGB", size, "white")
    drawing = ImageDraw.Draw(sheet)
    for x0, y0, x1, y1 in rectangles:
        drawing.rectangle((x0, y0, x1 - 1, y1 - 1), fill="black")
    return sheet


def test_ink_bounds_returns_tight_half_open_box_and_rejects_empty_sheet() -> None:
    sheet = make_sheet((80, 60), [(13, 7, 42, 31)])

    assert ink_bounds(sheet) == Box(13, 7, 42, 31)

    with pytest.raises(EmptySheetError, match="no tiene tinta"):
        ink_bounds(Image.new("RGB", (80, 60), "white"))


def test_box_exposes_dimensions_and_tuple() -> None:
    box = Box(7, 11, 31, 46)

    assert box.width == 24
    assert box.height == 35
    assert box.as_tuple() == (7, 11, 31, 46)


def test_column_runs_splits_only_on_minimum_gap_and_closes_final_run() -> None:
    has_ink = np.array(
        [False, True, True, False, False, False, True, False, True, True],
        dtype=bool,
    )

    assert column_runs(has_ink, min_gap=3) == [(1, 3), (6, 10)]


def test_panel_boxes_orders_views_drops_narrow_bar_and_uses_tight_heights() -> None:
    panels = [
        (20, 10, 60, 80),
        (80, 20, 120, 90),
        (140, 5, 180, 70),
        (200, 30, 240, 95),
    ]
    narrow_height_bar = (270, 0, 280, 100)
    sheet = make_sheet((300, 100), [*panels, narrow_height_bar])

    boxes = panel_boxes(sheet)

    assert [box.as_tuple() for box in boxes] == panels

    with pytest.raises(EmptySheetError, match="no tiene tinta"):
        panel_boxes(Image.new("RGB", sheet.size, "white"))


def test_split_sheet_writes_unpadded_panels_in_order(tmp_path: Path) -> None:
    panels = [
        (10, 15, 40, 75),
        (60, 5, 100, 80),
        (125, 25, 180, 90),
    ]
    sheet = make_sheet((200, 100), panels)
    sheet_path = tmp_path / "sheet.png"
    sheet.save(sheet_path)
    out_dir = tmp_path / "nested" / "views"

    written = split_sheet(sheet_path, out_dir)

    assert out_dir.is_dir()
    assert written == [out_dir / f"view_{index:02d}.png" for index in range(3)]
    for path, box in zip(written, panel_boxes(sheet), strict=True):
        with Image.open(path) as view:
            assert view.format == "PNG"
            assert view.size == (box.width, box.height)


def test_square_on_white_centers_image_with_white_padding() -> None:
    image = Image.new("RGB", (40, 20), (12, 34, 56))

    squared = square_on_white(image)

    assert squared.width == squared.height
    assert squared.width > max(image.size)
    left = (squared.width - image.width) // 2
    top = (squared.height - image.height) // 2
    assert squared.crop((left, top, left + image.width, top + image.height)) == image
    assert [
        squared.getpixel((0, 0)),
        squared.getpixel((squared.width - 1, 0)),
        squared.getpixel((0, squared.height - 1)),
        squared.getpixel((squared.width - 1, squared.height - 1)),
    ] == [(255, 255, 255)] * 4
