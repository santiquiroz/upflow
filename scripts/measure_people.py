from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

# Evita que el entorno importe `app` desde otro checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.config import Settings
from app.services.editor_segmenter import EditorSegmenter
from app.services.person_detector import PersonBox, PersonDetector
from app.services.person_roles import classify_roles

TILE_SIZE = 256
TILES_PER_SHEET = 24
TILES_PER_ROW = 4
MASK_COVERAGE_THRESHOLD = 0.5
MASK_MERGE_THRESHOLD = 0.5
LARGE_SUBJECT_AREA_THRESHOLD = 0.02


@dataclass
class Objective:
    photo_stem: str
    index: int
    mask: np.ndarray
    kind: str
    assigned_box: tuple[int, int, int, int] | None = None
    assigned_role: str | None = None

    @property
    def id(self) -> str:
        return f"{self.photo_stem}:{self.index}"


def should_merge_masks(first: np.ndarray, second: np.ndarray) -> bool:
    smaller_area = min(int(np.count_nonzero(first)), int(np.count_nonzero(second)))
    if smaller_area == 0:
        return False
    intersection = int(np.count_nonzero(first & second))
    return intersection / smaller_area > MASK_MERGE_THRESHOLD


def box_mask_coverage(
    mask: np.ndarray, box: tuple[int, int, int, int]
) -> float:
    mask_area = int(np.count_nonzero(mask))
    if mask_area == 0:
        return 0.0
    height, width = mask.shape
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return int(np.count_nonzero(mask[y1:y2, x1:x2])) / mask_area


def assign_box(
    mask: np.ndarray, people: list[PersonBox]
) -> tuple[int, PersonBox] | None:
    candidates = [
        (index, person)
        for index, person in enumerate(people)
        if box_mask_coverage(mask, person.box) >= MASK_COVERAGE_THRESHOLD
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            max(0, item[1].box[2] - item[1].box[0])
            * max(0, item[1].box[3] - item[1].box[1]),
            item[0],
        ),
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def aggregate_metrics(
    objectives: list[Objective],
    n_photos: int,
    photo_sizes: dict[str, tuple[int, int]],
    detection_times: list[float],
    excluded_ids: set[str] | None = None,
) -> dict[str, Any]:
    excluded_ids = excluded_ids or set()
    included = [item for item in objectives if item.id not in excluded_ids]
    detected = [item for item in included if item.assigned_box is not None]
    correct_roles = sum(
        item.assigned_role == ("subject" if item.kind == "protect" else "background")
        for item in detected
    )
    missed = [item.id for item in included if item.assigned_box is None]

    misclassified_subjects = {photo_stem: [] for photo_stem in photo_sizes}
    large_misclassified_photos: set[str] = set()
    for item in included:
        if item.kind != "protect" or item.assigned_role != "background":
            continue
        misclassified_subjects.setdefault(item.photo_stem, []).append(item.id)
        width, height = photo_sizes[item.photo_stem]
        if (
            width > 0
            and height > 0
            and np.count_nonzero(item.mask) / (width * height)
            >= LARGE_SUBJECT_AREA_THRESHOLD
        ):
            large_misclassified_photos.add(item.photo_stem)

    return {
        "n_photos": n_photos,
        "n_objectives": len(included),
        "recall": len(detected) / len(included) if included else 0.0,
        "role_accuracy": correct_roles / len(detected) if detected else 0.0,
        "subjects_proposed_as_background": dict(sorted(misclassified_subjects.items())),
        "large_subjects_proposed_as_background_photos": len(large_misclassified_photos),
        "detection_time_seconds": {
            "p50": _percentile(detection_times, 50),
            "p95": _percentile(detection_times, 95),
        },
        "undetected_objective_ids": missed,
    }


def _read_mask(png_bytes: bytes) -> np.ndarray:
    with Image.open(BytesIO(png_bytes)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def _validated_photo_path(photos_dir: Path, file_name: str) -> Path:
    relative = Path(file_name)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name in {"", ".", ".."}
    ):
        raise ValueError(f"Invalid photo filename in spec: {file_name!r}")
    path = (photos_dir / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Photo not found: {path}")
    return path


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


async def _segment_clicks(
    path: Path,
    spec: dict[str, Any],
    photo_stem: str,
    segmenter: EditorSegmenter,
    device: str,
) -> list[Objective]:
    objectives: list[Objective] = []
    for kind in ("protect", "bg"):
        for click in spec.get(kind, []):
            if not isinstance(click, list) or len(click) != 2:
                raise ValueError(f"Invalid {kind} click for {spec['file']}: {click!r}")
            mask = _read_mask(
                await segmenter.segment(path, float(click[0]), float(click[1]), device)
            )
            matches = [
                (index, _mask_overlap_ratio(item.mask, mask))
                for index, item in enumerate(objectives)
                if should_merge_masks(item.mask, mask)
            ]
            if matches:
                objective_index = max(matches, key=lambda match: match[1])[0]
                objectives[objective_index].mask |= mask
            else:
                objectives.append(
                    Objective(
                        photo_stem, len(objectives) + 1, mask, kind
                    )
                )
    return objectives


def _mask_overlap_ratio(first: np.ndarray, second: np.ndarray) -> float:
    smaller_area = min(int(np.count_nonzero(first)), int(np.count_nonzero(second)))
    if smaller_area == 0:
        return 0.0
    return int(np.count_nonzero(first & second)) / smaller_area


def _assign_roles(
    objectives: list[Objective], rgb: np.ndarray, people: list[PersonBox]
) -> None:
    roles = classify_roles(rgb, people)
    for objective in objectives:
        assigned = assign_box(objective.mask, people)
        if assigned is None:
            continue
        box_index, person = assigned
        objective.assigned_box = person.box
        objective.assigned_role = roles[box_index]


def _objective_tile(rgb: np.ndarray, objective: Objective) -> Image.Image:
    ys, xs = np.nonzero(objective.mask)
    if len(xs) == 0:
        x_min, x_max, y_min, y_max = 0, rgb.shape[1], 0, rgb.shape[0]
    else:
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        padding_x = max(1, math.ceil((x_max - x_min) * 0.3))
        padding_y = max(1, math.ceil((y_max - y_min) * 0.3))
        x_min = max(0, x_min - padding_x)
        x_max = min(rgb.shape[1], x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(rgb.shape[0], y_max + padding_y)

    crop = rgb[y_min:y_max, x_min:x_max]
    crop_mask = objective.mask[y_min:y_max, x_min:x_max]
    scale = min(TILE_SIZE / crop.shape[1], TILE_SIZE / crop.shape[0])
    resized_size = (
        max(1, round(crop.shape[1] * scale)),
        max(1, round(crop.shape[0] * scale)),
    )
    left = (TILE_SIZE - resized_size[0]) // 2
    top = (TILE_SIZE - resized_size[1]) // 2
    tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "black")
    tile.paste(
        Image.fromarray(crop).resize(resized_size, Image.Resampling.LANCZOS),
        (left, top),
    )
    mask = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
        resized_size, Image.Resampling.NEAREST
    )
    outline = np.asarray(mask) > 0
    eroded = np.asarray(mask.filter(ImageFilter.MinFilter(3))) > 0
    pixels = np.asarray(tile).copy()
    content = pixels[top : top + resized_size[1], left : left + resized_size[0]]
    content[outline & ~eroded] = (255, 0, 0)
    tile = Image.fromarray(pixels)

    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, TILE_SIZE, 24), fill=(0, 0, 0))
    draw.text((5, 5), f"{objective.id} [{objective.kind}]", fill=(255, 255, 255))
    return tile


def write_contact_sheets(
    objectives: list[Objective], photo_paths: dict[str, Path], sheets_dir: Path
) -> None:
    undetected = [item for item in objectives if item.assigned_box is None]
    sheets_dir.mkdir(parents=True, exist_ok=True)
    sheet_number = 1
    batch: list[Image.Image] = []
    current_stem: str | None = None
    rgb: np.ndarray | None = None
    for objective in undetected:
        if objective.photo_stem != current_stem:
            rgb = _read_rgb(photo_paths[objective.photo_stem])
            current_stem = objective.photo_stem
        assert rgb is not None
        batch.append(_objective_tile(rgb, objective))
        if len(batch) == TILES_PER_SHEET:
            _save_contact_sheet(batch, sheets_dir, sheet_number)
            sheet_number += 1
            batch.clear()
    if batch:
        _save_contact_sheet(batch, sheets_dir, sheet_number)


def _save_contact_sheet(
    tiles: list[Image.Image], sheets_dir: Path, sheet_number: int
) -> None:
    rows = math.ceil(len(tiles) / TILES_PER_ROW)
    sheet = Image.new("RGB", (TILES_PER_ROW * TILE_SIZE, rows * TILE_SIZE), "white")
    for offset, tile in enumerate(tiles):
        sheet.paste(
            tile,
            (
                (offset % TILES_PER_ROW) * TILE_SIZE,
                (offset // TILES_PER_ROW) * TILE_SIZE,
            ),
        )
    sheet.save(sheets_dir / f"contact-sheet-{sheet_number:03}.jpg")


def _load_truth(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(encoding="utf-8") as file:
        truth = json.load(file)
    return {
        objective_id
        for photo_data in truth.values()
        for objective_id in photo_data.get("not_person", [])
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", dest="eval_dir", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--sheets", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    specs_path = args.eval_dir / "specs" / "all8.json"
    with specs_path.open(encoding="utf-8") as file:
        specs = json.load(file)
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list in {specs_path}")
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        specs = specs[: args.limit]

    settings = Settings()
    detector = PersonDetector(settings)
    segmenter = EditorSegmenter(settings)
    objectives: list[Objective] = []
    photo_sizes: dict[str, tuple[int, int]] = {}
    photo_paths: dict[str, Path] = {}
    detection_times: list[float] = []

    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("file"), str):
            raise ValueError(f"Invalid photo entry in {specs_path}: {spec!r}")
        path = _validated_photo_path(args.photos, spec["file"])
        photo_stem = path.stem
        rgb = _read_rgb(path)
        photo_paths[photo_stem] = path
        photo_sizes[photo_stem] = (rgb.shape[1], rgb.shape[0])
        started = time.perf_counter()
        people = detector.detect(rgb, args.device)
        detection_times.append(time.perf_counter() - started)
        photo_objectives = await _segment_clicks(
            path, spec, photo_stem, segmenter, args.device
        )
        _assign_roles(photo_objectives, rgb, people)
        objectives.extend(photo_objectives)

    excluded_ids = _load_truth(args.truth)
    metrics = aggregate_metrics(
        objectives,
        len(specs),
        photo_sizes,
        detection_times,
        excluded_ids,
    )
    if args.sheets is not None:
        included = [item for item in objectives if item.id not in excluded_ids]
        write_contact_sheets(included, photo_paths, args.sheets)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
