from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from app.config import Settings
from app.services.fill_faces import faces_inside
from app.services.fill_people import (
    PERSON_INSIDE_RATIO,
    PERSON_MIN_SCORE,
    _padded_crop_bounds,
    _translate_box,
    make_people_finder,
    people_inside,
)
from app.services.person_detector import PersonBox, PersonDetector

DEFAULT_PERSONA_IDS = {
    "8:2145g0",
    "8:2234g2",
    "8:2268g1",
    "8:2274g0",
    "8:2199g2",
    "8:2187g0",
    "9:2145g1",
    "9:2268g1",
    "9:2274g0",
    "9:2272g0",
    "9:2077g0",
    "9:2222g2",
    "9:2163g2",
    "9:2177g4",
}
SWEEP_MIN_SCORES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
SWEEP_INSIDE_RATIOS = (0.4, 0.5, 0.6, 0.7, 0.8)
TILE_SIZE = 256
TILES_PER_SHEET = 24
TILES_PER_ROW = 4
_FILE_PATTERN = re.compile(r"^_DSC(\d{4})\.jpg$", re.IGNORECASE)


@dataclass
class DetectionGroup:
    id: str
    label: str
    mask: np.ndarray
    boxes: list[PersonBox]
    face_detected: bool
    image_path: Path | None = None
    person_detected: bool | None = None


@dataclass(frozen=True)
class GroupResult:
    id: str
    label: str
    face_detected: bool
    person_detected: bool


def parse_group_id(version: str, file_name: str, group: int) -> str:
    match = _FILE_PATTERN.fullmatch(file_name)
    if version not in {"8", "9"} or match is None or group < 0:
        raise ValueError(f"Invalid evaluation group: {version!r}, {file_name!r}, {group!r}")
    return f"{version}:{match.group(1)}g{group}"


def calculate_metrics(groups: list[GroupResult]) -> dict[str, Any]:
    persona = [group for group in groups if group.label == "persona"]
    limpio = [group for group in groups if group.label == "limpio"]
    sin_revisar = [group.id for group in groups if group.label == "sin revisar"]
    misses = [
        group.id
        for group in persona
        if not (group.face_detected or group.person_detected)
    ]
    false_alarms = [
        group.id
        for group in limpio
        if group.face_detected or group.person_detected
    ]
    flagged = []
    for group in groups:
        if not (group.face_detected or group.person_detected):
            continue
        if group.face_detected and group.person_detected:
            finder = "both"
        elif group.face_detected:
            finder = "face"
        else:
            finder = "person"
        flagged.append({"id": group.id, "finder": finder})
    return {
        "hits": len(persona) - len(misses),
        "total_persona": len(persona),
        "misses": misses,
        "false_alarms": false_alarms,
        "total_limpio": len(limpio),
        "sin_revisar": sin_revisar,
        "flagged": flagged,
    }


def evaluate_grid(
    groups: list[DetectionGroup],
    min_scores: Iterable[float],
    inside_ratios: Iterable[float],
) -> list[dict[str, Any]]:
    results = []
    for min_score in min_scores:
        for inside_ratio in inside_ratios:
            group_results = [
                GroupResult(
                    group.id,
                    group.label,
                    group.face_detected,
                    bool(people_inside(group.boxes, group.mask, min_score, inside_ratio)),
                )
                for group in groups
            ]
            metrics = calculate_metrics(group_results)
            results.append(
                {
                    "min_score": min_score,
                    "inside_ratio": inside_ratio,
                    "hits": metrics["hits"],
                    "total_persona": metrics["total_persona"],
                    "false_alarms": len(metrics["false_alarms"]),
                    "total_limpio": metrics["total_limpio"],
                    "misses": metrics["misses"],
                    "false_alarm_ids": metrics["false_alarms"],
                }
            )
    return sorted(results, key=lambda result: (-result["hits"], result["false_alarms"]))


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _read_log_groups(eval_dir: Path) -> list[tuple[str, Path, Path]]:
    groups = []
    for version, folder_name, log_name in (
        ("8", "out8", "all8.log"),
        ("9", "out9", "fix9.log"),
    ):
        folder = eval_dir / folder_name
        log_path = folder / log_name
        with log_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.startswith('{"file"'):
                    continue
                entry = json.loads(line)
                if not isinstance(entry, dict) or not isinstance(entry.get("groups"), list):
                    raise ValueError(f"Invalid group entry in {log_path}:{line_number}")
                file_name = entry.get("file")
                if not isinstance(file_name, str):
                    raise ValueError(f"Invalid file in {log_path}:{line_number}")
                match = _FILE_PATTERN.fullmatch(file_name)
                if match is None:
                    raise ValueError(f"Invalid filename in {log_path}:{line_number}: {file_name}")
                photo_dir = folder / f"_DSC{match.group(1)}"
                for group in entry["groups"]:
                    if not isinstance(group, dict) or group.get("engine") != "osor":
                        continue
                    group_number = group.get("group")
                    if not isinstance(group_number, int) or group_number < 0:
                        raise ValueError(f"Invalid group number in {log_path}:{line_number}")
                    if not isinstance(group.get("bbox"), list):
                        raise ValueError(f"Missing bbox in {log_path}:{line_number}")
                    group_id = parse_group_id(version, file_name, group_number)
                    groups.append(
                        (
                            group_id,
                            photo_dir / f"_DSC{match.group(1)}_editada.jpg",
                            photo_dir / f"mask_{group_number}.png",
                        )
                    )
    return groups


def _read_truth(path: Path | None, group_ids: set[str]) -> dict[str, str]:
    if path is None:
        labels = {
            group_id: "persona" if group_id in DEFAULT_PERSONA_IDS else "limpio"
            for group_id in group_ids
        }
        return labels

    with path.open(encoding="utf-8") as file:
        truth = json.load(file)
    if not isinstance(truth, dict):
        raise ValueError("Truth must be a JSON object")
    labels = {}
    for label in ("persona", "limpio"):
        ids = truth.get(label, [])
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"Truth field {label!r} must be a list of IDs")
        for group_id in ids:
            if group_id in labels:
                raise ValueError(f"Truth ID appears in both lists: {group_id}")
            labels[group_id] = label
    return {
        group_id: labels.get(group_id, "sin revisar")
        for group_id in group_ids
    }


def _detect_person_boxes(
    detector: PersonDetector, rgb: np.ndarray, mask: np.ndarray, device: str
) -> list[PersonBox]:
    if not np.any(mask):
        return []
    left, top, right, bottom = _padded_crop_bounds(mask, rgb.shape)
    detections = detector.detect(rgb[top:bottom, left:right], device=device)
    return [_translate_box(person, left, top) for person in detections]


def _load_detection_groups(
    paths: list[tuple[str, Path, Path]],
    labels: dict[str, str],
    detector: PersonDetector,
    device: str,
    min_score: float,
    inside_ratio: float,
    sweep: bool,
) -> list[DetectionGroup]:
    groups = []
    finder = make_people_finder(detector, device, min_score, inside_ratio)
    for group_id, image_path, mask_path in paths:
        rgb = _read_rgb(image_path)
        mask = _read_mask(mask_path)
        if sweep:
            boxes = _detect_person_boxes(detector, rgb, mask, device)
            person_detected = None
        else:
            boxes = []
            person_detected = bool(finder(rgb, mask))
        face_detected = bool(faces_inside(rgb, mask))
        groups.append(
            DetectionGroup(
                group_id,
                labels[group_id],
                mask,
                boxes,
                face_detected,
                image_path,
                person_detected,
            )
        )
    return groups


def _evaluate_at_thresholds(
    groups: list[DetectionGroup], min_score: float, inside_ratio: float
) -> tuple[list[GroupResult], dict[str, Any]]:
    results = [
        GroupResult(
            group.id,
            group.label,
            group.face_detected,
            (
                group.person_detected
                if group.person_detected is not None
                else bool(people_inside(group.boxes, group.mask, min_score, inside_ratio))
            ),
        )
        for group in groups
    ]
    return results, calculate_metrics(results)


def _tile(group: DetectionGroup, rgb: np.ndarray) -> Image.Image:
    rows, columns = np.nonzero(group.mask)
    if len(columns) == 0:
        left, top, right, bottom = 0, 0, rgb.shape[1], rgb.shape[0]
    else:
        left, right = int(columns.min()), int(columns.max()) + 1
        top, bottom = int(rows.min()), int(rows.max()) + 1
        pad_x = math.ceil((right - left) * 0.3)
        pad_y = math.ceil((bottom - top) * 0.3)
        left, top = max(0, left - pad_x), max(0, top - pad_y)
        right, bottom = min(rgb.shape[1], right + pad_x), min(rgb.shape[0], bottom + pad_y)

    crop = rgb[top:bottom, left:right]
    crop_mask = group.mask[top:bottom, left:right]
    scale = min(TILE_SIZE / crop.shape[1], TILE_SIZE / crop.shape[0])
    resized_size = (
        max(1, round(crop.shape[1] * scale)),
        max(1, round(crop.shape[0] * scale)),
    )
    offset = ((TILE_SIZE - resized_size[0]) // 2, (TILE_SIZE - resized_size[1]) // 2)
    tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE), "black")
    tile.paste(
        Image.fromarray(crop).resize(resized_size, Image.Resampling.LANCZOS),
        offset,
    )
    resized_mask = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
        resized_size, Image.Resampling.NEAREST
    )
    outline = np.asarray(resized_mask) > 0
    eroded = np.asarray(resized_mask.filter(ImageFilter.MinFilter(3))) > 0
    pixels = np.asarray(tile).copy()
    content = pixels[offset[1] : offset[1] + resized_size[1], offset[0] : offset[0] + resized_size[0]]
    content[outline & ~eroded] = (255, 0, 0)
    tile = Image.fromarray(pixels)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, TILE_SIZE, 24), fill="black")
    draw.text((5, 5), group.id, fill="white")
    return tile


def _save_sheet(tiles: list[Image.Image], path: Path) -> None:
    rows = math.ceil(len(tiles) / TILES_PER_ROW)
    sheet = Image.new("RGB", (TILES_PER_ROW * TILE_SIZE, rows * TILE_SIZE), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(
            tile,
            ((index % TILES_PER_ROW) * TILE_SIZE, (index // TILES_PER_ROW) * TILE_SIZE),
        )
    sheet.save(path, format="JPEG")


def write_contact_sheets(
    groups: list[DetectionGroup], ids: list[str], output_dir: Path, prefix: str
) -> None:
    selected = [group for group in groups if group.id in set(ids)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(selected), TILES_PER_SHEET):
        batch = selected[offset : offset + TILES_PER_SHEET]
        tiles = []
        for group in batch:
            if group.image_path is None:
                raise ValueError(f"Missing image path for contact sheet group: {group.id}")
            tiles.append(_tile(group, _read_rgb(group.image_path)))
        _save_sheet(tiles, output_dir / f"{prefix}-{offset // TILES_PER_SHEET + 1:03}.jpg")


def _print_metrics(metrics: dict[str, Any]) -> None:
    print(f"Persona: {metrics['hits']}/{metrics['total_persona']} aciertos")
    print(f"Limpio: {len(metrics['false_alarms'])}/{metrics['total_limpio']} falsas alarmas")
    print(f"Fallos persona: {', '.join(metrics['misses']) or 'ninguno'}")
    print(f"Falsas alarmas: {', '.join(metrics['false_alarms']) or 'ninguna'}")
    print(f"Sin revisar: {', '.join(metrics['sin_revisar']) or 'ninguno'}")
    for group in metrics["flagged"]:
        print(f"Detectado {group['id']}: {group['finder']}")


def _print_grid(results: list[dict[str, Any]]) -> None:
    print("min_score inside_ratio hits/persona falsas_alarmas/limpio")
    for result in results:
        print(
            f"{result['min_score']:.2f} {result['inside_ratio']:.2f} "
            f"{result['hits']}/{result['total_persona']} "
            f"{result['false_alarms']}/{result['total_limpio']}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", dest="eval_dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--sheets", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-score", type=float, default=PERSON_MIN_SCORE)
    parser.add_argument("--inside-ratio", type=float, default=PERSON_INSIDE_RATIO)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = _read_log_groups(args.eval_dir)
    labels = _read_truth(args.truth, {group_id for group_id, _, _ in paths})
    settings = Settings()
    detector = PersonDetector(settings)
    groups = _load_detection_groups(
        paths,
        labels,
        detector,
        args.device,
        args.min_score,
        args.inside_ratio,
        args.sweep,
    )
    _, metrics = _evaluate_at_thresholds(groups, args.min_score, args.inside_ratio)

    output: dict[str, Any]
    if args.sweep:
        grid = evaluate_grid(groups, SWEEP_MIN_SCORES, SWEEP_INSIDE_RATIOS)
        _print_metrics(metrics)
        _print_grid(grid)
        output = {"metrics": metrics, "sweep": grid}
    else:
        _print_metrics(metrics)
        output = metrics

    if args.sheets is not None:
        write_contact_sheets(groups, metrics["false_alarms"], args.sheets, "false-alarms")
        write_contact_sheets(groups, metrics["misses"], args.sheets, "misses")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
