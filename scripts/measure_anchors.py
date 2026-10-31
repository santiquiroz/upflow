from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Evita que el entorno importe `app` desde otro checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from app.config import Settings
from app.services.person_detector import PersonBox, PersonDetector
from app.services.person_roles import (
    anchor_point,
    assign_faces,
    face_box,
    merge_duplicates,
)

COLLISION_WIDTH_FRACTION = 0.01
DEFAULT_OUT_DIR = Path("C:/personal/upflow-eval/2026-10-30-personas/anchors")


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _contains(box: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def count_collisions(anchors: list[tuple[int, int]], image_width: int) -> int:
    limit = COLLISION_WIDTH_FRACTION * image_width
    return sum(
        1
        for index, first in enumerate(anchors)
        for second in anchors[index + 1 :]
        if np.hypot(first[0] - second[0], first[1] - second[1]) < limit
    )


def count_stolen_anchors(
    anchors: list[tuple[int, int]], boxes: list[tuple[int, int, int, int]]
) -> int:
    return sum(
        1
        for index, anchor in enumerate(anchors)
        if any(
            other != index
            and _area(boxes[other]) < _area(boxes[index])
            and _contains(boxes[other], anchor)
            for other in range(len(boxes))
        )
    )


def _measure_path(
    rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    anchors: list[tuple[int, int]],
) -> dict[str, int]:
    return {
        "people": len(boxes),
        "collisions": count_collisions(anchors, rgb.shape[1]),
        "stolen_anchors": count_stolen_anchors(anchors, boxes),
    }


def measure_photo(rgb: np.ndarray, detections: list[PersonBox]) -> dict[str, Any]:
    old_boxes = [person.box for person in detections]
    old_faces = [face_box(rgb, box) for box in old_boxes]
    old_anchors = [anchor_point(box, face) for box, face in zip(old_boxes, old_faces)]

    merged = merge_duplicates(detections)
    new_boxes = [person.box for person in merged]
    new_faces = assign_faces(rgb, new_boxes)
    new_anchors = [anchor_point(box, face) for box, face in zip(new_boxes, new_faces)]

    old = _measure_path(rgb, old_boxes, old_anchors)
    new = _measure_path(rgb, new_boxes, new_anchors)
    new["merged_duplicates"] = len(detections) - len(merged)
    old["merged_duplicates"] = 0
    return {"old": old, "new": new}


def _sum_counts(rows: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(row[key] for row in rows) for key in rows[0]} if rows else {}


def measure(photos: Path, device: str) -> dict[str, Any]:
    detector = PersonDetector(Settings())
    per_photo: dict[str, Any] = {}
    for path in sorted(photos.glob("*.jpg")):
        rgb = _read_rgb(path)
        detections = detector.detect(rgb, device)
        per_photo[path.name] = measure_photo(rgb, detections)

    old_rows = [photo["old"] for photo in per_photo.values()]
    new_rows = [photo["new"] for photo in per_photo.values()]
    return {
        "n_photos": len(per_photo),
        "old": _sum_counts(old_rows),
        "new": _sum_counts(new_rows),
        "per_photo": per_photo,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = measure(args.photos, args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "measure_anchors.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_photos", "old", "new")}, indent=2))


if __name__ == "__main__":
    main()
