from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from app.schemas import EditorPerson
from app.services.person_detector import (
    PersonBox,
    PersonDetector,
    PersonDetectorUnavailableError,
)
from app.services.person_roles import (
    Role,
    anchor_point,
    assign_faces,
    classify_roles,
    merge_duplicates,
)
from app.services.missing_pack import missing_pack_message

PEOPLE_CACHE_SIZE = 4
PeopleCacheKey = tuple[str, str]
CachedPeople = tuple[EditorPerson, ...]


class EditorPeople:
    def __init__(self, detector: PersonDetector) -> None:
        self.detector = detector
        self._lock = threading.Lock()
        self._cache: OrderedDict[PeopleCacheKey, CachedPeople] = OrderedDict()

    def available(self) -> bool:
        return self.detector.available()

    async def people(self, image_path: Path, device: str) -> list[EditorPerson]:
        if not self.available():
            raise PersonDetectorUnavailableError(missing_pack_message("person-detector"))
        return await asyncio.to_thread(self._people_blocking, image_path, device)

    def _people_blocking(self, image_path: Path, device: str) -> list[EditorPerson]:
        cache_key = (f"{image_path}:{image_path.stat().st_mtime_ns}", device)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        detections = self.detector.detect(rgb, device)
        people = _build_people(rgb, detections)
        self._cache_put(cache_key, people)
        return people

    def _cache_get(self, key: PeopleCacheKey) -> list[EditorPerson] | None:
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
        return [_copy_person(person) for person in cached]

    def _cache_put(self, key: PeopleCacheKey, people: list[EditorPerson]) -> None:
        cached = tuple(_copy_person(person) for person in people)
        with self._lock:
            self._cache[key] = cached
            self._cache.move_to_end(key)
            while len(self._cache) > PEOPLE_CACHE_SIZE:
                self._cache.popitem(last=False)


def _copy_person(person: EditorPerson) -> EditorPerson:
    return person.model_copy(deep=True)


def _build_people(rgb: np.ndarray, detections: list[PersonBox]) -> list[EditorPerson]:
    people = merge_duplicates(detections)
    roles = classify_roles(rgb, people)
    faces = assign_faces(rgb, [person.box for person in people])
    return [
        _to_person(index, person, roles[index], faces[index])
        for index, person in enumerate(people)
    ]


def _to_person(
    index: int, detection: PersonBox, role: Role, face: tuple[int, int, int, int] | None
) -> EditorPerson:
    anchor = anchor_point(detection.box, face)
    return EditorPerson(
        id=index,
        box=list(detection.box),
        score=detection.score,
        role=role,
        face=list(face) if face is not None else None,
        anchor=list(anchor),
    )
