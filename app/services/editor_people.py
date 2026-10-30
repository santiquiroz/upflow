from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from app.schemas import EditorPerson
from app.services.person_detector import PersonDetector
from app.services.person_roles import classify_roles, face_box

PEOPLE_CACHE_SIZE = 4


class EditorPeople:
    def __init__(self, detector: PersonDetector) -> None:
        self.detector = detector
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], list[EditorPerson]] = OrderedDict()

    def available(self) -> bool:
        return self.detector.available()

    async def people(self, image_path: Path, device: str) -> list[EditorPerson]:
        return await asyncio.to_thread(self._people_blocking, image_path, device)

    def _people_blocking(self, image_path: Path, device: str) -> list[EditorPerson]:
        cache_key = (f"{image_path}:{image_path.stat().st_mtime_ns}", device)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        detections = self.detector.detect(rgb, device)
        roles = classify_roles(rgb, detections)
        people = [
            EditorPerson(
                id=index,
                box=list(detection.box),
                score=detection.score,
                role=roles[index],
                face=list(face) if (face := face_box(rgb, detection.box)) is not None else None,
            )
            for index, detection in enumerate(detections)
        ]

        with self._lock:
            self._cache[cache_key] = people
            self._cache.move_to_end(cache_key)
            while len(self._cache) > PEOPLE_CACHE_SIZE:
                self._cache.popitem(last=False)
        return people
