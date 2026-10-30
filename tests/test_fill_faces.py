from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np

from app.services import fill_faces


def test_face_cascade_is_independent_per_thread(monkeypatch) -> None:
    created = []
    lock = threading.Lock()

    class FakeCascade:
        pass

    def create_cascade(_path):
        cascade = FakeCascade()
        with lock:
            created.append(cascade)
        return cascade

    monkeypatch.setattr(cv2, "CascadeClassifier", create_cascade)
    barrier = threading.Barrier(2)

    def get_cascade(_index):
        barrier.wait()
        return fill_faces.get_face_cascade()

    with ThreadPoolExecutor(max_workers=2) as pool:
        cascades = list(pool.map(get_cascade, range(2)))

    assert cascades[0] is not cascades[1]
    assert len(created) == 2


def test_blank_image_returns_no_faces(monkeypatch) -> None:
    class EmptyCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return ()

    monkeypatch.setattr(fill_faces, "get_face_cascade", lambda: EmptyCascade())
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    assert fill_faces.faces_inside(image, mask) == []


def test_face_box_is_dropped_when_center_is_outside_mask(monkeypatch) -> None:
    class FakeCascade:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[10, 10, 20, 20], [40, 30, 20, 20]])

    monkeypatch.setattr(fill_faces, "get_face_cascade", lambda: FakeCascade())
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[70:100, 100:140] = 255

    boxes = fill_faces.faces_inside(image, mask)

    assert len(boxes) == 1
    assert boxes[0] == (80, 60, 40, 40)


def test_face_found_only_at_the_coarser_scale_is_kept(monkeypatch) -> None:
    class CoarseOnlyCascade:
        def detectMultiScale(self, gray, *args, **kwargs):
            # Solo "ve" la cara en la imagen reducida a 0.25 (200 px -> 50 px).
            return np.array([[20, 20, 10, 10]]) if gray.shape[1] == 50 else ()

    monkeypatch.setattr(fill_faces, "get_face_cascade", lambda: CoarseOnlyCascade())
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[80:120, 80:120] = 255

    assert fill_faces.faces_inside(image, mask) == [(80, 80, 40, 40)]
