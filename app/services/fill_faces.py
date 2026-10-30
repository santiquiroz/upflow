from __future__ import annotations

import numpy as np

FACE_SCALE = 0.5
_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        import cv2

        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _cascade


def faces_inside(
    rgb: np.ndarray, mask: np.ndarray
) -> list[tuple[int, int, int, int]]:
    import cv2

    gray = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        None,
        fx=FACE_SCALE,
        fy=FACE_SCALE,
        interpolation=cv2.INTER_AREA,
    )
    boxes = _get_cascade().detectMultiScale(
        gray, 1.1, 5, minSize=(20, 20)
    )
    faces = []
    height, width = mask.shape
    for x, y, box_width, box_height in boxes:
        center_x = int(round((x + box_width / 2) / FACE_SCALE))
        center_y = int(round((y + box_height / 2) / FACE_SCALE))
        if not (0 <= center_x < width and 0 <= center_y < height):
            continue
        if mask[center_y, center_x] <= 0:
            continue
        faces.append((
            round(x / FACE_SCALE),
            round(y / FACE_SCALE),
            round(box_width / FACE_SCALE),
            round(box_height / FACE_SCALE),
        ))
    return faces
