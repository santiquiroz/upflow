from __future__ import annotations

import numpy as np

# A 0.5 se escapan los fantasmas suaves que OSOR deja en un relleno (2145 en Upflow);
# a 0.25 se escapan otros: medido en 59 fotos, las dos juntas aciertan 6 de 6 sin falsas alarmas.
FACE_SCALES = (0.5, 0.25)
_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        import cv2

        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _cascade


def _center_in_mask(box: tuple[int, int, int, int], mask: np.ndarray) -> bool:
    x, y, width, height = box
    center_x, center_y = x + width // 2, y + height // 2
    rows, columns = mask.shape
    return 0 <= center_x < columns and 0 <= center_y < rows and mask[center_y, center_x] > 0


def _faces_at_scale(gray: np.ndarray, scale: float) -> list[tuple[int, int, int, int]]:
    import cv2

    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    boxes = _get_cascade().detectMultiScale(small, 1.1, 5, minSize=(20, 20))
    return [tuple(round(value / scale) for value in box) for box in boxes]


def faces_inside(
    rgb: np.ndarray, mask: np.ndarray
) -> list[tuple[int, int, int, int]]:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return [
        box
        for scale in FACE_SCALES
        for box in _faces_at_scale(gray, scale)
        if _center_in_mask(box, mask)
    ]
