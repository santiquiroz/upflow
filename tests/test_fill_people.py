from __future__ import annotations

import numpy as np

from app.services.fill_people import make_people_finder, people_inside
from app.services.person_detector import PersonBox


def test_people_inside_keeps_boxes_mostly_in_mask():
    mask = np.zeros((100, 100), bool)
    mask[10:60, 10:60] = True
    boxes = [
        PersonBox((12, 12, 50, 58), 0.8),
        PersonBox((40, 40, 99, 99), 0.9),
        PersonBox((15, 15, 30, 30), 0.1),
    ]

    assert people_inside(boxes, mask, min_score=0.3, inside_ratio=0.6) == [
        (12, 12, 50, 58)
    ]


def test_make_people_finder_detects_in_expanded_mask_crop_and_restores_coordinates():
    class Detector:
        def detect(self, rgb, device):
            assert rgb.shape == (30, 30, 3)
            assert device == "dml:0"
            return [PersonBox((5, 5, 25, 25), 0.8)]

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 30:50] = True

    finder = make_people_finder(Detector(), device="dml:0")

    assert finder(rgb, mask) == [(30, 20, 50, 40)]


def test_make_people_finder_clips_crop_at_image_edge_and_restores_coordinates():
    class Detector:
        def detect(self, rgb, device):
            assert rgb.shape == (25, 30, 3)
            return [PersonBox((5, 0, 25, 20), 0.8)]

    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:20, 20:40] = True

    finder = make_people_finder(Detector())

    assert finder(rgb, mask) == [(20, 0, 40, 20)]


def test_make_people_finder_does_not_detect_for_empty_mask():
    class Detector:
        def detect(self, _rgb, _device):
            raise AssertionError("no debe detectar una máscara vacía")

    finder = make_people_finder(Detector())

    assert finder(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10))) == []
