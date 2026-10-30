import numpy as np
import pytest

from app.services.person_detector import PersonBox
from scripts.measure_guard import (
    DetectionGroup,
    GroupResult,
    calculate_metrics,
    evaluate_grid,
    parse_group_id,
)


def test_parse_group_id_uses_version_file_number_and_group():
    assert parse_group_id("8", "_DSC2145.jpg", 2) == "8:2145g2"


def test_parse_group_id_rejects_unexpected_filename():
    with pytest.raises(ValueError):
        parse_group_id("8", "photo.jpg", 0)


def test_calculate_metrics_counts_labels_and_finder_sources():
    metrics = calculate_metrics(
        [
            GroupResult("8:1g0", "persona", True, False),
            GroupResult("8:2g0", "persona", False, False),
            GroupResult("8:3g0", "limpio", True, True),
            GroupResult("8:4g0", "limpio", False, False),
            GroupResult("8:5g0", "sin revisar", True, False),
        ]
    )

    assert metrics == {
        "hits": 1,
        "total_persona": 2,
        "misses": ["8:2g0"],
        "false_alarms": ["8:3g0"],
        "total_limpio": 2,
        "sin_revisar": ["8:5g0"],
        "flagged": [
            {"id": "8:1g0", "finder": "face"},
            {"id": "8:3g0", "finder": "both"},
            {"id": "8:5g0", "finder": "face"},
        ],
    }


def test_evaluate_grid_uses_face_result_and_people_inside():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    groups = [
        DetectionGroup(
            "8:1g0",
            "persona",
            mask,
            [PersonBox((2, 2, 8, 8), 0.4)],
            False,
        ),
        DetectionGroup("8:2g0", "limpio", mask, [], True),
    ]

    results = evaluate_grid(groups, [0.2, 0.5], [0.5])

    assert results == [
        {
            "min_score": 0.2,
            "inside_ratio": 0.5,
            "hits": 1,
            "total_persona": 1,
            "false_alarms": 1,
            "total_limpio": 1,
            "misses": [],
            "false_alarm_ids": ["8:2g0"],
        },
        {
            "min_score": 0.5,
            "inside_ratio": 0.5,
            "hits": 0,
            "total_persona": 1,
            "false_alarms": 1,
            "total_limpio": 1,
            "misses": ["8:1g0"],
            "false_alarm_ids": ["8:2g0"],
        },
    ]
