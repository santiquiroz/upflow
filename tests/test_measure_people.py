import numpy as np

from app.services.person_detector import PersonBox
from scripts.measure_people import (
    Objective,
    aggregate_metrics,
    assign_box,
    should_merge_masks,
)


def test_should_merge_masks_when_intersection_exceeds_half_of_smaller_mask():
    first = np.array([[1, 1, 1, 0]], dtype=bool)
    second = np.array([[1, 1, 0, 1]], dtype=bool)

    assert should_merge_masks(first, second)


def test_should_not_merge_masks_when_intersection_is_exactly_half():
    first = np.array([[1, 1, 0, 0]], dtype=bool)
    second = np.array([[1, 0, 1, 0]], dtype=bool)

    assert not should_merge_masks(first, second)


def test_assign_box_uses_smallest_box_covering_half_of_mask():
    mask = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    people = [
        PersonBox((0, 0, 4, 2), 0.9),
        PersonBox((0, 0, 3, 2), 0.8),
        PersonBox((0, 0, 1, 2), 0.7),
    ]

    assigned = assign_box(mask, people)

    assert assigned == (1, people[1])


def test_aggregate_metrics_excludes_truth_and_counts_large_misclassified_subjects():
    subject = Objective(
        photo_stem="_DSC1",
        index=1,
        mask=np.ones((10, 10), dtype=bool),
        kind="protect",
        assigned_box=(0, 0, 5, 5),
        assigned_role="background",
    )
    background = Objective(
        photo_stem="_DSC1",
        index=2,
        mask=np.ones((2, 2), dtype=bool),
        kind="bg",
        assigned_box=None,
        assigned_role=None,
    )
    excluded = Objective(
        photo_stem="_DSC2",
        index=1,
        mask=np.ones((2, 2), dtype=bool),
        kind="bg",
        assigned_box=None,
        assigned_role=None,
    )

    metrics = aggregate_metrics(
        [subject, background, excluded],
        n_photos=2,
        photo_sizes={"_DSC1": (20, 20), "_DSC2": (20, 20)},
        detection_times=[0.1, 0.3],
        excluded_ids={"_DSC2:1"},
    )

    assert metrics["n_photos"] == 2
    assert metrics["n_objectives"] == 2
    assert metrics["recall"] == 0.5
    assert metrics["role_accuracy"] == 0.0
    assert metrics["subjects_proposed_as_background"] == {
        "_DSC1": ["_DSC1:1"],
        "_DSC2": [],
    }
    assert metrics["large_subjects_proposed_as_background_photos"] == 1
    assert metrics["undetected_objective_ids"] == ["_DSC1:2"]
    assert metrics["detection_time_seconds"]["p50"] == 0.2
    assert metrics["detection_time_seconds"]["p95"] == 0.29
