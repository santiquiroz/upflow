from __future__ import annotations

import pytest

from app.services.restore_batch import batch_conflicts, is_batch, validate_batch


def test_batch_keeps_the_shared_face_settings() -> None:
    options = {"batch": True, "faces": {"model": "gfpgan-v1.4", "blend": 0.6}, "repair": {"use_user_mask": False}}

    assert batch_conflicts(options) == []
    validate_batch(options, session=None)


def test_batch_names_every_choice_made_on_another_photo() -> None:
    options = {
        "geometry": {"rotate90": 1},
        "faces": {"selected": [0], "per_face": {0: 0.4}},
        "tone": {"gray_point": (1, 2)},
        "repair": {"use_user_mask": True},
    }

    assert batch_conflicts(options) == [
        "geometry",
        "faces.selected",
        "faces.per_face",
        "tone.gray_point",
        "repair.use_user_mask",
    ]


def test_a_single_photo_may_carry_its_own_choices() -> None:
    validate_batch({"faces": {"selected": [0]}, "geometry": {"rotate90": 1}}, session="abc")

    assert not is_batch({"batch": False})


def test_a_batch_with_a_session_is_refused() -> None:
    with pytest.raises(ValueError, match="batch"):
        validate_batch({"batch": True}, session="abc")
