from __future__ import annotations

import dataclasses

import pytest

from app.services.photo_restore_chain import (
    HALFTONE_DENOISE_LIMIT,
    RESTORE_CHAIN,
    RedundantRestoreSelection,
    RestoreStepSpec,
    UnknownRestoreStep,
    is_overprocessing,
    resolve_tone_flags,
    restore_step,
    step_ids,
    steps_from_selection,
)

CATALOG_ORDER = ("descreen", "repair", "deblock", "denoise", "tone", "faces", "colorize")


def spec(step_id: str, covers: tuple[str, ...]) -> RestoreStepSpec:
    return dataclasses.replace(restore_step("denoise"), id=step_id, covers=covers)


def test_the_catalog_runs_in_the_order_of_the_spec() -> None:
    assert step_ids(RESTORE_CHAIN) == CATALOG_ORDER


def test_native_steps_run_before_output_steps() -> None:
    phases = [step.phase for step in RESTORE_CHAIN]

    assert phases == sorted(phases, key=lambda phase: phase != "native")
    assert {step.id for step in RESTORE_CHAIN if step.phase == "output"} == {"faces", "colorize"}


def test_every_step_covers_its_own_family() -> None:
    for step in RESTORE_CHAIN:
        assert step.family in step.covers, step.id


def test_the_catalog_has_no_redundant_steps() -> None:
    assert step_ids(steps_from_selection(list(CATALOG_ORDER))) == CATALOG_ORDER


def test_only_repair_faces_and_colorize_invent_detail() -> None:
    inventing = {step.id for step in RESTORE_CHAIN if step.invents_detail}

    assert inventing == {"repair", "faces", "colorize"}


def test_every_inventing_step_names_its_warning() -> None:
    for step in RESTORE_CHAIN:
        assert (step.warning_key is not None) == step.invents_detail, step.id


def test_model_steps_name_their_pack_and_dsp_steps_need_none() -> None:
    packs = {step.id: step.pack for step in RESTORE_CHAIN}

    assert packs == {
        "descreen": None,
        "repair": "restore-core",
        "deblock": "restore-core",
        "denoise": "restore-core",
        "tone": None,
        "faces": "restore-faces",
        "colorize": "restore-colorize",
    }


def test_labels_follow_the_i18n_key_scheme() -> None:
    for step in RESTORE_CHAIN:
        assert step.label_key == f"restore.step.{step.id}"
        assert step.description_key == f"restore.step.{step.id}.description"


def test_the_order_comes_from_the_catalog_not_from_the_request() -> None:
    selected = steps_from_selection(["colorize", "denoise", "descreen"])

    assert step_ids(selected) == ("descreen", "denoise", "colorize")


def test_a_repeated_step_collapses_into_one_pass() -> None:
    assert step_ids(steps_from_selection(["tone", "tone"])) == ("tone",)


def test_an_empty_selection_is_an_empty_chain() -> None:
    assert steps_from_selection([]) == []


def test_unknown_steps_are_rejected_listing_the_valid_ones() -> None:
    with pytest.raises(UnknownRestoreStep) as raised:
        steps_from_selection(["denoise", "sharpen", "upscale"])

    message = str(raised.value)
    assert "sharpen, upscale" in message
    assert "descreen, repair, deblock" in message


def test_restore_step_rejects_an_unknown_id() -> None:
    with pytest.raises(UnknownRestoreStep):
        restore_step("sharpen")


def test_two_steps_covering_the_same_family_are_rejected_with_both_names() -> None:
    chain = (restore_step("deblock"), spec("denoise", ("denoise",)), spec("blind_denoise", ("denoise",)))

    with pytest.raises(RedundantRestoreSelection) as raised:
        steps_from_selection(["blind_denoise", "denoise"], chain=chain)

    message = str(raised.value)
    assert "'denoise'" in message and "'blind_denoise'" in message
    assert "noise" in message


def test_a_step_covering_two_families_excludes_both() -> None:
    chain = (
        restore_step("deblock"),
        spec("denoise", ("denoise",)),
        spec("deblock_and_denoise", ("deblock", "denoise")),
    )

    with pytest.raises(RedundantRestoreSelection):
        steps_from_selection(["deblock", "deblock_and_denoise"], chain=chain)


def test_steps_of_different_families_coexist_in_an_injected_chain() -> None:
    chain = (restore_step("deblock"), spec("denoise", ("denoise",)))

    assert step_ids(steps_from_selection(["denoise", "deblock"], chain=chain)) == ("deblock", "denoise")


def test_halftone_descreen_with_strong_denoise_is_overprocessing() -> None:
    steps = ["descreen", "denoise"]

    assert is_overprocessing(steps, descreen_mode="halftone", denoise_strength=0.5)


def test_halftone_descreen_with_a_light_denoise_is_not_overprocessing() -> None:
    steps = ["descreen", "denoise"]

    assert not is_overprocessing(steps, descreen_mode="halftone", denoise_strength=HALFTONE_DENOISE_LIMIT)


def test_paper_texture_descreen_does_not_add_smoothing() -> None:
    assert not is_overprocessing(["descreen", "denoise"], descreen_mode="texture", denoise_strength=1.0)


def test_overprocessing_needs_both_steps_selected() -> None:
    assert not is_overprocessing(["descreen"], descreen_mode="halftone", denoise_strength=1.0)
    assert not is_overprocessing(["denoise"], descreen_mode="halftone", denoise_strength=1.0)


def test_the_original_tone_is_kept_by_default() -> None:
    flags = resolve_tone_flags(["tone"])

    assert flags.keep_tone is True
    assert flags.neutral_gray is False


def test_neutral_gray_is_an_opt_in_that_replaces_keep_tone() -> None:
    flags = resolve_tone_flags(["tone"], keep_tone=True, neutral_gray=True)

    assert flags.keep_tone is False
    assert flags.neutral_gray is True


def test_colorize_turns_off_keep_original_tone() -> None:
    flags = resolve_tone_flags(["tone", "colorize"], keep_tone=True, neutral_gray=False)

    assert flags.keep_tone is False
    assert flags.neutral_gray is False


def test_colorize_keeps_an_explicit_neutral_gray_for_the_uncolored_result() -> None:
    flags = resolve_tone_flags(["colorize"], keep_tone=True, neutral_gray=True)

    assert flags.keep_tone is False
    assert flags.neutral_gray is True
