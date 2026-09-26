from __future__ import annotations

import pytest

from app.services.photo_restore_chain import is_overprocessing, step_ids, steps_from_selection
from app.services.photo_restore_presets import (
    DEFAULT_PRESET,
    FADED_CAST_MIN_DE,
    HEAVY_DAMAGE_MIN_COVERAGE,
    PHOTO_PRESETS,
    PRESET_CONDITIONS,
    PROPOSAL_PRIORITY,
    STRONG_FADED_CAST_MIN_DE,
    PhotoFacts,
    UnknownPhotoPreset,
    options_without_analysis,
    photo_preset,
    proposed_preset,
    resolve_preset,
    suggested_presets,
)

CLEAN = PhotoFacts()
EVERYTHING = PhotoFacts(
    damage_coverage=0.05,
    jpeg_blocking=True,
    noise_sigma=10 / 255,
    halftone=True,
    tone_kind="color",
    color_cast_de=12.0,
    max_eye_px=80.0,
)


def all_steps():
    return [step for preset in PHOTO_PRESETS for step in preset.steps]


def test_the_presets_are_the_five_of_the_spec_in_order() -> None:
    ids = tuple(preset.id for preset in PHOTO_PRESETS)

    assert ids == ("gentle", "heavy_damage", "newspaper", "faded_color_print", "portrait")
    assert DEFAULT_PRESET == "gentle"


def test_every_preset_step_is_a_known_chain_step_and_condition() -> None:
    for step in all_steps():
        steps_from_selection([step.step_id])
        assert step.when is None or step.when in PRESET_CONDITIONS


def test_every_suggestion_condition_is_known() -> None:
    for preset in PHOTO_PRESETS:
        assert preset.suggest_when is None or preset.suggest_when in PRESET_CONDITIONS


def test_no_preset_colorizes() -> None:
    assert all(step.step_id != "colorize" for step in all_steps())


def test_no_preset_neutralizes_or_drops_the_original_tone() -> None:
    tone_steps = [step for step in all_steps() if step.step_id == "tone"]

    assert tone_steps
    for step in tone_steps:
        assert step.options["keep_tone"] is True
        assert step.options["neutral_gray"] is False


def test_fix_faded_colors_is_always_conditioned_on_a_color_cast() -> None:
    for step in all_steps():
        if step.options.get("fix_faded"):
            assert step.when == "faded_cast"


def test_only_portrait_turns_on_faces() -> None:
    with_faces = {preset.id for preset in PHOTO_PRESETS if "faces" in preset.step_ids()}

    assert with_faces == {"portrait"}


def test_portrait_is_gentle_plus_faces_at_60_percent() -> None:
    gentle = photo_preset("gentle")
    portrait = photo_preset("portrait")
    faces = portrait.steps[-1]

    assert portrait.steps[:-1] == gentle.steps
    assert faces.step_id == "faces"
    assert faces.options["blend"] == pytest.approx(0.6)
    assert faces.when == "restorable_faces"


def test_the_preset_table_cannot_be_mutated() -> None:
    options = photo_preset("gentle").steps[0].options

    with pytest.raises(TypeError):
        options["sensitivity"] = 1.0  # type: ignore[index]


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(UnknownPhotoPreset):
        photo_preset("vivid")


def test_gentle_on_a_clean_photo_selects_nothing() -> None:
    selection = resolve_preset("gentle", CLEAN)

    assert selection.preset == "gentle"
    assert selection.steps == ()
    assert selection.options == {}


def test_gentle_turns_on_only_what_the_diagnosis_found() -> None:
    facts = PhotoFacts(damage_coverage=0.01, jpeg_blocking=True)

    selection = resolve_preset("gentle", facts)

    assert selection.steps == ("repair", "deblock")
    assert selection.options["repair"]["engine"] == "fast"
    assert selection.options["deblock"]["strength"] == pytest.approx(0.5)


def test_gentle_denoise_keeps_a_quarter_of_the_grain() -> None:
    selection = resolve_preset("gentle", PhotoFacts(noise_sigma=5 / 255))

    assert selection.steps == ("denoise",)
    assert selection.options["denoise"] == {"strength": pytest.approx(0.3), "keep_grain": pytest.approx(0.25)}


def test_fix_faded_colors_needs_a_cast_above_4_delta_e() -> None:
    below = resolve_preset("gentle", PhotoFacts(tone_kind="color", color_cast_de=FADED_CAST_MIN_DE))
    above = resolve_preset("gentle", PhotoFacts(tone_kind="color", color_cast_de=FADED_CAST_MIN_DE + 0.1))

    assert "tone" not in below.steps
    assert above.steps == ("tone",)
    assert above.options["tone"]["fix_faded"] is True
    assert above.options["tone"]["strength"] == pytest.approx(0.7)


@pytest.mark.parametrize("tone_kind", ["mono", "toned", "hand_tinted"])
def test_fix_faded_colors_never_touches_mono_toned_or_hand_tinted_photos(tone_kind: str) -> None:
    facts = PhotoFacts(tone_kind=tone_kind, color_cast_de=20.0)

    for preset in PHOTO_PRESETS:
        assert "tone" not in resolve_preset(preset.id, facts).steps, preset.id


def test_faded_color_print_strength_is_85_percent() -> None:
    selection = resolve_preset("faded_color_print", PhotoFacts(tone_kind="color", color_cast_de=9.0))

    assert selection.steps == ("denoise", "tone")
    assert selection.options["tone"]["strength"] == pytest.approx(0.85)


def test_heavy_damage_repairs_and_denoises_unconditionally() -> None:
    selection = resolve_preset("heavy_damage", CLEAN)

    assert selection.steps == ("repair", "denoise")
    assert selection.options["repair"]["grow_px"] == 2
    assert selection.options["repair"]["sensitivity"] > resolve_preset("gentle", EVERYTHING).options["repair"]["sensitivity"]
    assert selection.options["denoise"]["strength"] == pytest.approx(0.5)


def test_newspaper_descreens_halftone_and_denoises_lightly() -> None:
    selection = resolve_preset("newspaper", CLEAN)

    assert selection.steps == ("descreen", "denoise")
    assert selection.options["descreen"]["mode"] == "halftone"
    assert selection.options["denoise"]["strength"] == pytest.approx(0.2)


def test_portrait_without_restorable_faces_leaves_faces_off() -> None:
    selection = resolve_preset("portrait", PhotoFacts(max_eye_px=20.0))

    assert "faces" not in selection.steps


def test_portrait_with_a_restorable_face_turns_faces_on() -> None:
    selection = resolve_preset("portrait", PhotoFacts(max_eye_px=32.0))

    assert selection.steps == ("faces",)
    assert selection.options["faces"]["model"] == "gfpgan-v1.4"


def test_resolved_steps_are_in_catalog_order() -> None:
    for preset in PHOTO_PRESETS:
        steps = resolve_preset(preset.id, EVERYTHING).steps
        assert steps == step_ids(steps_from_selection(list(steps))), preset.id


def test_resolved_options_are_fresh_copies() -> None:
    first = resolve_preset("gentle", EVERYTHING)
    first.options["denoise"]["strength"] = 1.0

    second = resolve_preset("gentle", EVERYTHING)

    assert second.options["denoise"]["strength"] == pytest.approx(0.3)
    assert photo_preset("gentle").steps[2].options["strength"] == pytest.approx(0.3)


def test_a_clean_photo_proposes_gentle() -> None:
    assert proposed_preset(CLEAN) == "gentle"


def test_damage_above_3_percent_proposes_heavy_damage() -> None:
    assert proposed_preset(PhotoFacts(damage_coverage=HEAVY_DAMAGE_MIN_COVERAGE)) == "gentle"
    assert proposed_preset(PhotoFacts(damage_coverage=HEAVY_DAMAGE_MIN_COVERAGE + 0.001)) == "heavy_damage"


def test_a_halftone_proposes_newspaper_even_with_heavy_damage() -> None:
    assert proposed_preset(PhotoFacts(halftone=True, damage_coverage=0.2)) == "newspaper"


def test_a_strong_cast_on_a_color_photo_proposes_faded_color_print() -> None:
    color = PhotoFacts(tone_kind="color", color_cast_de=STRONG_FADED_CAST_MIN_DE + 0.1)
    sepia = PhotoFacts(tone_kind="toned", color_cast_de=STRONG_FADED_CAST_MIN_DE + 0.1)

    assert proposed_preset(color) == "faded_color_print"
    assert proposed_preset(sepia) == "gentle"


def test_portrait_is_suggested_but_never_proposed() -> None:
    facts = PhotoFacts(max_eye_px=64.0)

    assert proposed_preset(facts) == "gentle"
    assert "portrait" in suggested_presets(facts)


def test_suggestions_list_every_matching_preset_in_table_order() -> None:
    assert suggested_presets(EVERYTHING) == ("heavy_damage", "newspaper", "faded_color_print", "portrait")
    assert suggested_presets(CLEAN) == ()


def test_no_preset_repeats_a_step() -> None:
    for preset in PHOTO_PRESETS:
        assert len(set(preset.step_ids())) == len(preset.steps), preset.id


def test_no_preset_overprocesses_even_when_everything_is_detected() -> None:
    for preset in PHOTO_PRESETS:
        selection = resolve_preset(preset.id, EVERYTHING)
        descreen = selection.options.get("descreen", {"mode": "auto"})
        denoise = selection.options.get("denoise", {"strength": 0.0})
        assert not is_overprocessing(selection.steps, descreen["mode"], denoise["strength"]), preset.id


def test_proposals_only_name_suggestible_presets_and_never_portrait() -> None:
    suggestible = {preset.id for preset in PHOTO_PRESETS if preset.suggest_when is not None}

    assert set(PROPOSAL_PRIORITY) <= suggestible
    assert "portrait" not in PROPOSAL_PRIORITY


def test_without_analysis_a_preset_never_turns_on_fix_faded() -> None:
    tone = next(step for step in photo_preset("gentle").steps if step.step_id == "tone")
    assert tone.options["fix_faded"] is True
    assert options_without_analysis(tone) == {**tone.options, "fix_faded": False}


def test_without_analysis_the_other_conditional_steps_keep_their_tuning() -> None:
    repair = next(step for step in photo_preset("gentle").steps if step.step_id == "repair")
    assert options_without_analysis(repair) == dict(repair.options)
