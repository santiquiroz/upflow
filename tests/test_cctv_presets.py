import itertools
from typing import get_args

import pytest

from app.services.cctv_chain import CctvChainError, steps_from_request
from app.services.cctv_presets import (
    CCTV_PRESETS,
    PresetContext,
    PresetId,
    UNKNOWN_PRESET,
    preset_spec,
    preset_steps,
    presets_schema,
)
from app.services.video_analysis import PresetId as DiagnosisPresetId

LITE_SAR = (2, 1)
CONTEXTS = [
    PresetContext(interlaced=interlaced, sample_aspect=sar)
    for interlaced, sar in itertools.product((False, True), (None, LITE_SAR))
]


def _ids(steps):
    return [step["id"] for step in steps]


def _step(steps, step_id):
    return next(step for step in steps if step["id"] == step_id)


def test_the_catalog_has_the_four_presets_the_diagnosis_can_suggest():
    assert [preset.id for preset in CCTV_PRESETS] == ["day", "night_ir", "analog", "low_res"]
    assert set(get_args(PresetId)) == set(get_args(DiagnosisPresetId))


def test_every_preset_carries_its_text_keys():
    labels = {preset.id: preset.label for preset in CCTV_PRESETS}
    assert labels == {
        "day": "Day",
        "night_ir": "Night / IR",
        "analog": "Analog (interlaced)",
        "low_res": "Low-res sub-stream",
    }
    for preset in CCTV_PRESETS:
        assert preset.label_key == f"cctv.preset.{preset.id}"
        assert preset.description_key == f"cctv.preset.{preset.id}.description"
        assert preset.description.endswith(".")


@pytest.mark.parametrize("preset", [preset.id for preset in CCTV_PRESETS])
@pytest.mark.parametrize("lane", ["classic", "ai"])
@pytest.mark.parametrize("context", CONTEXTS)
def test_every_preset_passes_the_chain_validation_in_both_lanes(preset, lane, context):
    resolved = steps_from_request(preset_steps(preset, lane, context), lane)
    assert "osd_protect" in [step.id for step in resolved]


@pytest.mark.parametrize("preset", [preset.id for preset in CCTV_PRESETS])
@pytest.mark.parametrize("lane", ["classic", "ai"])
def test_no_preset_sharpens_interpolates_or_computes_new_pixels(preset, lane):
    context = PresetContext(interlaced=True, sample_aspect=LITE_SAR)
    steps = preset_steps(preset, lane, context)
    assert "sharpen" not in _ids(steps)
    assert "interpolate" not in _ids(steps)
    for step in steps:
        if step["id"] == "scale":
            assert step["params"]["flags"] == "neighbor"


def test_day_classic_is_weak_deblock_then_hqdn3d():
    steps = preset_steps("day", "classic", PresetContext())
    assert _ids(steps) == ["deblock", "denoise", "osd_protect"]
    assert _step(steps, "deblock")["params"] == {"filter": "deblock", "filter_type": "weak", "block": 8}
    assert _step(steps, "denoise")["params"] == {
        "filter": "hqdn3d",
        "luma_spatial": 3,
        "chroma_spatial": 2,
        "luma_tmp": 4,
        "chroma_tmp": 3,
    }


def test_conditional_steps_only_appear_when_the_diagnosis_found_them():
    context = PresetContext(interlaced=True, sample_aspect=LITE_SAR)
    steps = preset_steps("day", "classic", context)
    assert _ids(steps) == ["aspect", "deinterlace", "deblock", "denoise", "osd_protect"]
    assert _step(steps, "aspect")["params"] == {"num": 2, "den": 1}
    assert _step(steps, "deinterlace")["params"] == {"filter": "bwdif", "mode": "send_frame"}


def test_a_square_pixel_ratio_needs_no_aspect_step():
    steps = preset_steps("day", "classic", PresetContext(sample_aspect=(1, 1)))
    assert "aspect" not in _ids(steps)


def test_day_ai_replaces_deblock_and_denoise_with_ai_deblock_and_no_upscale():
    steps = preset_steps("day", "ai", PresetContext())
    assert _ids(steps) == ["ai_deblock", "osd_protect"]
    assert _step(steps, "ai_deblock")["params"] == {"strength": 40}


def test_night_classic_is_strong_deblock_temporal_denoise_gray_and_gamma():
    steps = preset_steps("night_ir", "classic", PresetContext())
    assert _ids(steps) == ["deblock", "denoise", "gray", "levels", "osd_protect"]
    assert _step(steps, "deblock")["params"]["filter_type"] == "strong"
    assert _step(steps, "denoise")["params"] == {
        "filter": "atadenoise",
        "0a": 0.04,
        "0b": 0.08,
        "1a": 0.04,
        "1b": 0.08,
        "s": 9,
    }
    assert _step(steps, "levels")["params"] == {"filter": "eq", "gamma": 1.2}


def test_night_ai_uses_a_stronger_ai_deblock_and_keeps_gray_and_gamma():
    steps = preset_steps("night_ir", "ai", PresetContext())
    assert _ids(steps) == ["ai_deblock", "gray", "levels", "osd_protect"]
    assert _step(steps, "ai_deblock")["params"] == {"strength": 60}


def test_analog_always_deinterlaces_keeping_the_frame_count():
    for lane in ("classic", "ai"):
        steps = preset_steps("analog", lane, PresetContext(interlaced=False))
        assert _step(steps, "deinterlace")["params"] == {"filter": "bwdif", "mode": "send_frame"}


def test_low_res_doubles_with_nearest_neighbor_and_suggests_ai_upscale():
    steps = preset_steps("low_res", "classic", PresetContext())
    assert _ids(steps) == ["deblock", "denoise", "scale", "osd_protect"]
    assert _step(steps, "scale")["params"] == {"factor": 2, "flags": "neighbor"}
    assert preset_spec("low_res").ai_upscale_hint == 2
    assert all(preset.ai_upscale_hint is None for preset in CCTV_PRESETS if preset.id != "low_res")


def test_presets_return_fresh_copies():
    first = preset_steps("day", "classic", PresetContext())
    first[0]["params"]["block"] = 16
    assert preset_steps("day", "classic", PresetContext())[0]["params"]["block"] == 8


def test_an_unknown_preset_is_rejected():
    with pytest.raises(CctvChainError) as caught:
        preset_spec("sunset")
    assert caught.value.code == UNKNOWN_PRESET


def test_the_presets_schema_lists_both_lanes():
    schema = presets_schema()
    day = next(preset for preset in schema if preset["id"] == "day")
    assert day["labelKey"] == "cctv.preset.day"
    assert _ids(day["lanes"]["classic"]) == ["aspect", "deinterlace", "deblock", "denoise", "osd_protect"]
    assert _step(day["lanes"]["classic"], "aspect")["when"] == "anamorphic"
    assert _ids(day["lanes"]["ai"]) == ["aspect", "deinterlace", "ai_deblock", "osd_protect"]
