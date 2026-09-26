import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.cctv_chain import (
    AI_LANE_PLACEMENT,
    CCTV_CHAIN,
    FFMPEG_FILTERS_DOC,
    LENS_GATE,
    OPEN_GATES,
    STABILIZE_GATE,
    CctvChainError,
    ai_lane_plan,
    catalog_schema,
    limitations_for,
    step_spec,
    steps_from_request,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ids(steps):
    return [step.id for step in steps]


def _error_code(raw_steps, lane):
    with pytest.raises(CctvChainError) as caught:
        steps_from_request(raw_steps, lane)
    return caught.value.code


def test_the_catalog_keeps_the_fixed_capture_chain_order():
    assert _ids(CCTV_CHAIN) == [
        "trim",
        "aspect",
        "deinterlace",
        "deblock",
        "ai_deblock",
        "stabilize",
        "denoise",
        "lens",
        "crop",
        "gray",
        "levels",
        "scale",
        "ai_upscale",
        "sharpen",
        "osd_protect",
        "interpolate",
        "ai_label",
    ]


def test_every_step_and_filter_carries_its_text_keys_and_plain_description():
    label_keys = [step.label_key for step in CCTV_CHAIN]
    assert len(set(label_keys)) == len(label_keys)
    for step in CCTV_CHAIN:
        assert step.label_key == f"cctv.step.{step.id}"
        assert step.label
        for spec in step.filters:
            assert spec.description_key == f"cctv.filter.{spec.name}.description"
            assert spec.description.endswith(".")


def test_ffmpeg_filters_point_at_their_section_of_the_ffmpeg_docs():
    for step in CCTV_CHAIN:
        for spec in step.filters:
            if spec.ffmpeg_filters:
                assert spec.doc_url is not None
                assert spec.doc_url.startswith(FFMPEG_FILTERS_DOC + "#")
    assert step_spec("aspect").filters[0].doc_url.endswith("#setdar_002c-setsar")
    assert step_spec("denoise").filters[0].doc_url.endswith("#hqdn3d")


def test_ffmpeg_backed_filters_name_the_filters_the_build_must_have():
    stabilize = step_spec("stabilize").filters[0]
    assert stabilize.ffmpeg_filters == ("vidstabdetect", "vidstabtransform")
    assert step_spec("gray").filters[0].ffmpeg_filters == ("format",)
    assert step_spec("ai_deblock").filters[0].ffmpeg_filters == ()


def test_prohibited_filters_are_not_in_any_step():
    prohibited = {
        "minterpolate",
        "framerate",
        "fps",
        "mpdecimate",
        "decimate",
        "nlmeans_opencl",
        "nlmeans_vulkan",
        "bwdif_vulkan",
        "deshake_opencl",
        "libplacebo",
        "sr_amf",
        "vpp_amf",
    }
    used = {name for step in CCTV_CHAIN for spec in step.filters for name in spec.ffmpeg_filters}
    assert used.isdisjoint(prohibited)


@pytest.mark.parametrize("step_id", ["ai_deblock", "ai_upscale", "interpolate", "ai_label"])
def test_the_classic_lane_rejects_ai_steps_and_interpolation(step_id):
    code = _error_code([{"id": step_id, "params": {}}], "classic")
    assert code in {"cctv.error.stepNotInLane", "cctv.error.stepDeferred"}


def test_interpolation_is_out_of_v1_in_both_lanes():
    assert _error_code([{"id": "interpolate"}], "classic") == "cctv.error.stepDeferred"
    assert _error_code([{"id": "interpolate"}], "ai") == "cctv.error.stepDeferred"


def _ids_of_schema(schema):
    return [step["id"] for step in schema]


def test_lens_correction_runs_only_in_the_classic_lane_behind_its_gate():
    crop = {"id": "crop", "params": {"w": 64, "h": 48, "x": 0, "y": 0}}

    steps = steps_from_request([crop, {"id": "lens"}], "classic")

    assert LENS_GATE in OPEN_GATES
    assert _ids(steps) == ["lens", "crop"]
    assert (steps[0].filter, dict(steps[0].params)) == ("lenscorrection", {"k1": 0.0, "k2": 0.0})
    assert _error_code([{"id": "lens"}], "ai") == "cctv.error.stepNotInLane"
    assert "lens" in _ids_of_schema(catalog_schema("classic"))
    assert "lens" not in _ids_of_schema(catalog_schema("ai"))
    assert "lens" not in _ids_of_schema(catalog_schema("classic", frozenset({STABILIZE_GATE})))


def test_lens_correction_flattens_a_fisheye_with_bounded_angles():
    (step,) = steps_from_request([{"id": "lens", "params": {"filter": "v360", "pitch": -30}}], "classic")

    assert dict(step.params) == {"ih_fov": 180.0, "iv_fov": 180.0, "d_fov": 120.0, "yaw": 0.0, "pitch": -30}
    too_wide = {"id": "lens", "params": {"filter": "v360", "d_fov": 180}}
    assert _error_code([too_wide], "classic") == "cctv.error.invalidParam"
    assert _error_code([{"id": "lens", "params": {"k1": "0.1:k2=9"}}], "classic") == "cctv.error.invalidParam"


def _lens_presets():
    return [(spec, preset) for spec in step_spec("lens").filters for preset in spec.presets]


@pytest.mark.parametrize(("spec", "preset"), _lens_presets(), ids=lambda item: getattr(item, "name", ""))
def test_every_lens_preset_is_a_valid_request_for_its_filter(spec, preset):
    raw = {"id": "lens", "params": {"filter": spec.name, **preset.params}}

    (step,) = steps_from_request([raw], "classic")

    assert {name: step.params[name] for name in preset.params} == dict(preset.params)


def test_lens_presets_reach_the_catalog_with_their_text_keys():
    lens = next(step for step in catalog_schema("classic") if step["id"] == "lens")
    presets = {filter_["name"]: filter_["presets"] for filter_ in lens["filters"]}

    assert [preset["name"] for preset in presets["lenscorrection"]] == ["mild", "wide", "very_wide", "ultra_wide"]
    assert presets["lenscorrection"][2] == {
        "name": "very_wide",
        "labelKey": "cctv.filter.lenscorrection.preset.very_wide",
        "label": "Very wide angle (2.8 mm, common in dome cameras)",
        "params": {"k1": -0.22, "k2": -0.02},
    }
    assert all(preset["params"]["ih_fov"] >= 180 for preset in presets["v360"])


def test_lens_correction_declares_that_pixels_were_moved():
    steps = steps_from_request([{"id": "lens", "params": {"k1": -0.2}}], "classic")

    assert [limitation.key for limitation in limitations_for(steps)] == ["cctv.limitation.lensCorrected"]


def test_stabilize_is_open_because_its_determinism_test_passes():
    assert STABILIZE_GATE in OPEN_GATES


def test_stabilize_runs_only_in_the_classic_lane():
    raw = [{"id": "denoise"}, {"id": "stabilize", "params": {"shakiness": 7}}, {"id": "deblock"}]

    steps = steps_from_request(raw, "classic")

    assert _ids(steps) == ["deblock", "stabilize", "denoise"]
    assert dict(steps[1].params) == {"shakiness": 7, "smoothing": 10}
    assert _error_code([{"id": "stabilize"}], "ai") == "cctv.error.stepNotInLane"
    assert "stabilize" in _ids_of_schema(catalog_schema("classic"))
    assert "stabilize" not in _ids_of_schema(catalog_schema("ai"))


def test_a_closed_gate_hides_and_refuses_its_step():
    closed = frozenset()

    assert _ids_of_schema(catalog_schema("classic", closed)).count("stabilize") == 0
    with pytest.raises(CctvChainError) as caught:
        steps_from_request([{"id": "stabilize"}], "classic", closed)
    assert caught.value.code == "cctv.error.stepDeferred"


def test_stabilization_declares_that_frames_were_moved_and_resampled():
    steps = steps_from_request([{"id": "stabilize"}], "classic")

    assert [limitation.key for limitation in limitations_for(steps)] == ["cctv.limitation.stabilized"]


@pytest.mark.parametrize("step_id", ["denoise", "deblock"])
def test_the_ai_lane_rejects_classic_cleanup_combined_with_ai_deblock(step_id):
    raw = [{"id": "ai_deblock", "params": {"strength": 40}}, {"id": step_id}]
    assert _error_code(raw, "ai") == "cctv.error.stepConflict"


def test_unknown_steps_and_duplicates_are_rejected():
    assert _error_code([{"id": "vignette"}], "classic") == "cctv.error.unknownStep"
    assert _error_code([{"id": "gray"}, {"id": "gray"}], "classic") == "cctv.error.duplicateStep"
    assert _error_code(["gray"], "classic") == "cctv.error.invalidStep"
    assert _error_code([{"params": {}}], "classic") == "cctv.error.invalidStep"


def test_the_client_never_sends_filter_strings():
    raw = [{"id": "denoise", "params": {"filter": "hqdn3d=4:3:6:4.5"}}]
    assert _error_code(raw, "classic") == "cctv.error.unknownFilter"
    raw = [{"id": "denoise", "params": {"filter": "hqdn3d", "extra": "x,format=gray"}}]
    assert _error_code(raw, "classic") == "cctv.error.invalidParam"


@pytest.mark.parametrize(
    "params",
    [
        {"filter": "hqdn3d", "luma_spatial": "4"},
        {"filter": "hqdn3d", "luma_spatial": True},
        {"filter": "hqdn3d", "luma_spatial": -1},
        {"filter": "hqdn3d", "luma_spatial": 1000},
        {"filter": "hqdn3d", "luma_spatial": math.nan},
        {"filter": "hqdn3d", "luma_spatial": math.inf},
        {"filter": "atadenoise", "s": 8},
        {"filter": "atadenoise", "s": 9.0},
        {"filter": "fftdnoiz", "prev": 2},
    ],
)
def test_numeric_params_are_checked_against_their_schema(params):
    assert _error_code([{"id": "denoise", "params": params}], "classic") == "cctv.error.invalidParam"


def test_enum_params_only_accept_their_choices():
    raw = [{"id": "deblock", "params": {"filter": "deblock", "filter_type": "medium"}}]
    assert _error_code(raw, "classic") == "cctv.error.invalidParam"


def test_crop_needs_all_four_values_and_even_sizes():
    raw = [{"id": "crop", "params": {"w": 640, "h": 480, "x": 0}}]
    assert _error_code(raw, "classic") == "cctv.error.missingParam"
    raw = [{"id": "crop", "params": {"w": 641, "h": 480, "x": 0, "y": 0}}]
    assert _error_code(raw, "classic") == "cctv.error.invalidParam"
    (crop,) = steps_from_request([{"id": "crop", "params": {"w": 640, "h": 480, "x": 3, "y": 5}}], "classic")
    assert dict(crop.params) == {"w": 640, "h": 480, "x": 3, "y": 5}


def test_trim_end_cannot_come_before_start():
    raw = [{"id": "trim", "params": {"start_frame": 10, "end_frame": 9}}]
    assert _error_code(raw, "classic") == "cctv.error.invalidParam"
    (trim,) = steps_from_request([{"id": "trim", "params": {"start_frame": 10, "end_frame": 10}}], "classic")
    assert trim.params["end_frame"] == 10


def test_defaults_fill_missing_params_and_the_first_filter_is_the_default():
    (denoise,) = steps_from_request([{"id": "denoise", "params": {"luma_spatial": 3}}], "classic")
    assert denoise.filter == "hqdn3d"
    assert dict(denoise.params) == {
        "luma_spatial": 3,
        "chroma_spatial": 3.0,
        "luma_tmp": 6.0,
        "chroma_tmp": 4.5,
    }


def test_resolved_params_are_read_only():
    (gamma,) = steps_from_request([{"id": "levels", "params": {"filter": "eq", "gamma": 1.2}}], "classic")
    with pytest.raises(TypeError):
        gamma.params["gamma"] = 3.0


def test_the_resolved_chain_follows_the_catalog_order_not_the_request_order():
    raw = [{"id": "gray"}, {"id": "deblock"}, {"id": "deinterlace"}]
    assert _ids(steps_from_request(raw, "classic")) == ["deinterlace", "deblock", "gray"]


def test_send_field_deinterlacing_is_ai_lane_only():
    raw = [{"id": "deinterlace", "params": {"filter": "bwdif", "mode": "send_field"}}]
    assert _error_code(raw, "classic") == "cctv.error.invalidParam"
    steps = steps_from_request(raw, "ai")
    assert steps[0].params["mode"] == "send_field"


def test_deinterlacing_defaults_to_bwdif_keeping_the_frame_count():
    (deinterlace,) = steps_from_request([{"id": "deinterlace"}], "classic")
    assert deinterlace.filter == "bwdif"
    assert deinterlace.params["mode"] == "send_frame"


def test_scaling_defaults_to_nearest_neighbor():
    (scale,) = steps_from_request([{"id": "scale", "params": {"factor": 2}}], "classic")
    assert scale.params["flags"] == "neighbor"


def test_sharpen_is_available_but_off_unless_requested():
    assert step_spec("sharpen").default_on is False
    assert "sharpen" not in _ids(steps_from_request([{"id": "denoise"}], "classic"))
    (sharpen,) = steps_from_request([{"id": "sharpen"}], "classic")
    assert sharpen.filter == "cas"


def test_the_ai_label_is_always_appended_in_the_ai_lane():
    steps = steps_from_request([{"id": "ai_deblock"}], "ai")
    assert _ids(steps) == ["ai_deblock", "ai_label"]
    steps = steps_from_request([{"id": "ai_label"}, {"id": "gray"}], "ai")
    assert _ids(steps) == ["gray", "ai_label"]
    assert _ids(steps_from_request([], "ai")) == ["ai_label"]
    assert _ids(steps_from_request([], "classic")) == []


def test_the_ai_placement_table_is_exhaustive_for_the_ai_whitelist():
    ai_allowed = {step.id for step in CCTV_CHAIN if "ai" in step.lanes}
    assert set(AI_LANE_PLACEMENT) == ai_allowed
    for step_id in ("stabilize", "lens", "interpolate"):
        assert step_id not in AI_LANE_PLACEMENT


def test_ai_lane_plan_puts_each_step_where_the_stream_runs_it():
    raw = [
        {"id": "trim", "params": {"start_frame": 0, "end_frame": 99}},
        {"id": "deinterlace"},
        {"id": "crop", "params": {"w": 640, "h": 480, "x": 0, "y": 0}},
        {"id": "gray"},
        {"id": "ai_deblock", "params": {"strength": 60}},
        {"id": "ai_upscale"},
        {"id": "osd_protect"},
        {"id": "levels", "params": {"filter": "eq", "gamma": 1.2}},
        {"id": "scale", "params": {"factor": 2}},
        {"id": "sharpen"},
    ]
    plan = ai_lane_plan(steps_from_request(raw, "ai"))
    assert _ids(plan.decode) == ["trim", "deinterlace", "crop", "gray"]
    assert _ids(plan.composite) == ["ai_deblock", "ai_upscale", "osd_protect"]
    assert _ids(plan.encode) == ["levels", "scale", "sharpen", "ai_label"]


def test_ai_lane_plan_refuses_a_chain_without_the_label():
    (gray,) = steps_from_request([{"id": "gray"}], "classic")
    with pytest.raises(CctvChainError) as caught:
        ai_lane_plan((gray,))
    assert caught.value.code == "cctv.error.missingAiLabel"


def test_limitations_declare_geometry_changes_and_computed_pixels():
    steps = steps_from_request(
        [
            {"id": "crop", "params": {"w": 640, "h": 480, "x": 0, "y": 0}},
            {"id": "scale", "params": {"factor": 2, "flags": "lanczos"}},
            {"id": "sharpen"},
            {"id": "gray"},
        ],
        "classic",
    )
    keys = [limitation.key for limitation in limitations_for(steps)]
    assert keys == [
        "cctv.limitation.geometryChanged",
        "cctv.limitation.grayscale",
        "cctv.limitation.newPixelValues",
        "cctv.limitation.sharpenHalos",
    ]
    geometry = limitations_for(steps)[0]
    assert geometry.text == "Geometry was changed (crop/scale): sizes and positions differ from the original."


def test_nearest_neighbor_scaling_only_declares_the_geometry_change():
    steps = steps_from_request([{"id": "scale", "params": {"factor": 2}}], "classic")
    assert [limitation.key for limitation in limitations_for(steps)] == ["cctv.limitation.geometryChanged"]
    assert limitations_for(steps_from_request([{"id": "denoise"}], "classic")) == ()


def test_the_catalog_schema_is_json_and_has_no_free_string_params():
    schema = catalog_schema("classic")
    json.dumps(schema)
    ids = [step["id"] for step in schema]
    assert "ai_deblock" not in ids and "interpolate" not in ids
    for step in schema:
        for spec in step["filters"]:
            assert {"name", "descriptionKey", "description", "docUrl", "params"} <= set(spec)
            for param in spec["params"]:
                assert param["type"] in {"int", "float", "enum"}
    deinterlace = next(step for step in schema if step["id"] == "deinterlace")
    mode = next(param for param in deinterlace["filters"][0]["params"] if param["name"] == "mode")
    assert mode["choices"] == ["send_frame"]
    ai_deinterlace = next(step for step in catalog_schema("ai") if step["id"] == "deinterlace")
    ai_mode = next(param for param in ai_deinterlace["filters"][0]["params"] if param["name"] == "mode")
    assert ai_mode["choices"] == ["send_frame", "send_field"]


def test_the_chain_module_imports_no_other_app_module():
    code = (
        "import sys, app.services.cctv_chain; "
        "print(sorted(m for m in sys.modules if m.startswith('app.') and m != 'app.services.cctv_chain'))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "['app.services']"
