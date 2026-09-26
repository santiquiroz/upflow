from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.engines.drunet_restore import CpuFallback
from app.services.image_io import MetadataPrivacy
from app.services.photo_restore_pipeline import (
    FaceSelection,
    ModelUse,
    PostResult,
    PreResult,
    RestoreRequest,
    StepRecord,
)
from app.services.restore_models import MIGAN_MODEL_ID, RestoreModelSpec
from app.services.restore_provenance import (
    BEFORE_AFTER_DIVIDER_PX,
    InputInfo,
    ModelCatalog,
    OutputFile,
    ProvenanceFacts,
    SidecarContext,
    UpscaleInfo,
    badge_applies,
    before_after_image,
    build_sidecar,
    composite_reasons,
    default_model_catalog,
    digital_source_type,
    download_names,
    effective_bits,
    encode_jpeg,
    facts_from_records,
    facts_from_sidecar,
    output_bit_depth,
    restoration_description,
    sanitize_stem,
    sha256_file,
    uncolored_marks,
    uncolored_sidecar,
    with_badge,
    write_sidecar,
    xmp_fields,
)
from app.services.xmp_packet import (
    DIGITAL_SOURCE_COMPOSITE,
    DIGITAL_SOURCE_ENHANCED,
    build_xmp_packet,
    read_xmp_properties,
)

SHA = "a" * 64


def record(step_id: str, *, model: ModelUse | None = None, invents: bool = False, **details) -> StepRecord:
    return StepRecord(
        step_id=step_id,
        strategy="dsp" if model is None else "model",
        params={},
        seconds=0.5,
        invents_detail=invents,
        model=model,
        details=details,
    )


@pytest.mark.parametrize(
    ("facts", "reasons"),
    [
        (ProvenanceFacts(), ()),
        (ProvenanceFacts(faces_restored=1), ("faces",)),
        (ProvenanceFacts(colorized=True), ("colorize",)),
        (ProvenanceFacts(fill_coverage=0.0101), ("largeFill",)),
        (ProvenanceFacts(fill_coverage=0.01), ()),
        (ProvenanceFacts(fill_coverage=0.0001, fill_touches_faces=True), ("fillOverFace",)),
        (ProvenanceFacts(generative_upscale=True), ("generativeUpscale",)),
    ],
)
def test_composite_reasons_follow_the_rules_of_the_spec(facts: ProvenanceFacts, reasons: tuple[str, ...]) -> None:
    assert composite_reasons(facts) == reasons
    expected = DIGITAL_SOURCE_COMPOSITE if reasons else DIGITAL_SOURCE_ENHANCED
    assert digital_source_type(facts) == expected


def test_facts_come_from_the_step_records_and_the_upscale() -> None:
    records = (
        record("repair", finalCoverage=0.002, touchesFaces=True),
        record("faces", model=ModelUse("gfpgan-v1.4", "cpu", "fp32"), restored=[0, 2]),
        record("colorize", model=ModelUse("ddcolor-tiny", "cpu", "fp32")),
    )

    facts = facts_from_records(records, UpscaleInfo(mode="ai", scale=2, generative=True))

    assert facts == ProvenanceFacts(
        faces_restored=2, colorized=True, fill_coverage=0.002, fill_touches_faces=True, generative_upscale=True
    )


@pytest.mark.parametrize(
    ("upscale", "generative"),
    [
        (UpscaleInfo(), False),
        (UpscaleInfo(mode="classic", scale=2, generative=True), False),
        (UpscaleInfo(mode="ai", scale=2, generative=False), False),
        (UpscaleInfo(mode="ai", scale=4, generative=True), True),
    ],
)
def test_only_a_generative_ai_upscale_counts_as_invented(upscale: UpscaleInfo, generative: bool) -> None:
    assert facts_from_records((), upscale).generative_upscale is generative


def test_facts_from_a_sidecar_count_only_restored_faces_still_enabled() -> None:
    sidecar = {
        "faces": [
            {"index": 0, "restored": True, "enabled": True},
            {"index": 1, "restored": True, "enabled": False},
            {"index": 2, "restored": False, "enabled": True},
        ],
        "damage": {"finalCoverage": 0.03, "touchesFaces": False},
        "upscale": {"mode": "ai", "generative": True},
        "colorize": None,
    }

    assert facts_from_sidecar(sidecar) == ProvenanceFacts(faces_restored=1, fill_coverage=0.03, generative_upscale=True)


def test_the_badge_goes_only_on_composite_results_the_user_did_not_opt_out_of() -> None:
    assert badge_applies(ProvenanceFacts(faces_restored=1))
    assert not badge_applies(ProvenanceFacts(faces_restored=1), requested=False)
    assert not badge_applies(ProvenanceFacts())


def test_with_badge_marks_only_the_bottom_right_corner_and_leaves_the_input_alone() -> None:
    image = np.full((400, 600, 3), 0.8, dtype=np.float32)

    marked = with_badge(image)

    assert np.all(image == np.float32(0.8))
    changed = np.argwhere(np.abs(marked - image).max(axis=-1) > 1e-3)
    assert changed.size > 0
    assert changed[:, 0].min() > 400 * 0.8
    assert changed[:, 1].min() > 600 * 0.5


def test_with_badge_skips_a_tiny_image() -> None:
    tiny = np.full((32, 32, 3), 0.5, dtype=np.float32)

    np.testing.assert_array_equal(with_badge(tiny), tiny)


def test_before_after_puts_both_sides_at_the_same_height() -> None:
    before = np.zeros((100, 150, 3), dtype=np.float32)
    after = np.ones((200, 300, 3), dtype=np.float32)

    joined = before_after_image(before, after, badge=False)

    assert joined.dtype == np.uint8
    assert joined.shape == (200, 300 + BEFORE_AFTER_DIVIDER_PX + 300, 3)
    assert joined[:, :300].max() == 0
    assert joined[:, -300:].min() == 255


def test_before_after_carries_the_badge_on_the_restored_side_only() -> None:
    before = np.full((300, 300, 3), 0.9, dtype=np.float32)
    after = np.full((300, 300, 3), 0.9, dtype=np.float32)

    plain = before_after_image(before, after, badge=False)
    marked = before_after_image(before, after, badge=True)

    np.testing.assert_array_equal(marked[:, :300], plain[:, :300])
    assert np.any(marked[:, -300:] != plain[:, -300:])


def test_before_after_encodes_as_a_jpeg() -> None:
    pixels = before_after_image(np.zeros((64, 64, 3), np.float32), np.ones((64, 64, 3), np.float32), badge=False)

    decoded = Image.open(io.BytesIO(encode_jpeg(pixels)))

    assert decoded.format == "JPEG"
    assert decoded.size == (pixels.shape[1], pixels.shape[0])


@pytest.mark.parametrize(
    ("original", "stem"),
    [
        ("Abuela 1958.jpg", "Abuela 1958"),
        ("C:\\fotos\\boda (2).tif", "boda (2)"),
        ("../../secret/passwd.png", "passwd"),
        ("Medellín 1958.png", "Medellín 1958"),
        ('foto:*?"<>|.jpg', "foto"),
        ("   .png", "photo"),
        ("x" * 300 + ".png", "x" * 100),
    ],
)
def test_sanitize_stem_keeps_a_readable_safe_name(original: str, stem: str) -> None:
    assert sanitize_stem(original) == stem


def test_download_names_are_readable_and_follow_the_output_format() -> None:
    names = download_names("Abuela 1958.tif", "jpeg", colorized=False)

    assert names.restored == "Abuela 1958_restored.jpg"
    assert names.uncolored == "Abuela 1958_uncolored.jpg"
    assert names.before_after == "Abuela 1958_before-after.jpg"
    assert names.sidecar == "Abuela 1958_restore.json"


def test_download_names_say_colorized_when_color_was_added() -> None:
    assert download_names("abuelo.png", "png", colorized=True).restored == "abuelo_colorized.png"


def test_download_names_reject_an_unknown_format() -> None:
    with pytest.raises(ValueError):
        download_names("a.png", "gif", colorized=False)


def test_output_bit_depth_drops_to_eight_bits_after_an_ai_upscale() -> None:
    assert output_bit_depth(16, UpscaleInfo()) == 16
    assert output_bit_depth(16, UpscaleInfo(mode="classic", scale=2)) == 16
    assert output_bit_depth(16, UpscaleInfo(mode="ai", scale=2)) == 8


def test_effective_bits_are_about_eleven_when_a_model_ran_in_fp16() -> None:
    fp16 = (record("denoise", model=ModelUse("drunet-color", "dml:0", "fp16")),)
    fp32 = (record("denoise", model=ModelUse("drunet-color", "cpu", "fp32")),)

    assert effective_bits(16, fp16) == 11
    assert effective_bits(16, fp32) == 16
    assert effective_bits(8, fp16) == 8


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"upflow" * 300_000)

    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def fake_spec(model_id: str) -> RestoreModelSpec:
    return RestoreModelSpec(
        id=model_id,
        name="DRUNet",
        bundle="core",
        filename=f"{model_id}.onnx",
        license_spdx="MIT",
        license_url="https://example.com/LICENSE",
        copyright="Copyright (c) 2022 Kai Zhang",
        attribution="DPIR / KAIR",
        data_lineage="D1a",
        commercial_use="yes",
        source_url="https://example.com/drunet",
        source_revision="abc",
        source_sha256=SHA,
        modifications=("exported to ONNX",),
        tile_min=128,
    )


def sidecar_inputs(tmp_path: Path) -> tuple[PreResult, PostResult, SidecarContext, ModelCatalog]:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    request = RestoreRequest(
        image=image,
        steps=("repair", "denoise", "faces"),
        tone_kind="mono",
        device="dml:0",
        faces=(FaceSelection(0, ((1.0, 2.0),) * 5, 0.6, box=(1.0, 2.0, 20.0, 25.0), score=0.99, eye_px=40.0),),
    )
    pre = PreResult(
        image,
        (
            record("repair", model=ModelUse(MIGAN_MODEL_ID, "dml:0", "fp32"), invents=True, finalCoverage=0.004),
            StepRecord(
                "denoise",
                "model",
                {"strength": 0.3},
                1.0,
                False,
                model=ModelUse("drunet-color", "dml:0", "fp16", 512),
                cpu_fallbacks=(CpuFallback("drunet-color", "tdrBudget"),),
            ),
        ),
        request,
    )
    faces = record("faces", model=ModelUse("gfpgan-v1.4", "dml:0", "fp16"), invents=True, restored=[0])
    post = PostResult(image, (faces,), 1.0)
    source = tmp_path / "abuela.tif"
    source.write_bytes(b"original bytes")
    output = tmp_path / "job.png"
    output.write_bytes(b"restored bytes")
    context = SidecarContext(
        input=InputInfo("abuela.tif", sha256_file(source), source.stat().st_size, 16, True, 6),
        outputs=(OutputFile("restored", output),),
        output_bit_depth=16,
        app_version="0.99.0",
        commit="deadbeef",
        geometry={"rotate90": 1, "crop": None, "angle": 0.0},
        privacy=MetadataPrivacy(gps_removed=True, date_time_original_moved=True),
        photo_date="1958",
        environment={"os": "Windows", "providers": ["CPUExecutionProvider"]},
    )
    catalog = ModelCatalog({"drunet-color": fake_spec("drunet-color")}, {("drunet-color", "fp16"): SHA})
    return pre, post, context, catalog


def test_a_classic_fill_on_a_mask_from_the_ai_detector_counts_as_ai_applied(tmp_path: Path) -> None:
    pre, post, context, catalog = sidecar_inputs(tmp_path)
    detector = ModelUse("bopbtl-scratch", "cpu", "fp32")
    telea = StepRecord("repair", "dsp", {}, 0.5, False, aux_models=(detector,), details={"finalCoverage": 0.01})
    pre = PreResult(pre.image, (telea,), pre.request)
    post = PostResult(post.image, (), 1.0)

    sidecar = build_sidecar(pre, post, context, catalog)

    assert sidecar["aiApplied"] is True
    assert sidecar["steps"][0]["auxiliaryModels"][0]["id"] == "bopbtl-scratch"


def test_a_dsp_only_restoration_is_not_ai_applied(tmp_path: Path) -> None:
    pre, post, context, catalog = sidecar_inputs(tmp_path)
    pre = PreResult(pre.image, (record("tone"),), pre.request)
    post = PostResult(post.image, (), 1.0)

    assert build_sidecar(pre, post, context, catalog)["aiApplied"] is False


def test_the_sidecar_records_hashes_licenses_and_the_digital_source_type(tmp_path: Path) -> None:
    pre, post, context, catalog = sidecar_inputs(tmp_path)

    sidecar = build_sidecar(pre, post, context, catalog)

    assert sidecar["schemaVersion"] == 1
    assert sidecar["upflow"] == {"version": "0.99.0", "commit": "deadbeef"}
    assert sidecar["input"]["sha256"] == hashlib.sha256(b"original bytes").hexdigest()
    output_sha = hashlib.sha256(b"restored bytes").hexdigest()
    assert sidecar["outputs"] == [{"role": "restored", "file": "job.png", "sha256": output_sha}]
    denoise = sidecar["steps"][1]
    assert denoise["model"]["sha256"] == SHA
    assert denoise["model"]["license"]["spdx"] == "MIT"
    assert denoise["model"]["license"]["dataLineage"] == "D1a"
    assert sidecar["steps"][2]["model"]["license"] is None
    assert sidecar["effectiveBits"] == 11
    assert sidecar["aiApplied"] is True
    assert sidecar["inventsDetail"] is True
    assert sidecar["digitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert sidecar["compositeReasons"] == ["faces"]
    assert sidecar["privacy"] == {
        "gpsRemoved": True,
        "dateTimeOriginalMovedToDigitized": True,
        "approximatePhotoDate": "1958",
    }
    assert sidecar["cpuFallback"] == [{"model": "drunet-color", "reason": "tdrBudget"}]
    assert sidecar["faces"][0]["restored"] is True
    assert sidecar["damage"]["finalCoverage"] == 0.004
    json.dumps(sidecar)


def test_the_sidecar_carries_the_migan_license_from_the_default_catalog() -> None:
    catalog = default_model_catalog()

    license_info = catalog.license_of(MIGAN_MODEL_ID)

    assert license_info is not None and license_info["spdx"] == "MIT"
    assert catalog.sha256_of(MIGAN_MODEL_ID, "fp32") is not None


def test_the_sidecar_hashes_the_model_file_that_was_loaded_not_the_catalog(tmp_path: Path) -> None:
    pre, post, context, _ = sidecar_inputs(tmp_path)
    loaded = tmp_path / "drunet-color-fp16.onnx"
    loaded.write_bytes(b"another revision")
    catalog = ModelCatalog(
        {"drunet-color": fake_spec("drunet-color")},
        {("drunet-color", "fp16"): SHA},
        lambda model_id, precision: loaded if model_id == "drunet-color" else None,
    )

    denoise = build_sidecar(pre, post, context, catalog)["steps"][1]

    assert denoise["model"]["sha256"] == hashlib.sha256(b"another revision").hexdigest()
    assert denoise["model"]["expectedSha256"] == SHA


def test_a_replaced_model_file_of_the_same_size_is_hashed_again(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"AAAA")
    catalog = ModelCatalog({}, {}, lambda model_id, precision: model)
    first = catalog.sha256_of("m", "fp32")

    model.write_bytes(b"BBBB")
    os.utime(model, ns=(model.stat().st_atime_ns, model.stat().st_mtime_ns + 1_000_000_000))

    assert catalog.sha256_of("m", "fp32") == hashlib.sha256(b"BBBB").hexdigest() != first


def test_a_model_file_that_cannot_be_found_has_no_hash(tmp_path: Path) -> None:
    catalog = ModelCatalog({}, {("m", "fp32"): SHA}, lambda model_id, precision: None)

    assert catalog.sha256_of("m", "fp32") is None
    assert catalog.expected_sha256_of("m", "fp32") == SHA


def colorized(sidecar: dict, *, faces: bool) -> dict:
    steps = [*sidecar["steps"], {"id": "colorize"}]
    return {**sidecar, "steps": steps, "colorize": {"strength": 1.0}, "faces": sidecar["faces"] if faces else []}


def test_the_uncolored_copy_does_not_declare_the_colorization(tmp_path: Path) -> None:
    pre, post, context, catalog = sidecar_inputs(tmp_path)
    sidecar = colorized(build_sidecar(pre, post, context, catalog), faces=False)

    view = uncolored_sidecar(sidecar)
    xmp, badge = uncolored_marks(sidecar, True, None)

    assert [step["id"] for step in view["steps"]] == ["repair", "denoise", "faces"]
    assert view["colorize"] is None and view["compositeReasons"] == []
    properties = read_xmp_properties(xmp)
    assert properties["Iptc4xmpExt:DigitalSourceType"] == DIGITAL_SOURCE_ENHANCED
    assert "colorization" not in xmp.lower()
    assert badge is False


def test_an_uncolored_copy_with_regenerated_faces_keeps_the_badge(tmp_path: Path) -> None:
    pre, post, context, catalog = sidecar_inputs(tmp_path)
    sidecar = colorized(build_sidecar(pre, post, context, catalog), faces=True)

    xmp, badge = uncolored_marks(sidecar, True, None)

    assert read_xmp_properties(xmp)["Iptc4xmpExt:DigitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert badge is True
    assert uncolored_marks(sidecar, False, None)[1] is False


def test_the_xmp_fields_describe_the_steps_and_round_trip(tmp_path: Path) -> None:
    pre, post, context, catalog = sidecar_inputs(tmp_path)
    sidecar = build_sidecar(pre, post, context, catalog)

    fields = xmp_fields(sidecar, photo_date="1958-06")
    properties = read_xmp_properties(build_xmp_packet(fields))

    assert properties["Iptc4xmpExt:DigitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert properties["xmp:CreatorTool"] == "Upflow 0.99.0"
    assert properties["photoshop:DateCreated"] == "1958-06"
    assert properties["dc:description"] == restoration_description(sidecar)
    assert "damage repair" in properties["dc:description"]


def test_the_description_names_the_upscale_mode() -> None:
    sidecar = {"steps": [{"id": "tone"}], "upscale": {"mode": "ai"}}

    assert restoration_description(sidecar) == "Restored with Upflow: color and tone, AI upscaling."


def test_write_sidecar_writes_json_atomically(tmp_path: Path) -> None:
    path = tmp_path / "job.restore.json"
    write_sidecar(path, {"schemaVersion": 1, "input": {"name": "Medellín.jpg"}})

    assert json.loads(path.read_text(encoding="utf-8"))["input"]["name"] == "Medellín.jpg"
    assert [p.name for p in tmp_path.iterdir()] == ["job.restore.json"]
