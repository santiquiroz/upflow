from __future__ import annotations

import ast
import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services import restore_recompose
from app.services.engines.colorize import ColorizeOptions, apply_ab, save_ab_artifact
from app.services.engines.face_restore import (
    BEFORE_FACES_NAME,
    FACES_MANIFEST_NAME,
    FacePatch,
    FaceRestoreResult,
    RestoredFace,
    blend_face,
    blended_patch,
    paste_faces,
    read_unit_png16,
    save_recompose_artifacts,
)
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import RestoreCancelled
from app.services.face_geometry import TEMPLATE_FFHQ_512, align_face, align_matrix
from app.services.restore_provenance import OutputFile, recomposed_sidecar
from app.services.restore_recompose import FaceChoice, RecomposeUnavailable, recompose
from app.services.xmp_packet import DIGITAL_SOURCE_COMPOSITE, DIGITAL_SOURCE_ENHANCED

FACE_SCALES = (0.18, 0.14)
FACE_OFFSETS = ((20.0, 30.0), (150.0, 120.0))
SIZE = 256


def smooth_photo(seed: int = 0, size: int = SIZE) -> np.ndarray:
    coarse = np.random.default_rng(seed).random((9, 9, 3), dtype=np.float32)
    return np.clip(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def face_landmarks(index: int) -> np.ndarray:
    return TEMPLATE_FFHQ_512 * FACE_SCALES[index] + np.asarray(FACE_OFFSETS[index])


def restored_face(source: np.ndarray, index: int, blend: float) -> RestoredFace:
    matrix = align_matrix(face_landmarks(index))
    aligned = align_face(source, matrix)
    restored = np.clip(aligned + np.float32(0.25), 0.0, 1.0).astype(np.float32)
    return RestoredFace(index, matrix, blend, aligned, restored)


def saved_result(directory: Path, *, faces: int = 2, scale: int = 1, blend: float = 0.6) -> FaceRestoreResult:
    source = smooth_photo()
    upscaled = cv2.resize(source, (SIZE * scale, SIZE * scale), interpolation=cv2.INTER_CUBIC)
    base = source if scale == 1 else np.clip(upscaled, 0.0, 1.0)
    restored = tuple(restored_face(source, index, blend) for index in range(faces))
    image = paste_faces(base, [blended_patch(face) for face in restored], float(scale))
    result = FaceRestoreResult(image=image, faces=restored, scale=float(scale))
    saved = save_recompose_artifacts(directory, base, result, free_bytes=lambda path: None)
    assert saved.available
    return result


def quantized_patch(directory: Path, index: int, blend: float) -> FacePatch:
    manifest = json.loads((directory / FACES_MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = next(face for face in manifest["faces"] if face["index"] == index)
    aligned = read_unit_png16(directory / entry["aligned"])
    restored = read_unit_png16(directory / entry["restored"])
    return FacePatch(blend_face(restored, aligned, blend), np.asarray(entry["matrix"]))


@pytest.fixture
def no_inference(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def forbidden(name: str):
        def spy(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"recompose must not call PhotoRestoreEngine.{name}")

        return spy

    for name in ("session", "ready_infer", "tile_infer", "begin_phase"):
        monkeypatch.setattr(PhotoRestoreEngine, name, forbidden(name))
    return calls


def test_recompose_with_the_saved_blends_reproduces_the_result(tmp_path: Path, no_inference: list[str]) -> None:
    original = saved_result(tmp_path)

    result = recompose(tmp_path, {})

    np.testing.assert_allclose(result.image, original.image, atol=2e-4)
    assert result.face_states == {0: (True, 0.6), 1: (True, 0.6)}
    assert no_inference == []


def test_recompose_with_a_new_blend_pastes_the_new_mix_without_models(tmp_path: Path, no_inference: list[str]) -> None:
    original = saved_result(tmp_path)

    result = recompose(tmp_path, {0: FaceChoice(enabled=True, blend=0.2)})

    before = read_unit_png16(tmp_path / BEFORE_FACES_NAME)
    expected = paste_faces(before, [quantized_patch(tmp_path, 0, 0.2), quantized_patch(tmp_path, 1, 0.6)], 1.0)
    np.testing.assert_array_equal(result.image, expected)
    assert np.abs(result.image - original.image).max() > 0.02
    assert no_inference == []


def test_recompose_turning_a_face_off_gives_back_its_previous_pixels_exactly(tmp_path: Path) -> None:
    saved_result(tmp_path, faces=1)

    result = recompose(tmp_path, {0: FaceChoice(enabled=False, blend=0.6)})

    np.testing.assert_array_equal(result.image, read_unit_png16(tmp_path / BEFORE_FACES_NAME))
    assert result.face_states == {0: (False, 0.6)}


def test_recompose_turning_one_face_off_keeps_the_other(tmp_path: Path) -> None:
    saved_result(tmp_path, faces=2)

    result = recompose(tmp_path, {1: FaceChoice(enabled=False, blend=0.6)})

    before = read_unit_png16(tmp_path / BEFORE_FACES_NAME)
    expected = paste_faces(before, [quantized_patch(tmp_path, 0, 0.6)], 1.0)
    np.testing.assert_array_equal(result.image, expected)


def test_recompose_works_at_the_output_scale(tmp_path: Path) -> None:
    original = saved_result(tmp_path, faces=1, scale=2)

    result = recompose(tmp_path, {})

    assert result.image.shape == (SIZE * 2, SIZE * 2, 3)
    np.testing.assert_allclose(result.image, original.image, atol=2e-4)


def test_recompose_rejects_a_face_that_was_not_restored(tmp_path: Path) -> None:
    saved_result(tmp_path, faces=1)

    with pytest.raises(ValueError, match="not restored"):
        recompose(tmp_path, {3: FaceChoice(enabled=True, blend=0.5)})


def test_face_choice_rejects_a_blend_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        FaceChoice(enabled=True, blend=1.5)


def test_recompose_reuses_the_saved_ab_without_colorizing_again(tmp_path: Path, no_inference: list[str]) -> None:
    saved_result(tmp_path, faces=1)
    ab_512 = np.zeros((512, 512, 2), dtype=np.float32)
    ab_512[..., 0], ab_512[..., 1] = 8.0, 22.0
    assert save_ab_artifact(tmp_path, ab_512, free_bytes=lambda path: None).available
    options = ColorizeOptions(strength=0.8, saturation=1.2)

    result = recompose(tmp_path, {0: FaceChoice(enabled=True, blend=0.3)}, colorize_options=options)

    assert result.ab_reused
    np.testing.assert_array_equal(result.image, apply_ab(result.uncolored, ab_512, options))
    before = read_unit_png16(tmp_path / BEFORE_FACES_NAME)
    np.testing.assert_array_equal(result.uncolored, paste_faces(before, [quantized_patch(tmp_path, 0, 0.3)], 1.0))
    assert no_inference == []


def test_recompose_without_the_saved_ab_cannot_colorize(tmp_path: Path) -> None:
    saved_result(tmp_path, faces=1)

    with pytest.raises(RecomposeUnavailable):
        recompose(tmp_path, {}, colorize_options=ColorizeOptions())


def test_recompose_without_saved_faces_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(RecomposeUnavailable) as raised:
        recompose(tmp_path, {})

    assert raised.value.code == "restore.error.recomposeUnavailable"


def test_recompose_refuses_manifest_names_that_leave_the_job_directory(tmp_path: Path) -> None:
    saved_result(tmp_path, faces=1)
    manifest_path = tmp_path / FACES_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["faces"][0]["restored"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RecomposeUnavailable, match="Invalid artifact name"):
        recompose(tmp_path, {})


def test_recompose_stops_when_cancelled(tmp_path: Path) -> None:
    saved_result(tmp_path, faces=1)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(RestoreCancelled):
        recompose(tmp_path, {}, cancel_event=cancel)


def test_recompose_module_does_not_import_the_inference_engine() -> None:
    tree = ast.parse(Path(restore_recompose.__file__).read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert "app.services.engines.photo_restore_engine" not in modules
    assert "onnxruntime" not in modules


def sidecar_with_faces(*, colorized: bool = False) -> dict[str, object]:
    return {
        "faces": [
            {"index": 0, "enabled": True, "restored": True, "blend": 0.6, "recomposedAt": None},
            {"index": 1, "enabled": False, "restored": False, "blend": 0.4, "recomposedAt": None},
        ],
        "damage": {"finalCoverage": 0.001, "touchesFaces": False},
        "upscale": {"mode": "none", "generative": False},
        "colorize": {"model": "ddcolor-tiny"} if colorized else None,
        "outputs": [],
        "digitalSourceType": DIGITAL_SOURCE_COMPOSITE,
    }


def test_turning_every_face_off_recalculates_the_digital_source_type(tmp_path: Path) -> None:
    output = tmp_path / "result.png"
    output.write_bytes(b"new pixels")

    outputs = [OutputFile("restored", output)]

    updated = recomposed_sidecar(sidecar_with_faces(), {0: (False, 0.6)}, outputs, "2026-09-25T10:00:00Z")

    assert updated["digitalSourceType"] == DIGITAL_SOURCE_ENHANCED
    assert updated["faces"][0] == {
        "index": 0,
        "enabled": False,
        "restored": True,
        "blend": 0.6,
        "recomposedAt": "2026-09-25T10:00:00Z",
    }
    assert updated["outputs"][0]["file"] == "result.png"


def test_turning_faces_off_keeps_composite_when_the_photo_was_colorized(tmp_path: Path) -> None:
    updated = recomposed_sidecar(sidecar_with_faces(colorized=True), {0: (False, 0.6)}, [], "2026-09-25T10:00:00Z")

    assert updated["digitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert updated["compositeReasons"] == ["colorize"]


def test_turning_a_face_back_on_is_composite_again(tmp_path: Path) -> None:
    sidecar = recomposed_sidecar(sidecar_with_faces(), {0: (False, 0.6)}, [], "t1")

    updated = recomposed_sidecar(sidecar, {0: (True, 0.5)}, [], "t2")

    assert updated["digitalSourceType"] == DIGITAL_SOURCE_COMPOSITE
    assert updated["faces"][0]["blend"] == 0.5
