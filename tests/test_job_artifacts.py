from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.engines.face_restore import FACES_MANIFEST_NAME, MANIFEST_SCHEMA_VERSION
from app.services.job_artifacts import (
    UnknownArtifact,
    image_artifact_path,
    image_download_name,
    media_type_for,
    restore_artifact,
    restored_download_name,
)

JOB = "0123456789abcdef0123456789abcdef"


def write_manifest(outputs: Path, faces: list[dict]) -> Path:
    directory = outputs / f"{JOB}.restore"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schemaVersion": MANIFEST_SCHEMA_VERSION, "scale": 1.0, "beforeFaces": "before.png", "faces": faces}
    (directory / FACES_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def face_entry(index: int, aligned: str | None = None, restored: str | None = None) -> dict:
    return {
        "index": index,
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "blend": 0.6,
        "aligned": aligned or f"face-{index}-aligned.png",
        "restored": restored or f"face-{index}-restored.png",
    }


@pytest.mark.parametrize(
    ("name", "file"),
    [
        ("preview", f"{JOB}.preview.jpg"),
        ("view", f"{JOB}.view.jpg"),
        ("beforeafter", f"{JOB}.beforeafter.jpg"),
        ("uncolored", f"{JOB}.uncolored.png"),
        ("sidecar", f"{JOB}.restore.json"),
    ],
)
def test_image_artifacts_map_to_the_job_outputs(tmp_path: Path, name: str, file: str) -> None:
    assert image_artifact_path(tmp_path, JOB, "png", name) == tmp_path / file


def test_uncolored_keeps_the_output_extension(tmp_path: Path) -> None:
    assert image_artifact_path(tmp_path, JOB, "jpeg", "uncolored").name == f"{JOB}.uncolored.jpg"


def test_face_artifacts_come_from_the_faces_manifest(tmp_path: Path) -> None:
    directory = write_manifest(tmp_path, [face_entry(0), face_entry(3)])

    assert image_artifact_path(tmp_path, JOB, "png", "face:3:before") == directory / "face-3-aligned.png"
    assert image_artifact_path(tmp_path, JOB, "png", "face:3:after") == directory / "face-3-restored.png"


def test_a_face_that_was_not_restored_is_unknown(tmp_path: Path) -> None:
    write_manifest(tmp_path, [face_entry(0)])

    with pytest.raises(UnknownArtifact, match="Face 2"):
        image_artifact_path(tmp_path, JOB, "png", "face:2:after")


def test_faces_without_a_manifest_are_unknown(tmp_path: Path) -> None:
    with pytest.raises(UnknownArtifact, match="no restored faces"):
        image_artifact_path(tmp_path, JOB, "png", "face:0:after")


def test_a_manifest_name_that_leaves_the_job_folder_is_refused(tmp_path: Path) -> None:
    write_manifest(tmp_path, [face_entry(0, restored="../secret.png")])

    with pytest.raises(UnknownArtifact):
        image_artifact_path(tmp_path, JOB, "png", "face:0:after")


@pytest.mark.parametrize(
    "name",
    ["../sidecar", "sidecar/../../x", "..", "", "original", "face:0:middle", "face:-1:after", "face:0:after:x", "SIDECAR"],
)
def test_names_outside_the_whitelist_are_refused(tmp_path: Path, name: str) -> None:
    write_manifest(tmp_path, [face_entry(0)])

    with pytest.raises(UnknownArtifact):
        image_artifact_path(tmp_path, JOB, "png", name)


@pytest.mark.parametrize(
    ("name", "colorized", "expected"),
    [
        ("beforeafter", False, "Grandma 1952_before-after.jpg"),
        ("sidecar", False, "Grandma 1952_restore.json"),
        ("uncolored", True, "Grandma 1952_uncolored.png"),
        ("view", False, "Grandma 1952_view.jpg"),
        ("preview", False, "Grandma 1952_preview.jpg"),
        ("face:2:after", False, "Grandma 1952_face-2-after.png"),
    ],
)
def test_download_names_use_the_original_stem(name: str, colorized: bool, expected: str) -> None:
    assert image_download_name(name, "C:/fotos/Grandma 1952.tif", "png", colorized) == expected


def test_restore_artifact_carries_name_and_media_type(tmp_path: Path) -> None:
    (tmp_path / f"{JOB}.beforeafter.jpg").write_bytes(b"jpeg")

    artifact = restore_artifact(tmp_path, JOB, "png", "beforeafter", "boda.png", {"colorize": None})

    assert artifact.path == tmp_path / f"{JOB}.beforeafter.jpg"
    assert artifact.download_name == "boda_before-after.jpg"
    assert artifact.media_type == "image/jpeg"


def test_a_whitelisted_artifact_missing_on_disk_is_unknown(tmp_path: Path) -> None:
    with pytest.raises(UnknownArtifact, match="no 'view'"):
        restore_artifact(tmp_path, JOB, "png", "view", "boda.png", {})


def test_restored_download_name_reads_the_job_summary() -> None:
    assert restored_download_name({"downloadNames": {"restored": "boda_colorized.png"}}) == "boda_colorized.png"
    assert restored_download_name({"artifacts": ["preview"]}) is None
    assert restored_download_name(None) is None


def test_media_types() -> None:
    assert media_type_for(Path("a.json")) == "application/json"
    assert media_type_for(Path("a.PNG")) == "image/png"
    assert media_type_for(Path("a.bin")) == "application/octet-stream"
