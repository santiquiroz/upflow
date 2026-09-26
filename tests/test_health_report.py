from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services import restore_models
from app.services.health_report import build_health_report, installed_restore_packs
from app.services.restore_models import RestoreArtifact, RestoreBundle


def published_bundle(name: str) -> RestoreBundle:
    artifact = RestoreArtifact(
        model_id=f"{name}-model",
        precision="fp32",
        filename=f"{name}-model.onnx",
        url=f"https://example.com/{name}-model.onnx",
        sha256="d" * 64,
        size=4,
    )
    return RestoreBundle(name, (artifact,))


def install(model_dir: Path, bundle: RestoreBundle) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for artifact in bundle.artifacts:
        (model_dir / artifact.filename).write_bytes(b"x" * artifact.size)
    (model_dir / bundle.manifest_name).write_text(json.dumps({"bundle": bundle.name}), encoding="utf-8")


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), RESTORE_MODEL_DIR=str(tmp_path / "restore"))


def test_no_restore_pack_is_reported_when_nothing_is_installed(tmp_path: Path) -> None:
    report = build_health_report(make_settings(tmp_path), None, None, None, None)

    assert report["restorePacksInstalled"] == []


def test_only_the_complete_restore_bundles_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    for name in ("core", "faces"):
        monkeypatch.setitem(restore_models.RESTORE_BUNDLES, name, published_bundle(name))
    install(settings.restore_model_dir_path, restore_models.RESTORE_BUNDLES["core"])
    install(settings.restore_model_dir_path, restore_models.RESTORE_BUNDLES["faces"])
    (settings.restore_model_dir_path / "faces-model.onnx").unlink()

    assert installed_restore_packs(settings) == ["restore-core"]
