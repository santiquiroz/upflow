from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import MODEL_CATALOG, Settings
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

# ---------------------------------------------------------------------------
# SP1 Task 2 - model_registry: JSON-backed catalog of installed models.
#
# Builtins (ncnn Real-ESRGAN/RIFE) always seed automatically from the
# existing app.config.MODEL_CATALOG and can never be removed. Custom (onnx)
# entries are registered/removed through the registry API and persisted
# atomically (write-temp-then-replace) to registry.json under
# settings.models_path.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_onnx_entry(**overrides: object) -> ModelEntry:
    defaults: dict[str, object] = {
        "id": "swinir-real-sr-x4",
        "name": "SwinIR Real SR x4",
        "kind": ModelKind.onnx,
        "source": "https://huggingface.co/example/swinir-real-sr-x4",
        "size_bytes": 12_345,
        "scale": 4,
        "arch": "swinir",
        "file_path": "onnx/swinir-real-sr-x4.onnx",
        "status": ModelStatus.installed,
    }
    defaults.update(overrides)
    return ModelEntry(**defaults)


CATALOG_IDS = {option["key"] for option in MODEL_CATALOG}


# ---------------------------------------------------------------------------
# Builtin seeding
# ---------------------------------------------------------------------------


def test_seeds_builtin_entries_from_model_catalog_on_first_use(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))

    ids = {entry.id for entry in registry.list()}

    assert ids == CATALOG_IDS


def test_seeded_builtins_are_marked_builtin_ncnn_and_installed(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))

    for entry in registry.list():
        assert entry.kind == ModelKind.builtin_ncnn
        assert entry.status == ModelStatus.installed
        assert entry.error is None


def test_seeding_persists_registry_json_under_models_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    ModelRegistry(settings)

    registry_path = settings.models_path / "registry.json"
    assert registry_path.exists()
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    assert {item["id"] for item in raw} == CATALOG_IDS


def test_seeding_is_idempotent_across_registry_instances(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ModelRegistry(settings)

    second = ModelRegistry(settings)

    assert len(second.list()) == len(MODEL_CATALOG)


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


def test_get_returns_builtin_entry_by_id(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))

    entry = registry.get("realesrgan-x4plus")

    assert entry is not None
    assert entry.id == "realesrgan-x4plus"
    assert entry.kind == ModelKind.builtin_ncnn


def test_get_returns_none_for_unknown_id(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))

    assert registry.get("does-not-exist") is None


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


def test_register_adds_custom_onnx_entry(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))
    entry = make_onnx_entry()

    registry.register(entry)

    assert registry.get(entry.id) == entry
    assert len(registry.list()) == len(MODEL_CATALOG) + 1


def test_register_persists_custom_entry_across_instances(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())

    reloaded = ModelRegistry(settings)
    reloaded_entry = reloaded.get("swinir-real-sr-x4")

    assert reloaded_entry is not None
    assert reloaded_entry.name == "SwinIR Real SR x4"
    assert reloaded_entry.kind == ModelKind.onnx
    assert reloaded_entry.scale == 4


def test_register_overwrites_entry_with_same_id(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))
    registry.register(make_onnx_entry(status=ModelStatus.converting))

    registry.register(make_onnx_entry(status=ModelStatus.installed))

    updated = registry.get("swinir-real-sr-x4")
    assert updated is not None
    assert updated.status == ModelStatus.installed
    assert len(registry.list()) == len(MODEL_CATALOG) + 1


def test_register_stores_error_entry_for_failed_conversion(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))
    entry = make_onnx_entry(status=ModelStatus.error, error="conversion failed: bad shape")

    registry.register(entry)

    stored = registry.get(entry.id)
    assert stored is not None
    assert stored.status == ModelStatus.error
    assert stored.error == "conversion failed: bad shape"


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------


def test_remove_raises_value_error_for_builtin(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))

    with pytest.raises(ValueError, match="realesrgan-x4plus"):
        registry.remove("realesrgan-x4plus")

    assert registry.get("realesrgan-x4plus") is not None


def test_remove_raises_value_error_for_unknown_id(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))

    with pytest.raises(ValueError, match="does-not-exist"):
        registry.remove("does-not-exist")


def test_remove_deletes_custom_entry(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))
    registry.register(make_onnx_entry())

    registry.remove("swinir-real-sr-x4")

    assert registry.get("swinir-real-sr-x4") is None
    assert len(registry.list()) == len(MODEL_CATALOG)


def test_remove_persists_deletion_across_instances(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    registry.remove("swinir-real-sr-x4")

    reloaded = ModelRegistry(settings)

    assert reloaded.get("swinir-real-sr-x4") is None


# ---------------------------------------------------------------------------
# Atomic persistence (write-temp-then-replace)
# ---------------------------------------------------------------------------


def test_persist_writes_via_temp_file_and_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry(make_settings(tmp_path))

    replace_sources: list[Path] = []
    original_replace = Path.replace

    def spy_replace(self: Path, target: object) -> Path:
        replace_sources.append(self)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", spy_replace)

    registry.register(make_onnx_entry())

    assert len(replace_sources) == 1
    assert replace_sources[0].suffix == ".tmp"
    assert replace_sources[0] != registry._registry_path


def test_persist_leaves_no_leftover_tmp_files(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)

    registry.register(make_onnx_entry())
    registry.remove("swinir-real-sr-x4")

    assert list(settings.models_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Settings: MODELS_DIR / models_path
# ---------------------------------------------------------------------------


def test_settings_models_path_defaults_to_models_under_runtime(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.models_path == settings.runtime_path / "models"
