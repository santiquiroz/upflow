from __future__ import annotations

import json
import threading
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


# ---------------------------------------------------------------------------
# Review fix 1 - register() must not bypass builtin protection
# ---------------------------------------------------------------------------


def test_register_raises_value_error_for_builtin_reserved_id(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))
    impostor = make_onnx_entry(id="realesrgan-x4plus", name="Impostor")

    with pytest.raises(ValueError, match="realesrgan-x4plus"):
        registry.register(impostor)

    builtin = registry.get("realesrgan-x4plus")
    assert builtin is not None
    assert builtin.kind == ModelKind.builtin_ncnn
    assert builtin.name == "RealESRGAN x4 Plus"


def test_register_rejected_builtin_id_stays_removable_protected(tmp_path: Path) -> None:
    # Regression for the reproduced exploit: register(builtin id) used to
    # overwrite the entry as kind=onnx, which then made remove() succeed.
    registry = ModelRegistry(make_settings(tmp_path))

    with pytest.raises(ValueError):
        registry.register(make_onnx_entry(id="realesrgan-x4plus"))
    with pytest.raises(ValueError):
        registry.remove("realesrgan-x4plus")

    assert registry.get("realesrgan-x4plus") is not None


def test_register_raises_value_error_for_external_builtin_kind(tmp_path: Path) -> None:
    # Only the internal seed creates builtin entries; letting callers register
    # kind=builtin-ncnn under a new id would create an unremovable zombie.
    registry = ModelRegistry(make_settings(tmp_path))
    zombie = make_onnx_entry(id="my-custom", kind=ModelKind.builtin_ncnn)

    with pytest.raises(ValueError, match="my-custom"):
        registry.register(zombie)

    assert registry.get("my-custom") is None


# ---------------------------------------------------------------------------
# Review fix 2 - corrupt registry.json must not crash the constructor
# ---------------------------------------------------------------------------


def seed_corrupt_registry(settings: Settings, content: str) -> Path:
    registry_path = settings.models_path / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(content, encoding="utf-8")
    return registry_path


def assert_recovers_from_corrupt_file(tmp_path: Path, content: str) -> None:
    settings = make_settings(tmp_path)
    seed_corrupt_registry(settings, content)

    registry = ModelRegistry(settings)

    assert {entry.id for entry in registry.list()} == CATALOG_IDS
    backups = list(settings.models_path.glob("registry.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == content


def test_malformed_json_is_backed_up_and_builtins_reseeded(tmp_path: Path) -> None:
    assert_recovers_from_corrupt_file(tmp_path, "{not valid json!!!")


def test_valid_json_with_wrong_schema_is_backed_up_and_reseeded(tmp_path: Path) -> None:
    assert_recovers_from_corrupt_file(tmp_path, json.dumps([{"foo": "bar"}]))


def test_valid_json_with_bad_enum_value_is_backed_up_and_reseeded(tmp_path: Path) -> None:
    entry = make_onnx_entry()
    registry_path = tmp_path / "runtime" / "models" / "registry.json"
    settings = make_settings(tmp_path)
    ModelRegistry(settings).register(entry)
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    raw[0]["kind"] = "not-a-kind"
    assert_recovers_from_corrupt_file(tmp_path, json.dumps(raw))


def test_corrupt_recovery_persists_fresh_registry(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    seed_corrupt_registry(settings, "{corrupt")
    ModelRegistry(settings)

    reloaded = ModelRegistry(settings)

    assert {entry.id for entry in reloaded.list()} == CATALOG_IDS
    assert len(list(settings.models_path.glob("registry.json.corrupt-*"))) == 1


# ---------------------------------------------------------------------------
# Review fix 3 - concurrent writers must not race (Task 5 will call the
# registry from asyncio.to_thread workers; unlocked os.replace collisions
# were reproduced as PermissionError on Windows)
# ---------------------------------------------------------------------------


def test_concurrent_registers_do_not_race_and_all_persist(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    thread_count = 8
    entries_per_thread = 5
    barrier = threading.Barrier(thread_count)
    errors: list[Exception] = []

    def register_batch(worker: int) -> None:
        barrier.wait()
        try:
            for item in range(entries_per_thread):
                registry.register(
                    make_onnx_entry(id=f"custom-{worker}-{item}", name=f"Custom {worker}-{item}")
                )
        except Exception as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=register_batch, args=(worker,)) for worker in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    expected_ids = {
        f"custom-{worker}-{item}"
        for worker in range(thread_count)
        for item in range(entries_per_thread)
    }
    reloaded = ModelRegistry(settings)
    assert expected_ids <= {entry.id for entry in reloaded.list()}


def test_concurrent_register_and_remove_do_not_race(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    for item in range(20):
        registry.register(make_onnx_entry(id=f"seeded-{item}", name=f"Seeded {item}"))
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def remover() -> None:
        barrier.wait()
        try:
            for item in range(20):
                registry.remove(f"seeded-{item}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def registrar() -> None:
        barrier.wait()
        try:
            for item in range(20):
                registry.register(make_onnx_entry(id=f"fresh-{item}", name=f"Fresh {item}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=remover), threading.Thread(target=registrar)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    reloaded = ModelRegistry(settings)
    ids = {entry.id for entry in reloaded.list()}
    assert {f"fresh-{item}" for item in range(20)} <= ids
    assert not any(model_id.startswith("seeded-") for model_id in ids)


# ---------------------------------------------------------------------------
# Review minor - register() stores a defensive copy
# ---------------------------------------------------------------------------


def test_register_stores_defensive_copy_of_entry(tmp_path: Path) -> None:
    registry = ModelRegistry(make_settings(tmp_path))
    entry = make_onnx_entry()
    registry.register(entry)

    entry.status = ModelStatus.error
    entry.error = "mutated after register"

    stored = registry.get("swinir-real-sr-x4")
    assert stored is not None
    assert stored.status == ModelStatus.installed
    assert stored.error is None
