from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.engines.sdcpp_models import (
    SDCPP_MODEL_PREFIX,
    list_sdcpp_models,
    resolve_sdcpp_model,
    sdcpp_model_id,
)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = {
        "RUNTIME_DIR": str(tmp_path / "runtime"),
        "SDCPP_BINARY": str(tmp_path / "sd.exe"),
        "SDCPP_MODELS_DIR": str(tmp_path / "models"),
        "SDCPP_MODEL": str(tmp_path / "legacy.safetensors"),
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type]


def test_no_models_when_the_folder_is_empty(tmp_path: Path) -> None:
    assert list_sdcpp_models(make_settings(tmp_path)) == []


def test_every_checkpoint_in_the_folder_is_a_model(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "dreamshaper_8.safetensors").write_bytes(b"x")
    (models_dir / "juggernaut-xl.safetensors").write_bytes(b"x")
    (models_dir / "leeme.txt").write_text("no soy un modelo")

    models = list_sdcpp_models(make_settings(tmp_path))

    assert [m.id for m in models] == [
        f"{SDCPP_MODEL_PREFIX}dreamshaper_8",
        f"{SDCPP_MODEL_PREFIX}juggernaut-xl",
    ]
    assert models[0].name == "dreamshaper_8"


def test_gguf_checkpoints_count_too(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "sd15-q4.gguf").write_bytes(b"x")

    assert [m.id for m in list_sdcpp_models(make_settings(tmp_path))] == [
        f"{SDCPP_MODEL_PREFIX}sd15-q4"
    ]


def test_the_legacy_single_model_setting_still_shows_up(tmp_path: Path) -> None:
    # Instalaciones anteriores apuntaban SDCPP_MODEL a un archivo suelto: no se
    # las deja sin su modelo por haber agregado la carpeta.
    legacy = tmp_path / "legacy.safetensors"
    legacy.write_bytes(b"x")

    models = list_sdcpp_models(make_settings(tmp_path))

    assert [m.id for m in models] == [f"{SDCPP_MODEL_PREFIX}legacy"]


def test_the_legacy_model_is_not_listed_twice(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    shared = models_dir / "dreamshaper_8.safetensors"
    shared.write_bytes(b"x")

    models = list_sdcpp_models(make_settings(tmp_path, SDCPP_MODEL=str(shared)))

    assert len(models) == 1


def test_resolve_returns_the_path_of_a_listed_model(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    checkpoint = models_dir / "dreamshaper_8.safetensors"
    checkpoint.write_bytes(b"x")
    settings = make_settings(tmp_path)

    resolved = resolve_sdcpp_model(f"{SDCPP_MODEL_PREFIX}dreamshaper_8", settings)

    assert resolved == checkpoint


def test_resolve_rejects_an_unknown_model(tmp_path: Path) -> None:
    assert resolve_sdcpp_model(f"{SDCPP_MODEL_PREFIX}no-existe", make_settings(tmp_path)) is None


def test_resolve_ignores_ids_of_other_engines(tmp_path: Path) -> None:
    assert resolve_sdcpp_model("gen--algun--modelo", make_settings(tmp_path)) is None


@pytest.mark.parametrize("evil", ["../secreto", "..\\secreto", "sub/dir", "a/../../b"])
def test_resolve_never_escapes_the_models_folder(tmp_path: Path, evil: str) -> None:
    # El id viaja en la API: sin esto, un id armado a mano leería archivos de
    # cualquier parte del disco.
    assert resolve_sdcpp_model(f"{SDCPP_MODEL_PREFIX}{evil}", make_settings(tmp_path)) is None


def test_the_model_id_is_built_from_the_file_name(tmp_path: Path) -> None:
    assert sdcpp_model_id(Path("x/dreamshaper_8.safetensors")) == f"{SDCPP_MODEL_PREFIX}dreamshaper_8"
