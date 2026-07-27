from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.config import Settings
from app.core.log_file import LOG_FILENAME, configure_file_logging, log_file_path


# ---------------------------------------------------------------------------
# Log a archivo, apagado por defecto y encendible en caliente. Sin esto un
# reporte de "va lento" desde otra maquina es puro adivinar: el warning que
# avisa que el upscale cayo a tiling por OOM ya existe, pero va a una consola
# que nadie guarda.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    # Settings NO tiene populate_by_name: los kwargs por nombre de campo se
    # ignoran en silencio, hay que usar los alias en mayusculas.
    return Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _detach_handlers_between_tests():
    yield
    configure_file_logging(make_settings(Path("."), ENABLE_FILE_LOGGING=False))


def test_disabled_by_default_writes_no_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.enable_file_logging is False

    assert configure_file_logging(settings) is None
    logging.getLogger("upflow.test").warning("no deberia persistirse")
    assert not log_file_path(settings).exists()


def test_enabling_writes_records_to_the_log_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_FILE_LOGGING=True)

    path = configure_file_logging(settings)

    assert path == log_file_path(settings)
    logging.getLogger("upflow.test").warning("cayo a tiling por OOM")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "cayo a tiling por OOM" in path.read_text(encoding="utf-8")


def test_the_file_lands_in_a_predictable_place_testers_can_find(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_FILE_LOGGING=True)
    path = log_file_path(settings)

    assert path.name == LOG_FILENAME
    assert path.parent.name == "logs"
    assert path.parent.parent == settings.runtime_path


def test_configuring_twice_does_not_duplicate_handlers(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_FILE_LOGGING=True)

    configure_file_logging(settings)
    configure_file_logging(settings)

    tagged = [h for h in logging.getLogger().handlers if getattr(h, "_upflow_file_handler", False)]
    assert len(tagged) == 1, "cada reconfiguracion agregaba un handler mas"


def test_turning_it_off_detaches_the_handler(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_FILE_LOGGING=True)
    configure_file_logging(settings)

    settings.enable_file_logging = False
    assert configure_file_logging(settings) is None

    tagged = [h for h in logging.getLogger().handlers if getattr(h, "_upflow_file_handler", False)]
    assert tagged == []


def test_rotation_is_bounded_so_logs_cannot_fill_the_disk(tmp_path: Path) -> None:
    # Un log sin techo en una app que ya tuvo un bug de disco lleno es
    # exactamente el pie que no queremos volver a pisar.
    settings = make_settings(tmp_path, ENABLE_FILE_LOGGING=True, LOG_FILE_MAX_MB=1, LOG_FILE_BACKUPS=2)
    configure_file_logging(settings)

    handler = next(h for h in logging.getLogger().handlers if getattr(h, "_upflow_file_handler", False))
    assert handler.maxBytes == 1 * 1024 * 1024
    assert handler.backupCount == 2


def test_a_broken_log_directory_never_takes_the_app_down(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path, ENABLE_FILE_LOGGING=True)

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disco lleno")

    monkeypatch.setattr(Path, "mkdir", boom)
    # Logging es diagnostico: si no se puede escribir, se sigue sin log.
    assert configure_file_logging(settings) is None


# ---------------------------------------------------------------------------
# Diagnostico en job.metadata: por que este job fue lento. Claves informativas
# nuevas (como streamPipeline / videoEncoder), no de progreso: el frontend no
# cambia. Sin esto, "va lentisimo en la PC de mi amigo" es adivinar.
# ---------------------------------------------------------------------------


def test_engine_reports_the_precision_it_selected(tmp_path: Path) -> None:
    from tests.test_onnx_video_upscaler import make_engine, touch_builtin_onnx

    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8-fp16.onnx")
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")

    from app.services.backend_registry import get_builtin_onnx_model

    model = get_builtin_onnx_model("realesr-animevideov3-x4")
    engine._select_model_file(model, "dml:0")
    assert engine.last_precision == "fp16"

    engine._select_model_file(model, "cpu")
    assert engine.last_precision == "fp32", "en CPU fp16 se emula: debe reportar fp32"


def test_engine_reports_fp32_when_the_fp16_export_is_missing(tmp_path: Path) -> None:
    # Fallo silencioso por diseño ("A missing fp16 sibling silently uses fp32"):
    # medido 7.26x mas lento, y hasta ahora nada lo delataba.
    from tests.test_onnx_video_upscaler import make_engine, touch_builtin_onnx

    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")

    from app.services.backend_registry import get_builtin_onnx_model

    engine._select_model_file(get_builtin_onnx_model("realesr-animevideov3-x4"), "dml:0")
    assert engine.last_precision == "fp32"


def test_engine_starts_without_tiling_flagged(tmp_path: Path) -> None:
    from tests.test_onnx_video_upscaler import make_engine

    assert make_engine(tmp_path).last_tiled is False
