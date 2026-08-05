from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import QueueFullError
from app.services.subtitles import TranscriptSegment
from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
from app.services.storage import StorageService
from app.services.transcribe_job_manager import TranscribeJobManager

MODEL_ID = "asr--onnx-community--whisper-tiny.en"


class FakeEngine:
    def __init__(self, text: str = "hola mundo", fail: Exception | None = None) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[dict] = []

    async def run(self, **kwargs) -> list[TranscriptSegment]:
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        kwargs["progress_cb"](1, 2)
        kwargs["progress_cb"](2, 2)
        # Un segmento por palabra, con tiempos: es la forma REAL que devuelve el
        # motor desde que los subtitulos existen.
        words = self.text.split()
        return [
            TranscriptSegment(start=float(i), end=float(i + 1), text=word)
            for i, word in enumerate(words)
        ]


class SlowEngine:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, **_kwargs) -> list[TranscriptSegment]:
        self.started.set()
        await asyncio.sleep(30)
        return []


class FakeDevices:
    def validate(self, device_id: str) -> dict:
        if device_id not in ("cpu", "dml:0"):
            raise ValueError(f"Unknown device id: {device_id!r}")
        return {"id": device_id, "kind": "cpu"}


def make_settings(tmp_path: Path, **overrides) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **overrides)
    StorageService(settings).ensure_directories()
    return settings


def register_asr_model(
    registry: ModelRegistry, settings: Settings, model_id: str = MODEL_ID
) -> None:
    model_dir = settings.models_path / "asr" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry.register(
        ModelEntry(
            id=model_id,
            name="onnx-community/whisper-tiny.en",
            kind=ModelKind.asr_onnx,
            source="hf:onnx-community/whisper-tiny.en",
            size_bytes=257 * 1024 * 1024,
            file_path=f"asr/{model_id}",
        )
    )


def make_manager(
    tmp_path: Path, engine=None, **settings_overrides
) -> tuple[TranscribeJobManager, Settings, ModelRegistry]:
    settings = make_settings(tmp_path, **settings_overrides)
    registry = ModelRegistry(settings)
    manager = TranscribeJobManager(
        settings,
        engine or FakeEngine(),
        DeviceSemaphores(settings),
        registry=registry,
        devices=FakeDevices(),
    )
    register_asr_model(registry, settings)
    return manager, settings, registry


def make_audio(settings: Settings, name: str = "charla.wav") -> Path:
    path = settings.uploads_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF....WAVE")
    return path


# ---------------------------------------------------------------------------
# Crear el job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creating_a_job_queues_it(tmp_path: Path):
    manager, settings, _r = make_manager(tmp_path)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="charla.wav",
        model_id=MODEL_ID,
    )

    assert job.status is JobStatus.queued
    assert manager.get_job(job.id) is job


@pytest.mark.asyncio
async def test_an_unknown_model_is_rejected(tmp_path: Path):
    manager, settings, _r = make_manager(tmp_path)
    with pytest.raises(ValueError, match="speech recognition"):
        await manager.create_job(
            source_path=make_audio(settings),
            original_filename="a.wav",
            model_id="no-existe",
        )


@pytest.mark.asyncio
async def test_a_model_of_another_kind_is_rejected(tmp_path: Path):
    # Un upscaler o un modelo de difusion no transcriben: el kind es el gate.
    manager, settings, registry = make_manager(tmp_path)
    registry.register(
        ModelEntry(
            id="otro",
            name="x",
            kind=ModelKind.onnx,
            source="hf:x",
            size_bytes=1,
            file_path="otro.onnx",
        )
    )
    with pytest.raises(ValueError, match="speech recognition"):
        await manager.create_job(
            source_path=make_audio(settings),
            original_filename="a.wav",
            model_id="otro",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["e", "esp", "e5", "12"])
async def test_a_malformed_language_is_rejected(tmp_path: Path, language: str):
    manager, settings, _r = make_manager(tmp_path)
    with pytest.raises(ValueError, match="ISO 639-1"):
        await manager.create_job(
            source_path=make_audio(settings),
            original_filename="a.wav",
            model_id=MODEL_ID,
            language=language,
        )


@pytest.mark.asyncio
async def test_no_language_is_accepted_and_means_autodetect(tmp_path: Path):
    manager, settings, _r = make_manager(tmp_path)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )
    assert job.language is None


@pytest.mark.asyncio
async def test_auto_device_is_rejected(tmp_path: Path):
    # Igual que en generacion: el motor cachea por dispositivo, asi que "auto" no
    # tiene un dispositivo concreto que cachear.
    manager, settings, _r = make_manager(tmp_path)
    with pytest.raises(ValueError, match="auto"):
        await manager.create_job(
            source_path=make_audio(settings),
            original_filename="a.wav",
            model_id=MODEL_ID,
            device="auto",
        )


@pytest.mark.asyncio
async def test_an_unknown_device_is_rejected(tmp_path: Path):
    manager, settings, _r = make_manager(tmp_path)
    with pytest.raises(ValueError):
        await manager.create_job(
            source_path=make_audio(settings),
            original_filename="a.wav",
            model_id=MODEL_ID,
            device="dml:9",
        )


@pytest.mark.asyncio
async def test_a_full_queue_is_reported(tmp_path: Path):
    manager, settings, _r = make_manager(tmp_path, MAX_QUEUE_SIZE=1)
    await manager.create_job(
        source_path=make_audio(settings, "a.wav"),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )
    with pytest.raises(QueueFullError):
        await manager.create_job(
            source_path=make_audio(settings, "b.wav"),
            original_filename="b.wav",
            model_id=MODEL_ID,
        )


# ---------------------------------------------------------------------------
# Ejecutar el job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_finished_job_carries_the_text_and_a_txt_file(tmp_path: Path):
    engine = FakeEngine(text="el resultado transcripto")
    manager, settings, _r = make_manager(tmp_path, engine)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="charla.wav",
        model_id=MODEL_ID,
    )
    await manager._process_next()

    assert job.status is JobStatus.completed
    # El texto ES la respuesta; el .txt existe para que la descarga use el mismo
    # camino que el resto de los jobs.
    assert job.text == "el resultado transcripto"
    assert job.output_path.read_text(encoding="utf-8") == "el resultado transcripto"


@pytest.mark.asyncio
async def test_progress_is_recorded(tmp_path: Path):
    manager, settings, _r = make_manager(tmp_path)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )
    await manager._process_next()

    assert job.progress_pct == 100.0


@pytest.mark.asyncio
async def test_the_engine_receives_the_model_dir_and_the_language(tmp_path: Path):
    engine = FakeEngine()
    manager, settings, _r = make_manager(tmp_path, engine)
    await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
        language="es",
    )
    await manager._process_next()

    call = engine.calls[0]
    assert call["model_dir"].is_dir()
    assert call["request"].language == "es"


@pytest.mark.asyncio
async def test_the_uploaded_audio_is_deleted_when_the_job_ends(tmp_path: Path):
    # Es un archivo del usuario en staging: dejarlo acumularia uploads para siempre.
    manager, settings, _r = make_manager(tmp_path)
    audio = make_audio(settings)
    await manager.create_job(
        source_path=audio, original_filename="a.wav", model_id=MODEL_ID
    )
    await manager._process_next()

    assert not audio.exists()


@pytest.mark.asyncio
async def test_an_engine_failure_fails_the_job_with_its_message(tmp_path: Path):
    engine = FakeEngine(fail=RuntimeError("el modelo no carga"))
    manager, settings, _r = make_manager(tmp_path, engine)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )
    await manager._process_next()

    assert job.status is JobStatus.failed
    assert "el modelo no carga" in job.error
    assert job.text is None


@pytest.mark.asyncio
async def test_a_missing_model_folder_fails_the_job(tmp_path: Path):
    # El registro puede tener la entrada y el disco no la carpeta: borrarla a mano es
    # posible, y el job tiene que decirlo en vez de explotar raro.
    manager, settings, _r = make_manager(tmp_path)
    import shutil

    shutil.rmtree(settings.models_path / "asr" / MODEL_ID)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )
    await manager._process_next()

    assert job.status is JobStatus.failed
    assert "missing on disk" in job.error


# ---------------------------------------------------------------------------
# Cancelar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelling_a_queued_job_skips_it_without_running(tmp_path: Path):
    engine = FakeEngine()
    manager, settings, _r = make_manager(tmp_path, engine)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )

    assert manager.cancel_job(job.id) is True
    await manager._process_next()

    assert job.status is JobStatus.cancelled
    assert engine.calls == []


@pytest.mark.asyncio
async def test_cancelling_a_running_job_marks_it_cancelled(tmp_path: Path):
    engine = SlowEngine()
    manager, settings, _r = make_manager(tmp_path, engine)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )
    runner = asyncio.ensure_future(manager._process_next())
    await asyncio.wait_for(engine.started.wait(), timeout=5)

    assert manager.cancel_job(job.id) is True
    await asyncio.wait_for(runner, timeout=5)

    assert job.status is JobStatus.cancelled
    assert job.error is None


@pytest.mark.asyncio
async def test_cancelling_a_finished_job_is_refused(tmp_path: Path):
    manager, settings, _r = make_manager(tmp_path)
    job = await manager.create_job(
        source_path=make_audio(settings),
        original_filename="a.wav",
        model_id=MODEL_ID,
    )
    await manager._process_next()

    assert manager.cancel_job(job.id) is False


@pytest.mark.asyncio
async def test_cancelling_an_unknown_job_is_refused(tmp_path: Path):
    manager, _s, _r = make_manager(tmp_path)
    assert manager.cancel_job("nope") is False


def test_the_manager_is_wired_into_the_app_state():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app):
        assert isinstance(app.state.transcribe_jobs, TranscribeJobManager)
        # Y comparte el registro con el resto: un modelo instalado por el instalador
        # de ASR tiene que ser visible para el manager sin nada mas en el medio.
        assert app.state.transcribe_jobs.registry is app.state.model_registry


@pytest.mark.asyncio
async def test_a_video_output_mode_muxes_before_the_source_is_deleted(tmp_path: Path, monkeypatch):
    """El fuente se borra al terminar el job, asi que el muxeo tiene que pasar
    ANTES: despues ya no habria video al que pegarle los subtitulos."""
    manager, settings, _r = make_manager(tmp_path)
    muxed: list[Path] = []

    async def fake_mux(job):
        # El fuente todavia tiene que existir en este punto.
        assert job.source_path.exists()
        destination = settings.outputs_path / f"{job.id}.subtitled.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"video con subs")
        muxed.append(destination)
        return destination

    monkeypatch.setattr(manager, "_mux_subtitles_into_video", fake_mux)
    job = await manager.create_job(
        source_path=make_audio(settings, "clip.mp4"),
        original_filename="clip.mp4",
        model_id=MODEL_ID,
        output_mode="video",
    )
    await manager._process_next()

    assert job.status is JobStatus.completed
    assert muxed, "no se muxeo nada"
    assert job.subtitled_video_path == muxed[0]


@pytest.mark.asyncio
async def test_the_default_mode_does_not_touch_ffmpeg(tmp_path: Path, monkeypatch):
    manager, settings, _r = make_manager(tmp_path)
    called = False

    async def fake_mux(job):
        nonlocal called
        called = True
        return job.source_path

    monkeypatch.setattr(manager, "_mux_subtitles_into_video", fake_mux)
    await manager.create_job(
        source_path=make_audio(settings),
        original_filename="charla.wav",
        model_id=MODEL_ID,
    )
    await manager._process_next()

    assert called is False
