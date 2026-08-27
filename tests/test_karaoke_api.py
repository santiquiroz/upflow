from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import (
    cancel_karaoke_job,
    create_karaoke_job,
    download_karaoke_job,
    download_karaoke_instrumental,
    get_karaoke_job,
    karaoke_job_to_response,
    render_karaoke_job,
    update_karaoke_lyrics,
)
from app.config import Settings
from app.models import JobStatus
from app.schemas import KaraokeLyricEdit, KaraokeLyricsUpdateRequest
from app.services.device_semaphores import DeviceSemaphores
from app.services.karaoke_job_manager import KaraokeJobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
from app.services.storage import StorageService
from app.services.subtitles import TranscriptSegment

MODEL_ID = "asr--onnx-community--whisper-tiny"


class FakeEngine:
    async def run(self, **kwargs) -> list[TranscriptSegment]:
        kwargs["progress_cb"](1, 1)
        return [
            TranscriptSegment(start=0.0, end=1.0, text="texto"),
            TranscriptSegment(start=1.0, end=2.0, text="cantado"),
        ]


class FakeSeparator:
    async def run(self, decoded, stem_wavs, device, model_id=None, on_chunk=None):
        for salida in stem_wavs:
            Path(salida).write_bytes(b"wav")


class FakeTranslation:
    def available(self, pair) -> bool:
        return True

    def translate(self, texts, pair):
        return [t.upper() for t in texts]


async def fake_run_process(command, que_fallo):
    Path(command[-1]).write_bytes(b"out")


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def make_manager(tmp_path: Path) -> tuple[KaraokeJobManager, Settings]:
    from app.services.engines.separation_models import SEPARATION_MODELS

    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    model_dir = settings.models_path / "asr" / MODEL_ID
    model_dir.mkdir(parents=True, exist_ok=True)
    registry.register(
        ModelEntry(
            id=MODEL_ID,
            name="onnx-community/whisper-tiny",
            kind=ModelKind.asr_onnx,
            source="hf:onnx-community/whisper-tiny",
            size_bytes=1,
            file_path=f"asr/{MODEL_ID}",
        )
    )
    separator = FakeSeparator()
    manager = KaraokeJobManager(
        settings,
        FakeEngine(),
        DeviceSemaphores(settings),
        registry=registry,
        separators={spec.architecture: separator for spec in SEPARATION_MODELS.values()},
        restorers={},
        translation=FakeTranslation(),
    )
    manager._run_process = fake_run_process
    return manager, settings


def upload(name: str = "cancion.wav") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(b"RIFF....WAVE"))


async def create(manager: KaraokeJobManager, settings: Settings, **kwargs):
    params = {
        "request": None,
        "file": upload(),
        "asr_model_id": MODEL_ID,
        "separation_model_id": None,
        "cleanup_steps": [],
        "restore_mode": None,
        "language": None,
        "romanize": False,
        "translate_to": None,
        "device": None,
        "karaoke_jobs": manager,
        "storage": StorageService(settings),
        "settings": settings,
    }
    params.update(kwargs)
    return await create_karaoke_job(**params)


@pytest.mark.asyncio
async def test_crear_acepta_el_upload_y_encola(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    assert response.status is JobStatus.queued
    assert response.status_url.endswith(response.job_id)
    job = manager.get_job(response.job_id)
    assert job is not None and job.phase == "preparing"


@pytest.mark.asyncio
async def test_parametros_del_pipeline_llegan_al_job(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(
        manager,
        settings,
        cleanup_steps=["denoise"],
        language="ja",
        romanize=True,
        translate_to="es",
    )

    job = manager.get_job(response.job_id)
    assert job.cleanup_steps == ["denoise"]
    assert job.language == "ja"
    assert job.romanize is True
    assert job.translate_to == "es"


@pytest.mark.asyncio
async def test_un_parametro_invalido_da_400(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    with pytest.raises(HTTPException) as exc:
        await create(manager, settings, separation_model_id="inventado")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_en_review_la_respuesta_trae_letra_e_instrumental(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings, language="ja", translate_to="es")
    await manager._process_next()

    job = manager.get_job(response.job_id)
    respuesta = karaoke_job_to_response(job)

    assert respuesta.phase == "review"
    assert [l.text for l in respuesta.lines] == ["texto", "cantado"]
    assert respuesta.lines[0].translation == "TEXTO"
    assert respuesta.instrumental_url.endswith("/instrumental")


@pytest.mark.asyncio
async def test_editar_la_letra_por_endpoint(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)
    await manager._process_next()

    respuesta = await update_karaoke_lyrics(
        response.job_id,
        KaraokeLyricsUpdateRequest(lines=[KaraokeLyricEdit(index=0, text="otro")]),
        karaoke_jobs=manager,
        request=None,
    )

    assert respuesta.lines[0].text == "otro"


@pytest.mark.asyncio
async def test_editar_antes_de_review_da_409(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    with pytest.raises(HTTPException) as exc:
        await update_karaoke_lyrics(
            response.job_id,
            KaraokeLyricsUpdateRequest(lines=[KaraokeLyricEdit(index=0, text="x")]),
            karaoke_jobs=manager,
            request=None,
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_render_con_estilo_invalido_da_409(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)
    await manager._process_next()

    with pytest.raises(HTTPException) as exc:
        await render_karaoke_job(
            response.job_id,
            request=None,
            background_kind="generated",
            subtitle_size="giant",
            subtitle_position="bottom",
            subtitle_color="#FFFF00",
            subtitle_highlight_color="#FFFFFF",
            background=None,
            karaoke_jobs=manager,
            storage=StorageService(settings),
            settings=settings,
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_instrumental_antes_de_review_da_409(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    with pytest.raises(HTTPException) as exc:
        await download_karaoke_instrumental(
            response.job_id, karaoke_jobs=manager, request=None
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_descargar_antes_de_completar_da_409(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    with pytest.raises(HTTPException) as exc:
        await download_karaoke_job(response.job_id, karaoke_jobs=manager, request=None)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_cancelar_devuelve_el_estado_del_job(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    respuesta = await cancel_karaoke_job(
        response.job_id, karaoke_jobs=manager, request=None
    )

    assert respuesta.status is JobStatus.cancelled


@pytest.mark.asyncio
async def test_job_inexistente_da_404(tmp_path: Path):
    manager, _settings = make_manager(tmp_path)
    with pytest.raises(HTTPException) as exc:
        await get_karaoke_job("nope", karaoke_jobs=manager, request=None)
    assert exc.value.status_code == 404
