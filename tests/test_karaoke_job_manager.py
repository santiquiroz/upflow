from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.separation_models import (
    DEFAULT_SEPARATION_MODEL,
    SEPARATION_MODELS,
)
from app.services.karaoke_job_manager import KaraokeJobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
from app.services.storage import StorageService
from app.services.subtitles import TranscriptSegment
from app.services.translate import parse_pair

MULTILINGUAL_ID = "asr--onnx-community--whisper-tiny"


class FakeEngine:
    def __init__(self, text: str = "hola mundo") -> None:
        self.text = text

    async def run(self, **kwargs) -> list[TranscriptSegment]:
        kwargs["progress_cb"](1, 1)
        return [
            TranscriptSegment(start=float(i), end=float(i + 1), text=palabra)
            for i, palabra in enumerate(self.text.split())
        ]


class FakeSeparator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, decoded, stem_wavs, device, model_id=None, on_chunk=None):
        self.calls.append(model_id)
        for salida in stem_wavs:
            Path(salida).write_bytes(b"wav")


class FakeTranslation:
    def __init__(self, installed: bool = True) -> None:
        self.installed = installed
        self.received: list[list[str]] = []

    def available(self, pair) -> bool:
        return self.installed

    def translate(self, texts, pair):
        self.received.append(list(texts))
        return [t.upper() for t in texts]


class FakeDevices:
    def validate(self, device_id: str) -> dict:
        return {"id": device_id}


async def fake_run_process(command, que_fallo):
    Path(command[-1]).write_bytes(b"out")


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def register_model(registry: ModelRegistry, settings: Settings, model_id: str) -> None:
    nombre = f"onnx-community/{model_id.rsplit('--', 1)[-1]}"
    model_dir = settings.models_path / "asr" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry.register(
        ModelEntry(
            id=model_id,
            name=nombre,
            kind=ModelKind.asr_onnx,
            source=f"hf:{nombre}",
            size_bytes=1024,
            file_path=f"asr/{model_id}",
        )
    )


def all_architectures() -> dict[str, FakeSeparator]:
    separator = FakeSeparator()
    return {spec.architecture: separator for spec in SEPARATION_MODELS.values()}


def make_manager(tmp_path: Path, *, translation=None, engine=None):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    manager = KaraokeJobManager(
        settings,
        engine or FakeEngine(),
        DeviceSemaphores(settings),
        registry=registry,
        separators=all_architectures(),
        restorers={},
        devices=FakeDevices(),
        translation=translation or FakeTranslation(),
    )
    manager._run_process = fake_run_process
    register_model(registry, settings, MULTILINGUAL_ID)
    return manager, settings


def make_audio(settings: Settings, name: str = "cancion.wav") -> Path:
    path = settings.uploads_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF....WAVE")
    return path


async def crear(manager, settings, **extra):
    return await manager.create_job(
        source_path=make_audio(settings),
        original_filename="cancion.wav",
        asr_model_id=MULTILINGUAL_ID,
        **extra,
    )


# ---------------------------------------------------------------------------
# Etapa 1: preparar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preparar_deja_el_job_en_review_con_instrumental(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await crear(manager, settings)

    await manager._process_next()

    assert job.phase == "review"
    assert job.status == JobStatus.completed
    assert job.instrumental_path is not None and job.instrumental_path.exists()
    assert len(job.segments) == 2
    # El fuente sigue vivo: puede ser el fondo del render.
    assert job.source_path.exists()


@pytest.mark.asyncio
async def test_la_limpieza_corre_despues_de_separar_y_en_orden(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    separator = manager.separators[SEPARATION_MODELS[DEFAULT_SEPARATION_MODEL].architecture]
    job = await crear(
        manager, settings, cleanup_steps=["reverb_hq", "denoise"]
    )

    await manager._process_next()

    # Separacion primero; la cadena en el orden del CATALOGO (denoise antes
    # que dereverb), no en el orden en que llegaron los ids.
    assert separator.calls == [DEFAULT_SEPARATION_MODEL, "denoise", "reverb_hq"]
    assert job.phase == "review"


@pytest.mark.asyncio
async def test_seleccion_de_limpieza_redundante_se_rechaza_al_crear(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    with pytest.raises(ValueError):
        await crear(manager, settings, cleanup_steps=["deecho_normal", "deecho_aggressive"])


@pytest.mark.asyncio
async def test_modelo_de_separacion_desconocido_se_rechaza_al_crear(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    with pytest.raises(ValueError):
        await crear(manager, settings, separation_model_id="inventado")


@pytest.mark.asyncio
async def test_traducir_sin_idioma_explicito_se_rechaza(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    with pytest.raises(ValueError):
        await crear(manager, settings, translate_to="es")


@pytest.mark.asyncio
async def test_traducir_sin_el_par_instalado_se_rechaza(tmp_path: Path):
    manager, settings = make_manager(tmp_path, translation=FakeTranslation(installed=False))
    with pytest.raises(ValueError):
        await crear(manager, settings, language="ja", translate_to="es")


@pytest.mark.asyncio
async def test_la_traduccion_recibe_el_texto_original_no_el_romaji(tmp_path: Path):
    traduccion = FakeTranslation()
    manager, settings = make_manager(
        tmp_path, translation=traduccion, engine=FakeEngine(text="鉄壁 空")
    )
    job = await crear(
        manager, settings, language="ja", romanize=True, translate_to="es"
    )

    await manager._process_next()

    # El modelo de traduccion espera japones real, no romaji.
    assert traduccion.received == [["鉄壁", "空"]]
    # Y lo que queda para quemar en pantalla si es romaji.
    assert all(s.text.isascii() for s in job.segments)
    assert job.translated_lines == ["鉄壁".upper(), "空".upper()]


# ---------------------------------------------------------------------------
# Revision: editar la letra
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_editar_la_letra_reemplaza_texto_y_descarta_tiempos_por_palabra(
    tmp_path: Path,
):
    manager, settings = make_manager(tmp_path)
    job = await crear(manager, settings)
    await manager._process_next()

    manager.update_lyrics(job.id, [{"index": 0, "text": "chau"}])

    assert job.segments[0].text == "chau"
    assert job.segments[0].words == ()
    # Los tiempos de LINEA sobreviven: son los que anclan el karaoke.
    assert job.segments[0].start == 0.0


@pytest.mark.asyncio
async def test_editar_fuera_de_review_se_rechaza(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await crear(manager, settings)

    with pytest.raises(ValueError):
        manager.update_lyrics(job.id, [{"index": 0, "text": "chau"}])


@pytest.mark.asyncio
async def test_editar_una_linea_inexistente_se_rechaza(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await crear(manager, settings)
    await manager._process_next()

    with pytest.raises(ValueError):
        manager.update_lyrics(job.id, [{"index": 99, "text": "chau"}])


# ---------------------------------------------------------------------------
# Etapa 2: render
# ---------------------------------------------------------------------------


async def preparar_y_pedir_render(manager, settings, **render_extra):
    job = await crear(manager, settings)
    await manager._process_next()
    manager.request_render(job.id, **render_extra)
    return job


@pytest.mark.asyncio
async def test_render_completo_produce_el_video_y_limpia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager, settings = make_manager(tmp_path)

    async def duracion(*_a, **_k):
        return 3.0

    monkeypatch.setattr(
        "app.services.karaoke_job_manager.probe_duration_seconds", duracion
    )
    job = await preparar_y_pedir_render(manager, settings)
    assert job.phase == "rendering"

    await manager._process_next()

    assert job.phase == "completed"
    assert job.output_path is not None and job.output_path.exists()
    # Terminado de verdad: fuente y carpeta de trabajo ya no hacen falta.
    assert not job.source_path.exists()
    assert job.work_dir is None


@pytest.mark.asyncio
async def test_un_fallo_del_render_vuelve_a_review_sin_perder_el_instrumental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager, settings = make_manager(tmp_path)

    async def duracion(*_a, **_k):
        return 3.0

    async def render_roto(command, que_fallo):
        raise RuntimeError("ffmpeg murio")

    monkeypatch.setattr(
        "app.services.karaoke_job_manager.probe_duration_seconds", duracion
    )
    job = await preparar_y_pedir_render(manager, settings)
    manager._run_process = render_roto

    await manager._process_next()

    assert job.phase == "review"
    assert job.status == JobStatus.failed
    assert "ffmpeg murio" in (job.error or "")
    assert job.instrumental_path is not None and job.instrumental_path.exists()


@pytest.mark.asyncio
async def test_render_con_estilo_invalido_se_rechaza_antes_de_encolar(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await crear(manager, settings)
    await manager._process_next()

    with pytest.raises(ValueError):
        manager.request_render(job.id, subtitle_color="rojo")
    assert job.phase == "review"


@pytest.mark.asyncio
async def test_render_con_fondo_de_archivo_sin_archivo_se_rechaza(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await crear(manager, settings)
    await manager._process_next()

    with pytest.raises(ValueError):
        manager.request_render(job.id, background_kind="image")
