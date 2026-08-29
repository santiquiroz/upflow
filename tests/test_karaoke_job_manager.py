from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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


SEPARATION_SR = 44100


def dos_tonos() -> np.ndarray:
    # 2 s de "voces": 220 Hz el primer segundo (linea 0) y 440 Hz el segundo
    # (linea 1). Dos timbres distinguibles sin cargar ningun modelo.
    t = np.arange(SEPARATION_SR, dtype=np.float64) / SEPARATION_SR
    grave = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    agudo = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    return np.concatenate([grave, agudo]).astype(np.float32)


class SingerWavSeparator(FakeSeparator):
    """El stem vocal sale como wav REAL: los embeddings lo leen con soundfile."""

    async def run(self, decoded, stem_wavs, device, model_id=None, on_chunk=None):
        self.calls.append(model_id)
        for salida in stem_wavs:
            if Path(salida).stem == "vocals":
                sf.write(str(salida), dos_tonos(), SEPARATION_SR, subtype="FLOAT")
            else:
                Path(salida).write_bytes(b"wav")


class FakeEmbedder:
    """Determinista: la frecuencia dominante de la ventana decide el cantante."""

    def __init__(self, installed: bool = True) -> None:
        self.installed = installed
        self.windows: int = 0

    def available(self) -> bool:
        return self.installed

    def encode(self, audio: np.ndarray) -> np.ndarray:
        self.windows += 1
        cruces = int(np.count_nonzero(np.diff(np.signbit(audio))))
        hz = cruces / 2.0 / (len(audio) / 16000.0)
        if hz < 330.0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


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


def all_architectures(separator: FakeSeparator | None = None) -> dict[str, FakeSeparator]:
    separator = separator or FakeSeparator()
    return {spec.architecture: separator for spec in SEPARATION_MODELS.values()}


def make_manager(tmp_path: Path, *, translation=None, engine=None, separator=None, embedder=None):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    manager = KaraokeJobManager(
        settings,
        engine or FakeEngine(),
        DeviceSemaphores(settings),
        registry=registry,
        separators=all_architectures(separator),
        restorers={},
        devices=FakeDevices(),
        translation=translation or FakeTranslation(),
        embedder=embedder,
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


# ---------------------------------------------------------------------------
# F2a: deteccion de cantantes en la preparacion
# ---------------------------------------------------------------------------


def make_singer_manager(tmp_path: Path, **extra):
    return make_manager(
        tmp_path, separator=SingerWavSeparator(), embedder=FakeEmbedder(), **extra
    )


async def preparar_con_cantantes(manager, settings, **extra):
    job = await crear(manager, settings, detect_singers=True, **extra)
    await manager._process_next()
    return job


@pytest.mark.asyncio
async def test_detectar_cantantes_etiqueta_cada_linea_y_retiene_las_voces(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)

    job = await preparar_con_cantantes(manager, settings)

    assert job.phase == "review"
    # 220 Hz y 440 Hz: dos timbres, dos cantantes, en orden de aparicion.
    assert job.line_singers == ["s1", "s2"]
    # El stem vocal sobrevive al descarte: el render con mute lo necesita.
    assert job.vocals_path is not None and job.vocals_path.exists()


@pytest.mark.asyncio
async def test_sin_detectar_cantantes_el_stem_vocal_se_sigue_descartando(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)

    job = await crear(manager, settings)
    await manager._process_next()

    assert job.vocals_path is None
    assert not (job.work_dir / "vocals.wav").exists()
    assert job.line_singers == []


@pytest.mark.asyncio
async def test_singer_count_sin_detect_singers_se_rechaza_al_crear(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    with pytest.raises(ValueError):
        await crear(manager, settings, singer_count=2)


@pytest.mark.asyncio
async def test_singer_count_fuera_de_rango_se_rechaza_al_crear(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    for fuera in (1, 5):
        with pytest.raises(ValueError):
            await crear(manager, settings, detect_singers=True, singer_count=fuera)


@pytest.mark.asyncio
async def test_detectar_cantantes_sin_encoder_instalado_se_rechaza_al_crear(tmp_path: Path):
    manager, settings = make_manager(
        tmp_path, separator=SingerWavSeparator(), embedder=FakeEmbedder(installed=False)
    )
    with pytest.raises(ValueError, match="conversion de voz"):
        await crear(manager, settings, detect_singers=True)


@pytest.mark.asyncio
async def test_detectar_cantantes_con_un_modelo_sin_stem_vocal_se_rechaza(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    # reverb_hq emite dry/wet: no hay stem de voces del que sacar embeddings.
    with pytest.raises(ValueError, match="vocals"):
        await crear(
            manager, settings, detect_singers=True, separation_model_id="reverb_hq"
        )


# ---------------------------------------------------------------------------
# F2a: reasignar y renombrar cantantes en review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasignar_el_cantante_de_una_linea(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await preparar_con_cantantes(manager, settings)

    manager.update_lyrics(job.id, [{"index": 0, "singer": "s2"}])

    assert job.line_singers == ["s2", "s2"]
    # Reasignar el cantante no toca la letra ni sus tiempos por palabra.
    assert job.segments[0].text == "hola"


@pytest.mark.asyncio
async def test_reasignar_a_un_cantante_desconocido_se_rechaza(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await preparar_con_cantantes(manager, settings)

    with pytest.raises(ValueError, match="s9"):
        manager.update_lyrics(job.id, [{"index": 0, "singer": "s9"}])


@pytest.mark.asyncio
async def test_asignar_cantante_sin_deteccion_se_rechaza(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await crear(manager, settings)
    await manager._process_next()

    with pytest.raises(ValueError):
        manager.update_lyrics(job.id, [{"index": 0, "singer": "s1"}])


@pytest.mark.asyncio
async def test_renombrar_un_cantante(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await preparar_con_cantantes(manager, settings)

    manager.update_lyrics(job.id, [], singers=[{"id": "s1", "label": "Ana"}])

    assert job.singer_names == {"s1": "Ana"}


@pytest.mark.asyncio
async def test_renombrar_un_cantante_desconocido_se_rechaza(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await preparar_con_cantantes(manager, settings)

    with pytest.raises(ValueError, match="s7"):
        manager.update_lyrics(job.id, [], singers=[{"id": "s7", "label": "Ana"}])


# ---------------------------------------------------------------------------
# F2a: render con colores por cantante y mute de practica
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_con_mute_singer_produce_y_expone_la_mezcla_de_practica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager, settings = make_singer_manager(tmp_path)

    async def duracion(*_a, **_k):
        return 2.0

    monkeypatch.setattr(
        "app.services.karaoke_job_manager.probe_duration_seconds", duracion
    )
    job = await preparar_con_cantantes(manager, settings)

    comandos: list[list[str]] = []

    async def grabar(command, que_fallo):
        comandos.append(command)
        Path(command[-1]).write_bytes(b"out")

    manager._run_process = grabar
    manager.request_render(job.id, mute_singer="s1")
    await manager._process_next()

    assert job.phase == "completed"
    assert job.practice_audio_path is not None and job.practice_audio_path.exists()
    assert job.practice_audio_path.name == f"{job.id}.practice.flac"
    # La mezcla se arma con amix y el video usa ESA pista, no el instrumental.
    mezcla = next(c for c in comandos if any("amix" in arg for arg in c))
    assert str(job.practice_audio_path) == mezcla[-1]
    video = next(c for c in comandos if "libx264" in c)
    assert str(job.practice_audio_path) in video


@pytest.mark.asyncio
async def test_las_etiquetas_y_colores_llegan_al_ass_del_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager, settings = make_singer_manager(tmp_path)

    async def duracion(*_a, **_k):
        return 2.0

    monkeypatch.setattr(
        "app.services.karaoke_job_manager.probe_duration_seconds", duracion
    )
    job = await preparar_con_cantantes(manager, settings)

    manager.request_render(job.id, singer_colors={"s2": "#FF0000"})
    await manager._process_next()

    assert job.phase == "completed"
    ass = (settings.outputs_path / f"{job.id}.ass").read_text(encoding="utf-8")
    assert "Style: Singer_s2," in ass
    assert ",Singer_s2,,0" in ass  # la linea de s2 usa SU estilo
    assert ",Karaoke,,0" in ass  # la de s1, sin color propio, usa el principal


@pytest.mark.asyncio
async def test_render_con_color_de_cantante_invalido_se_rechaza(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await preparar_con_cantantes(manager, settings)

    with pytest.raises(ValueError):
        manager.request_render(job.id, singer_colors={"s1": "rojo"})
    assert job.phase == "review"


@pytest.mark.asyncio
async def test_render_con_cantante_desconocido_se_rechaza(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await preparar_con_cantantes(manager, settings)

    with pytest.raises(ValueError, match="s9"):
        manager.request_render(job.id, singer_colors={"s9": "#FF0000"})
    with pytest.raises(ValueError, match="s9"):
        manager.request_render(job.id, mute_singer="s9")


@pytest.mark.asyncio
async def test_mute_singer_sin_deteccion_se_rechaza(tmp_path: Path):
    manager, settings = make_singer_manager(tmp_path)
    job = await crear(manager, settings)
    await manager._process_next()

    with pytest.raises(ValueError):
        manager.request_render(job.id, mute_singer="s1")
