from __future__ import annotations

import asyncio
import io
from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import audio_capabilities, audio_job_to_response, create_audio_job
from app.config import Settings
from app.models import AudioJob
from app.services.audio_job_manager import AudioJobManager
from app.services.audio_pipeline import AudioPipeline
from app.services.cleanup_chain import (
    CLEANUP_CHAIN,
    OVERPROCESSING_PASS_THRESHOLD,
    RedundantCleanupSelection,
    UnknownCleanupStep,
    cleanup_steps_from_selection,
)
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.separation_models import SEPARATION_MODELS
from app.services.progress import (
    apply_audio_chunk_progress,
    build_audio_stages,
    cleanup_stage_key,
)
from app.services.storage import StorageService

# ---------------------------------------------------------------------------
# La LIMPIEZA como cadena propia: los modelos de categoria "cleanup" dejan de
# ser un modo exclusivo de separacion y pasan a ser filtros encadenables sobre
# cualquier audio. Cubre el orden fijo por catalogo, la exclusion por familia,
# la convivencia con el resto de la cadena clasica, las etapas por pasada, la
# cancelacion y los 400 del contrato. Sigue la forma de test_audio_karaoke.py.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        KARAOKE_MODEL_DIR=str(tmp_path / "karaoke"),
        **overrides,
    )


def install_models(settings: Settings, *model_ids: str) -> None:
    model_dir = settings.karaoke_model_dir_path
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        (model_dir / SEPARATION_MODELS[model_id].filename).write_bytes(b"fake-onnx")


def install_all_cleanup_models(settings: Settings) -> None:
    install_models(settings, *[step.model_id for step in CLEANUP_CHAIN])


class FakeSeparator:
    """Escribe stems fake y deja registro de cada pasada, con su entrada."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[Path, ...], str, str]] = []

    async def run(
        self,
        input_wav: Path,
        outputs: Sequence[Path],
        device: str,
        model_id: str = "denoise",
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        main_wav, other_wav = tuple(outputs)
        self.calls.append((input_wav, tuple(outputs), device, model_id))
        main_wav.parent.mkdir(parents=True, exist_ok=True)
        main_wav.write_bytes(b"fake-clean")
        other_wav.write_bytes(b"fake-removed")


class CommandRecordingPipeline(AudioPipeline):
    """Fakea _run_process para que no corra ffmpeg de verdad; graba comandos."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.commands: list[list[str]] = []

    async def _run_process(self, command: list[str], failure_message: str) -> None:
        self.commands.append(command)
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-audio-bytes")


def make_pipeline(
    settings: Settings, separator: FakeSeparator | None = None
) -> CommandRecordingPipeline:
    # Los modelos de limpieza cruzan las dos arquitecturas (VR para de-noise y
    # de-echo, MDX para reverb_hq): el mismo fake atiende ambas.
    separators = {"vr": separator, "mdx": separator} if separator is not None else None
    return CommandRecordingPipeline(settings, {}, {}, separators=separators)


def make_manager(settings: Settings, separator: FakeSeparator | None = None) -> AudioJobManager:
    return AudioJobManager(settings, make_pipeline(settings, separator), DeviceSemaphores(settings))


def write_upload_source(settings: Settings, name: str = "song.wav") -> Path:
    source_path = settings.uploads_path / name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-source-bytes")
    return source_path


def make_cleanup_job(source_path: Path, steps: list[str], **overrides: object) -> AudioJob:
    fields: dict[str, object] = dict(
        source_path=source_path,
        original_filename=source_path.name,
        cleanup_steps=steps,
    )
    fields.update(overrides)
    return AudioJob(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Catalogo: el ORDEN lo fija el catalogo, no el request
# ---------------------------------------------------------------------------


def test_the_chain_order_is_denoise_then_deecho_then_dereverb() -> None:
    # El orden tiene causalidad documentada (cleanup_chain.py): el ruido de
    # banda ancha confunde a los modelos de eco, y la cola difusa de la reverb
    # entierra los reflejos discretos que el de-echo busca.
    families = [step.family for step in CLEANUP_CHAIN]

    assert families[0] == "denoise"
    assert families[-1] == "dereverb"
    assert families.index("deecho") < families.index("dereverb")


@pytest.mark.parametrize(
    "sent",
    [
        ["denoise", "deecho_normal", "reverb_hq"],
        ["reverb_hq", "deecho_normal", "denoise"],
        ["deecho_normal", "reverb_hq", "denoise"],
        ["reverb_hq", "denoise", "deecho_normal"],
    ],
)
def test_ids_in_any_order_produce_the_same_chain(sent: list[str]) -> None:
    steps = cleanup_steps_from_selection(sent)

    assert [step.model_id for step in steps] == ["denoise", "deecho_normal", "reverb_hq"]


def test_a_repeated_id_collapses_into_one_pass() -> None:
    # Pedir dos veces el mismo modelo no puede significar dos pasadas distintas.
    steps = cleanup_steps_from_selection(["denoise", "denoise"])

    assert [step.model_id for step in steps] == ["denoise"]


def test_an_empty_selection_is_an_empty_chain() -> None:
    assert cleanup_steps_from_selection([]) == []


# ---------------------------------------------------------------------------
# Exclusion por familia: regla del catalogo, no ifs en la UI
# ---------------------------------------------------------------------------


def test_the_two_deecho_intensities_exclude_each_other() -> None:
    with pytest.raises(RedundantCleanupSelection) as excinfo:
        cleanup_steps_from_selection(["deecho_normal", "deecho_aggressive"])

    message = str(excinfo.value)
    assert "deecho_normal" in message and "deecho_aggressive" in message
    assert "quitar eco" in message


def test_deecho_dereverb_excludes_the_plain_deecho_models() -> None:
    with pytest.raises(RedundantCleanupSelection, match="quitar eco"):
        cleanup_steps_from_selection(["deecho_dereverb", "deecho_normal"])


def test_deecho_dereverb_also_excludes_reverb_hq() -> None:
    # Hace eco Y reverb en una pasada: cubre las dos familias, asi que sumarle
    # reverb_hq seria pagar una pasada con perdida por algo ya resuelto.
    with pytest.raises(RedundantCleanupSelection, match="quitar reverb"):
        cleanup_steps_from_selection(["deecho_dereverb", "reverb_hq"])


def test_deecho_dereverb_alone_is_a_valid_two_family_chain() -> None:
    steps = cleanup_steps_from_selection(["denoise", "deecho_dereverb"])

    assert [step.model_id for step in steps] == ["denoise", "deecho_dereverb"]


def test_the_longest_valid_chain_is_one_pass_per_family() -> None:
    steps = cleanup_steps_from_selection(["denoise", "deecho_aggressive", "reverb_hq"])

    assert len(steps) == OVERPROCESSING_PASS_THRESHOLD


def test_an_unknown_step_lists_the_valid_ones() -> None:
    with pytest.raises(UnknownCleanupStep) as excinfo:
        cleanup_steps_from_selection(["no-existe"])

    message = str(excinfo.value)
    assert "no-existe" in message
    for step in CLEANUP_CHAIN:
        assert step.model_id in message


def test_a_karaoke_model_is_not_a_cleanup_step() -> None:
    # inst_hq_3 esta en el catalogo de separacion pero no en el de limpieza:
    # separa dos cosas que el usuario quiere, no quita un defecto.
    with pytest.raises(UnknownCleanupStep, match="inst_hq_3"):
        cleanup_steps_from_selection(["inst_hq_3"])


# ---------------------------------------------------------------------------
# Manager: normalizacion, packs faltantes y convivencia con el resto
# ---------------------------------------------------------------------------


async def test_the_job_stores_the_chain_already_normalized(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_all_cleanup_models(settings)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        cleanup_steps=["reverb_hq", "denoise"],
    )

    # Ordenado por catalogo: pipeline, etapas y respuesta ven la MISMA cadena.
    assert job.cleanup_steps == ["denoise", "reverb_hq"]


async def test_a_cleanup_only_job_is_a_valid_request(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_all_cleanup_models(settings)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        cleanup_steps=["denoise"],
    )

    assert job.cleanup_steps == ["denoise"]
    assert job.denoise is None and job.restore is None


async def test_cleanup_combines_with_master_in_the_same_job(tmp_path: Path) -> None:
    # Esto es lo que la limpieza gana al dejar de ser un modo exclusivo.
    settings = make_settings(tmp_path)
    install_all_cleanup_models(settings)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        cleanup_steps=["denoise", "deecho_normal"],
        master="streaming",
        voice_steps=["deesser"],
    )

    assert job.cleanup_steps == ["denoise", "deecho_normal"]
    assert job.master == "streaming"
    assert job.voice_steps == ["deesser"]


async def test_a_job_with_nothing_requested_and_the_same_format_is_still_rejected(
    tmp_path: Path,
) -> None:
    # Sin pasos Y con el formato de salida igual al del archivo no queda nada
    # que hacer. Sin pasos pero con OTRO formato si: es una conversion pura, y
    # eso lo cubre test_audio_conversion.py.
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="ya esta en WAV"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            output_format="wav",
        )


async def test_a_missing_cleanup_model_asks_for_the_pack_naming_the_variant(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    install_models(settings, "denoise")
    manager = make_manager(settings)

    with pytest.raises(ValueError, match=r"\(deecho_normal\)") as excinfo:
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            cleanup_steps=["denoise", "deecho_normal"],
        )
    assert "boton" in str(excinfo.value)


async def test_separate_and_cleanup_together_are_rejected_for_their_own_reason(
    tmp_path: Path,
) -> None:
    # No es el motivo del stem ambiguo: es que una entrega DOS archivos y la
    # otra UNO, asi que la combinacion no define que se entrega.
    settings = make_settings(tmp_path)
    install_models(settings, "inst_hq_3", "denoise")
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="DOS archivos") as excinfo:
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            cleanup_steps=["denoise"],
        )
    assert "karaoke corre solo" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Pipeline: una pasada por paso, encadenadas, con UNA sola salida
# ---------------------------------------------------------------------------


async def test_each_pass_feeds_the_next_one(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator = FakeSeparator()
    pipeline = make_pipeline(settings, separator)
    job = make_cleanup_job(write_upload_source(settings), ["denoise", "deecho_normal"])

    await pipeline.run(job)

    assert [call[3] for call in separator.calls] == ["denoise", "deecho_normal"]
    # La entrada de la segunda pasada es la salida limpia de la primera.
    first_clean_output = separator.calls[0][1][0]
    second_input = separator.calls[1][0]
    assert second_input == first_clean_output


async def test_the_chain_produces_a_single_output_and_no_stems(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_cleanup_job(
        write_upload_source(settings), ["denoise", "deecho_normal"], output_format="flac"
    )

    output_path = await pipeline.run(job)

    assert output_path.name == f"{job.id}.flac"
    # Sin secundario: los stems removidos mueren con el work dir.
    assert job.stem_output_paths == {}
    assert [p.name for p in settings.outputs_path.iterdir()] == [f"{job.id}.flac"]


async def test_the_chain_decodes_to_stereo_44100_like_separation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_cleanup_job(write_upload_source(settings), ["denoise"])

    await pipeline.run(job)

    decode = pipeline.commands[0]
    assert decode[decode.index("-ac") + 1] == "2"
    assert decode[decode.index("-ar") + 1] == "44100"


async def test_a_following_classic_denoise_gets_48k_back(tmp_path: Path) -> None:
    # Los separadores emiten 44100; DeepFilterNet/RNNoise estan especificados a
    # 48000, que es a lo que decodifica el resto del pipeline.
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    pipeline.audio_enhancers = {"rnnoise": _NoopEnhancer()}
    job = make_cleanup_job(write_upload_source(settings), ["denoise"], denoise="rnnoise")

    await pipeline.run(job)

    resample = [c for c in pipeline.commands if Path(c[-1]).name == "cleaned-48k.wav"]
    assert len(resample) == 1
    assert resample[0][resample[0].index("-ar") + 1] == "48000"


async def test_without_a_classic_denoise_there_is_no_extra_resample(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_cleanup_job(write_upload_source(settings), ["denoise"])

    await pipeline.run(job)

    assert not [c for c in pipeline.commands if Path(c[-1]).name == "cleaned-48k.wav"]


class _NoopEnhancer:
    async def run(self, input_wav: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"fake-denoised")


async def test_the_pipeline_without_the_engine_fails_loudly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, separator=None)
    job = make_cleanup_job(write_upload_source(settings), ["denoise"])

    with pytest.raises(RuntimeError, match="sin ese motor de separacion"):
        await pipeline.run(job)


async def test_three_passes_stamp_the_overprocessing_warning(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_cleanup_job(
        write_upload_source(settings), ["denoise", "deecho_normal", "reverb_hq"]
    )

    await pipeline.run(job)

    assert job.metadata["cleanupPasses"] == 3
    # El aviso viaja en el job para que un agente por MCP lo vea igual que la UI.
    assert "sobreprocesado" in job.metadata["cleanupOverprocessed"]


async def test_two_passes_do_not_warn(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_cleanup_job(write_upload_source(settings), ["denoise", "deecho_normal"])

    await pipeline.run(job)

    assert job.metadata["cleanupPasses"] == 2
    assert "cleanupOverprocessed" not in job.metadata


# ---------------------------------------------------------------------------
# Etapas y progreso: una por pasada, con el nombre del modelo
# ---------------------------------------------------------------------------


def test_there_is_one_stage_per_pass_named_after_its_model(tmp_path: Path) -> None:
    job = make_cleanup_job(tmp_path / "song.wav", ["denoise", "deecho_normal"])

    stages = build_audio_stages(job)

    assert [stage.key for stage in stages] == [
        "decoding",
        cleanup_stage_key("denoise"),
        cleanup_stage_key("deecho_normal"),
        "finalizing",
    ]
    # El usuario ve QUE modelo corre ahora: es lo unico que hace entendible una
    # espera de varios minutos repetida.
    assert SEPARATION_MODELS["denoise"].name in stages[1].label


def test_cleanup_stages_sit_before_restore_and_mastering(tmp_path: Path) -> None:
    job = make_cleanup_job(
        tmp_path / "song.wav", ["denoise"], restore="apollo", master="streaming"
    )

    keys = [stage.key for stage in build_audio_stages(job)]

    assert keys.index(cleanup_stage_key("denoise")) < keys.index("restoring")
    assert keys.index("restoring") < keys.index("mastering")


def test_a_separate_job_shows_no_cleanup_stages(tmp_path: Path) -> None:
    job = AudioJob(
        source_path=tmp_path / "song.wav",
        original_filename="song.wav",
        separate=True,
        separation_model="inst_hq_3",
    )

    keys = [stage.key for stage in build_audio_stages(job)]

    assert keys == ["decoding", "separating", "finalizing"]


async def test_chunk_progress_advances_within_the_pass_that_is_running(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    job = make_cleanup_job(write_upload_source(settings), ["denoise", "deecho_normal"])
    per_pass: dict[str, list[float]] = {}

    class ChunkedSeparator(FakeSeparator):
        async def run(self, input_wav, outputs, device, model_id="denoise", on_chunk=None):  # type: ignore[no-untyped-def]
            assert on_chunk is not None
            seen: list[float] = []
            for done in range(1, 5):
                on_chunk(done, 4)
                seen.append(job.metadata["progress"])
                assert job.metadata["stage"] == cleanup_stage_key(model_id)
            per_pass[model_id] = seen
            await super().run(input_wav, outputs, device, model_id=model_id)

    await make_pipeline(settings, ChunkedSeparator()).run(job)

    for seen in per_pass.values():
        assert all(earlier < later for earlier, later in zip(seen, seen[1:]))
    # La segunda pasada progresa POR ENCIMA de la primera: cada una tiene su
    # propia banda y no se pisan.
    assert per_pass["deecho_normal"][0] > per_pass["denoise"][-1]


def test_chunk_progress_does_not_restamp_stage_started_at(tmp_path: Path) -> None:
    from app.services.progress import advance_audio_stage

    job = make_cleanup_job(tmp_path / "song.wav", ["denoise"])
    stage_key = cleanup_stage_key("denoise")
    advance_audio_stage(job, stage_key)
    stamped = job.metadata["stageStartedAt"]

    apply_audio_chunk_progress(job, 1, 4, stage_key)

    assert job.metadata["stage"] == stage_key
    assert job.metadata["stageStartedAt"] == stamped


# ---------------------------------------------------------------------------
# Cancelacion: entre pasadas y dentro de una
# ---------------------------------------------------------------------------


async def test_cancelling_between_passes_stops_the_chain(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job = make_cleanup_job(
        write_upload_source(settings), ["denoise", "deecho_normal", "reverb_hq"]
    )
    started: list[str] = []

    class SlowSeparator(FakeSeparator):
        async def run(self, input_wav, outputs, device, model_id="denoise", on_chunk=None):  # type: ignore[no-untyped-def]
            started.append(model_id)
            # Punto de suspension real: es donde aterriza el cancel del manager.
            await asyncio.sleep(0.05)
            await super().run(input_wav, outputs, device, model_id=model_id)

    pipeline = make_pipeline(settings, SlowSeparator())
    task = asyncio.ensure_future(pipeline.run(job))
    await asyncio.sleep(0.08)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    # Arranco la segunda pero nunca llego a la tercera.
    assert started == ["denoise", "deecho_normal"]


async def test_cancelling_leaves_no_output_behind(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job = make_cleanup_job(write_upload_source(settings), ["denoise", "deecho_normal"])

    class SlowSeparator(FakeSeparator):
        async def run(self, input_wav, outputs, device, model_id="denoise", on_chunk=None):  # type: ignore[no-untyped-def]
            await asyncio.sleep(0.05)
            await super().run(input_wav, outputs, device, model_id=model_id)

    task = asyncio.ensure_future(make_pipeline(settings, SlowSeparator()).run(job))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not settings.outputs_path.exists() or not list(settings.outputs_path.iterdir())


# ---------------------------------------------------------------------------
# API: contrato del form, respuesta y catalogo de capacidades
# ---------------------------------------------------------------------------


def make_upload(name: str = "song.wav") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(b"fake-audio"))


async def create_via_route(
    settings: Settings, manager: AudioJobManager, **form: object
) -> object:
    return await create_audio_job(
        request=None,  # type: ignore[arg-type]
        file=make_upload(),
        denoise=None,
        restore=None,
        device=None,
        output_format="flac",
        audio_jobs=manager,
        storage=StorageService(settings),
        settings=settings,
        **form,  # type: ignore[arg-type]
    )


async def test_the_route_accepts_the_chain_as_a_csv_field(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_all_cleanup_models(settings)
    manager = make_manager(settings)

    created = await create_via_route(settings, manager, cleanup_steps="reverb_hq,denoise")

    job = manager.get_job(created.job_id)  # type: ignore[attr-defined]
    assert job is not None and job.cleanup_steps == ["denoise", "reverb_hq"]


async def test_an_unknown_step_is_a_400_that_lists_the_catalog(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_all_cleanup_models(settings)
    manager = make_manager(settings)

    with pytest.raises(HTTPException) as excinfo:
        await create_via_route(settings, manager, cleanup_steps="no-existe")

    assert excinfo.value.status_code == 400
    assert "denoise" in excinfo.value.detail


async def test_a_redundant_combination_is_a_400_that_names_the_pair(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_all_cleanup_models(settings)
    manager = make_manager(settings)

    with pytest.raises(HTTPException) as excinfo:
        await create_via_route(
            settings, manager, cleanup_steps="deecho_normal,deecho_aggressive"
        )

    assert excinfo.value.status_code == 400
    assert "deecho_normal" in excinfo.value.detail
    assert "deecho_aggressive" in excinfo.value.detail


async def test_separate_plus_chain_is_a_400(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_models(settings, "inst_hq_3", "denoise")
    manager = make_manager(settings)

    with pytest.raises(HTTPException) as excinfo:
        await create_via_route(settings, manager, separate=True, cleanup_steps="denoise")

    assert excinfo.value.status_code == 400
    assert "DOS archivos" in excinfo.value.detail


def test_the_response_echoes_the_normalized_chain(tmp_path: Path) -> None:
    job = make_cleanup_job(tmp_path / "song.wav", ["denoise", "reverb_hq"])

    response = audio_job_to_response(job)

    assert response.cleanup_steps == ["denoise", "reverb_hq"]
    # El contrato de dos salidas queda SOLO para el modo separacion.
    assert response.stems is None


async def test_capabilities_expose_the_chain_in_execution_order(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_models(settings, "denoise")

    capabilities = await audio_capabilities(settings=settings)

    ids = [step.id for step in capabilities.cleanup_steps]
    assert ids == [step.model_id for step in CLEANUP_CHAIN]
    assert capabilities.cleanup_overprocessing_threshold == OVERPROCESSING_PASS_THRESHOLD
    by_id = {step.id: step for step in capabilities.cleanup_steps}
    assert by_id["denoise"].installed is True
    assert by_id["reverb_hq"].installed is False
    # `covers` es lo que le permite a la UI aplicar la exclusion sin hard-codear
    # ids: deecho_dereverb resuelve dos familias.
    assert set(by_id["deecho_dereverb"].covers) == {"deecho", "dereverb"}
    assert by_id["reverb_hq"].covers == ["dereverb"]
