from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import (
    audio_job_to_response,
    create_audio_job,
    download_audio_job,
)
from app.config import Settings
from app.models import AudioJob, JobStatus
from app.services.audio_job_manager import AudioJobManager
from app.services.audio_pipeline import AudioPipeline
from app.services.capabilities import resolve_capabilities
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.mdx_models import DEFAULT_SEPARATION_MODEL, SEPARATION_MODELS
from app.services.pack_provisioner import build_command
from app.services.progress import build_audio_stages
from app.services.storage import StorageService

# ---------------------------------------------------------------------------
# v0.59.0 - modo karaoke (separacion voz/instrumental) en el modulo Audio:
# validacion de exclusividad en el manager, stage "separating" con DOS salidas
# en el pipeline, contrato de API (separate + stem= + vocalsDownloadUrl),
# capability audio.karaoke y pack por variante. Sigue la forma de
# test_audio_pipeline.py / test_capabilities.py / test_pack_provisioner.py.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        KARAOKE_MODEL_DIR=str(tmp_path / "karaoke"),
        **overrides,
    )


def install_fake_model(settings: Settings, model_id: str = "inst_hq_3") -> Path:
    model_dir = settings.karaoke_model_dir_path
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / SEPARATION_MODELS[model_id].filename
    path.write_bytes(b"fake-onnx")
    return path


class FakeSeparator:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, Path, str, str]] = []

    async def run(
        self,
        input_wav: Path,
        instrumental_wav: Path,
        vocals_wav: Path,
        device: str,
        model_id: str = DEFAULT_SEPARATION_MODEL,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        self.calls.append((input_wav, instrumental_wav, vocals_wav, device, model_id))
        instrumental_wav.parent.mkdir(parents=True, exist_ok=True)
        instrumental_wav.write_bytes(b"fake-instrumental")
        vocals_wav.write_bytes(b"fake-vocals")


class CommandRecordingPipeline(AudioPipeline):
    """Fakes _run_process so no real ffmpeg runs; records commands."""

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
    return CommandRecordingPipeline(settings, {}, {}, separator=separator)


def make_manager(settings: Settings, separator: FakeSeparator | None = None) -> AudioJobManager:
    pipeline = make_pipeline(settings, separator)
    return AudioJobManager(settings, pipeline, DeviceSemaphores(settings))


def write_upload_source(settings: Settings, name: str = "song.wav") -> Path:
    source_path = settings.uploads_path / name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-source-bytes")
    return source_path


def make_separate_job(source_path: Path, **overrides: object) -> AudioJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        separate=True,
        separation_model="inst_hq_3",
    )
    fields.update(overrides)
    return AudioJob(**fields)


# ---------------------------------------------------------------------------
# Manager: exclusividad y validacion de modelo
# ---------------------------------------------------------------------------


async def test_separate_job_is_created_with_the_default_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        separate=True,
    )

    assert job.separate is True
    assert job.separation_model == DEFAULT_SEPARATION_MODEL


@pytest.mark.parametrize(
    "extra",
    [
        {"denoise": "rnnoise"},
        {"restore": "apollo"},
        {"voice_steps": ["deesser"]},
        {"master": "streaming"},
        {"voice_delivery": "broadcast"},
        {"voice_presence_db": 6.0},
    ],
)
async def test_separate_is_exclusive_against_every_other_step(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="karaoke corre solo"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            **extra,  # type: ignore[arg-type]
        )


async def test_empty_separation_model_resolves_to_the_first_installed(tmp_path: Path) -> None:
    # Solo voc_ft instalado: la capability marca karaoke disponible, asi que el
    # default fijo (inst_hq_3) daria un 400 missing-pack enganoso.
    settings = make_settings(tmp_path)
    install_fake_model(settings, "voc_ft")
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        separate=True,
    )

    assert job.separation_model == "voc_ft"


async def test_unknown_separation_model_lists_the_catalog(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="inst_hq_3.*voc_ft"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="no-existe",
        )


async def test_missing_model_asks_for_the_pack_naming_the_variant(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    manager = make_manager(settings)

    # inst_hq_3 esta instalado pero se pide voc_ft: el mensaje nombra ESA
    # variante, que es la que el boton de descarga tiene que bajar.
    with pytest.raises(ValueError, match=r"\(voc_ft\)") as excinfo:
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="voc_ft",
        )
    assert "boton" in str(excinfo.value)


async def test_separation_model_without_separate_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="separation_model"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            denoise=None,
            restore=None,
            separation_model="inst_hq_3",
        )


# ---------------------------------------------------------------------------
# Pipeline: stage separating con DOS salidas y formato aplicado a ambas
# ---------------------------------------------------------------------------


async def test_separation_produces_instrumental_and_vocals_outputs(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator = FakeSeparator()
    pipeline = make_pipeline(settings, separator)
    job = make_separate_job(write_upload_source(settings), output_format="flac")

    output_path = await pipeline.run(job)

    assert output_path.name == f"{job.id}.instrumental.flac"
    assert job.secondary_output_path is not None
    assert job.secondary_output_path.name == f"{job.id}.vocals.flac"
    assert output_path.exists()
    assert job.secondary_output_path.exists()


async def test_separation_reencodes_both_outputs_with_the_chosen_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_separate_job(write_upload_source(settings), output_format="mp3")

    await pipeline.run(job)

    encode_commands = [c for c in pipeline.commands if "libmp3lame" in c]
    encoded_outputs = {Path(c[-1]).name for c in encode_commands}
    assert encoded_outputs == {f"{job.id}.instrumental.mp3", f"{job.id}.vocals.mp3"}


async def test_separation_wav_output_moves_without_reencode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_separate_job(write_upload_source(settings), output_format="wav")

    output_path = await pipeline.run(job)

    # Solo el decode paso por ffmpeg: los dos stems se mueven tal cual.
    assert len(pipeline.commands) == 1
    assert output_path.exists()
    assert job.secondary_output_path is not None and job.secondary_output_path.exists()


async def test_separation_decodes_to_stereo_44100(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_separate_job(write_upload_source(settings))

    await pipeline.run(job)

    decode_command = pipeline.commands[0]
    assert "-ac" in decode_command and decode_command[decode_command.index("-ac") + 1] == "2"
    assert decode_command[decode_command.index("-ar") + 1] == "44100"


async def test_separator_receives_the_job_model_and_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator = FakeSeparator()
    pipeline = make_pipeline(settings, separator)
    job = make_separate_job(
        write_upload_source(settings), separation_model="voc_ft", device="dml:0"
    )

    await pipeline.run(job)

    (_input, _inst, _voc, device, model_id) = separator.calls[0]
    assert device == "dml:0"
    assert model_id == "voc_ft"


async def test_pipeline_without_separator_fails_loudly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, separator=None)
    job = make_separate_job(write_upload_source(settings))

    with pytest.raises(RuntimeError, match="sin\\s+motor de separacion"):
        await pipeline.run(job)


async def test_separation_stage_metadata_reaches_completed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_separate_job(write_upload_source(settings))

    await pipeline.run(job)

    assert job.metadata["stage"] == "completed"
    stage_keys = [stage["key"] for stage in job.metadata["stages"]]
    assert stage_keys == ["decoding", "separating", "finalizing"]


async def test_chunk_progress_is_strictly_increasing_within_the_separating_band(
    tmp_path: Path,
) -> None:
    # Sin el callback la barra quedaba congelada en ~11% (10/95) durante toda
    # la separacion; con el, cada chunk mueve el progreso dentro de
    # (10/95, 90/95] sin salirse a la banda de finalizing.
    settings = make_settings(tmp_path)
    job = make_separate_job(write_upload_source(settings))
    seen: list[float] = []

    class ChunkedSeparator(FakeSeparator):
        async def run(
            self,
            input_wav: Path,
            instrumental_wav: Path,
            vocals_wav: Path,
            device: str,
            model_id: str = DEFAULT_SEPARATION_MODEL,
            on_chunk: Callable[[int, int], None] | None = None,
        ) -> None:
            assert on_chunk is not None
            for done in range(1, 5):
                on_chunk(done, 4)
                seen.append(job.metadata["progress"])
            await super().run(
                input_wav, instrumental_wav, vocals_wav, device, model_id=model_id
            )

    pipeline = make_pipeline(settings, ChunkedSeparator())

    await pipeline.run(job)

    floor, ceiling = 10 / 95, 90 / 95
    assert all(earlier < later for earlier, later in zip(seen, seen[1:]))
    assert all(floor < value <= ceiling + 1e-9 for value in seen)
    assert seen[-1] == pytest.approx(ceiling)


def test_chunk_progress_does_not_restamp_stage_started_at(tmp_path: Path) -> None:
    from app.services.progress import advance_audio_stage, apply_audio_chunk_progress

    job = make_separate_job(tmp_path / "song.wav")
    advance_audio_stage(job, "separating")
    stamped = job.metadata["stageStartedAt"]

    apply_audio_chunk_progress(job, 1, 4)

    assert job.metadata["stage"] == "separating"
    assert job.metadata["stageStartedAt"] == stamped
    # Sin framesDone/framesTotal nuevos: "frames" mentiria para chunks.
    assert job.metadata["framesDone"] == 0
    assert job.metadata["framesTotal"] is None


def test_audio_stages_for_a_separate_job_hide_the_classic_chain(tmp_path: Path) -> None:
    job = make_separate_job(tmp_path / "song.wav")

    keys = [stage.key for stage in build_audio_stages(job)]

    assert keys == ["decoding", "separating", "finalizing"]


def test_audio_stages_without_separate_are_unchanged(tmp_path: Path) -> None:
    job = AudioJob(
        source_path=tmp_path / "song.wav",
        original_filename="song.wav",
        denoise="rnnoise",
    )

    keys = [stage.key for stage in build_audio_stages(job)]

    assert "separating" not in keys
    assert keys[0] == "decoding" and keys[-1] == "finalizing"


# ---------------------------------------------------------------------------
# API: form fields, respuesta y descarga por stem (CONTRATO, lo consume MCP)
# ---------------------------------------------------------------------------


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def test_create_route_forwards_separate_and_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "voc_ft")
    storage = StorageService(settings)
    manager = make_manager(settings)

    response = await create_audio_job(
        request=None,
        file=make_upload("song.wav", b"fake-audio-bytes"),
        denoise=None,
        restore=None,
        device=None,
        output_format="flac",
        separate=True,
        separation_model="voc_ft",
        audio_jobs=manager,
        storage=storage,
        settings=settings,
    )

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.separate is True
    assert job.separation_model == "voc_ft"


async def test_create_route_rejects_separate_combined_with_denoise(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    storage = StorageService(settings)
    manager = make_manager(settings)

    with pytest.raises(HTTPException) as excinfo:
        await create_audio_job(
            request=None,
            file=make_upload("song.wav", b"fake-audio-bytes"),
            denoise="rnnoise",
            restore=None,
            device=None,
            output_format="flac",
            separate=True,
            audio_jobs=manager,
            storage=storage,
            settings=settings,
        )
    assert excinfo.value.status_code == 400


def test_response_exposes_vocals_download_url_when_completed(tmp_path: Path) -> None:
    job = make_separate_job(tmp_path / "song.wav")
    job.status = JobStatus.completed
    job.output_path = tmp_path / f"{job.id}.instrumental.flac"
    job.secondary_output_path = tmp_path / f"{job.id}.vocals.flac"

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    assert serialized["separate"] is True
    assert serialized["separationModel"] == "inst_hq_3"
    assert serialized["downloadUrl"] == f"/api/v1/audio/jobs/{job.id}/download"
    assert serialized["vocalsDownloadUrl"] == (
        f"/api/v1/audio/jobs/{job.id}/download?stem=vocals"
    )


def test_response_has_null_vocals_url_for_non_separate_jobs(tmp_path: Path) -> None:
    job = AudioJob(
        source_path=tmp_path / "x.wav", original_filename="x.wav", denoise="rnnoise"
    )
    job.status = JobStatus.completed
    job.output_path = tmp_path / "out.flac"

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    assert serialized["separate"] is False
    assert serialized["vocalsDownloadUrl"] is None


async def make_completed_manager_job(
    tmp_path: Path, *, separate: bool
) -> tuple[AudioJobManager, AudioJob]:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = (
        make_separate_job(tmp_path / "song.wav")
        if separate
        else AudioJob(source_path=tmp_path / "x.wav", original_filename="x.wav", denoise="rnnoise")
    )
    job.status = JobStatus.completed
    job.output_path = tmp_path / "instrumental.flac"
    job.output_path.write_bytes(b"instrumental")
    if separate:
        job.secondary_output_path = tmp_path / "vocals.flac"
        job.secondary_output_path.write_bytes(b"vocals")
    manager.jobs[job.id] = job
    return manager, job


async def test_download_defaults_to_the_instrumental_stem(tmp_path: Path) -> None:
    manager, job = await make_completed_manager_job(tmp_path, separate=True)

    response = await download_audio_job(job.id, stem="instrumental", audio_jobs=manager)

    assert Path(response.path) == job.output_path


async def test_download_serves_the_vocals_stem(tmp_path: Path) -> None:
    manager, job = await make_completed_manager_job(tmp_path, separate=True)

    response = await download_audio_job(job.id, stem="vocals", audio_jobs=manager)

    assert Path(response.path) == job.secondary_output_path


async def test_download_rejects_an_unknown_stem_with_400(tmp_path: Path) -> None:
    manager, job = await make_completed_manager_job(tmp_path, separate=True)

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="drums", audio_jobs=manager)
    assert excinfo.value.status_code == 400


async def test_download_vocals_on_a_non_separate_job_is_409(tmp_path: Path) -> None:
    manager, job = await make_completed_manager_job(tmp_path, separate=False)

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="vocals", audio_jobs=manager)
    assert excinfo.value.status_code == 409


# ---------------------------------------------------------------------------
# Capability + pack (patron audiosr/shap-e-img2img)
# ---------------------------------------------------------------------------


class FakeRegistry:
    def list(self) -> list[object]:
        return []


def _find(resolved: list, capability_id: str):
    return next(item for item in resolved if item.id == capability_id)


class TestKaraokeEnElArbol:
    def test_sin_modelos_pide_bajar_el_paquete(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)

        karaoke = _find(resolve_capabilities(settings, FakeRegistry()), "audio.karaoke")

        assert karaoke.status == "needs_setup"
        assert "karaoke" in karaoke.missing_packs
        assert karaoke.setup_reason_key == "capability.setup.missingPack"

    def test_la_carpeta_vacia_no_alcanza(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        settings.karaoke_model_dir_path.mkdir(parents=True)

        karaoke = _find(resolve_capabilities(settings, FakeRegistry()), "audio.karaoke")

        assert karaoke.status == "needs_setup"

    def test_con_un_modelo_queda_disponible_sin_flag(self, tmp_path: Path) -> None:
        # Sin SettingRequirement a proposito: descargar = usable.
        settings = make_settings(tmp_path)
        install_fake_model(settings, "voc_ft")

        karaoke = _find(resolve_capabilities(settings, FakeRegistry()), "audio.karaoke")

        assert karaoke.status == "available"


class TestPackKaraokePorVariante:
    def test_la_variante_viaja_al_script_como_modelo(self) -> None:
        comando = build_command("karaoke", variant="voc_ft")

        assert comando[-2:] == ["-Model", "voc_ft"]
        assert comando[-3].endswith("download-karaoke.ps1")

    def test_una_variante_con_forma_rara_se_rechaza(self) -> None:
        with pytest.raises(ValueError):
            build_command("karaoke", variant="voc_ft; rm -rf /")

    def test_el_script_de_descarga_existe(self) -> None:
        from app.services.pack_provisioner import script_path

        assert script_path("karaoke").exists()

    def test_los_ids_del_catalogo_pasan_el_patron_de_variante(self) -> None:
        # Un id nuevo que el patron rechace dejaria el boton de descarga roto.
        for model_id in SEPARATION_MODELS:
            assert build_command("karaoke", variant=model_id)[-1] == model_id

    def test_una_variante_fuera_del_catalogo_se_rechaza_antes_de_encolar(self) -> None:
        # Pasa el regex pero no existe: sin el catalogo cerrado el job se
        # encolaba y moria despues solo contra el ValidateSet de PowerShell.
        with pytest.raises(ValueError, match="foo_bar"):
            build_command("karaoke", variant="foo_bar")

    async def test_la_api_devuelve_400_para_una_variante_fuera_del_catalogo(
        self, tmp_path: Path
    ) -> None:
        from app.api.routes import provision_pack
        from app.services.pack_provisioner import PackProvisioner

        provisioner = PackProvisioner(make_settings(tmp_path))
        request = type(
            "R",
            (),
            {"app": type("A", (), {"state": type("S", (), {"pack_provisioner": provisioner})()})()},
        )()

        with pytest.raises(HTTPException) as excinfo:
            await provision_pack(pack="karaoke", request=request, variant="foo_bar")
        assert excinfo.value.status_code == 400
        assert "foo_bar" in str(excinfo.value.detail)

    def test_el_script_no_derrapa_del_catalogo(self) -> None:
        # Anti-drift: un id nuevo en SEPARATION_MODELS que no llegue al ps1
        # (ValidateSet, filename o hash) es un boton de descarga que siempre
        # falla async — este test lo vuelve un fallo de suite.
        from app.services.pack_provisioner import script_path

        script = script_path("karaoke").read_text(encoding="utf-8")
        validate_set_line = next(
            line for line in script.splitlines() if "ValidateSet" in line
        )
        for spec in SEPARATION_MODELS.values():
            assert f"'{spec.id}'" in validate_set_line
            assert spec.filename in script
            assert spec.uvr_hash in script
            assert spec.sha256 in script
