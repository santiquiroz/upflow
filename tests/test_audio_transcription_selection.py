from __future__ import annotations

import io
from pathlib import Path

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
from app.services.engines.music_transcription import NoteEvent
from app.services.storage import StorageService
from tests.test_audio_karaoke import (
    CommandRecordingPipeline,
    FakeSeparator,
    install_fake_model,
    make_manager,
    make_settings,
    write_upload_source,
)
from tests.test_minus_one import install_fake_umx

# ---------------------------------------------------------------------------
# F3a - transcripcion por stem: validacion en el manager (separate + stems con
# altura + pack instalado), wiring en el pipeline (WAV decodificado ->
# MIDI/MusicXML/tab), contrato de API (CSV, echo, ?stem=&fmt=) y enforcement
# del pack music-transcription (capability + missing_pack). Sigue la forma de
# test_minus_one.py / test_audio_karaoke.py.
# ---------------------------------------------------------------------------


def install_fake_transcription_model(settings: Settings) -> Settings:
    path = settings.music_transcription_model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-onnx")
    return settings


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


class FakeTranscriptionEngine:
    def __init__(self, notes: list[NoteEvent] | None = None) -> None:
        self.calls: list[tuple[Path, str, str | None]] = []
        self.notes = notes if notes is not None else [NoteEvent(0.0, 0.5, 60, 1.0)]

    def transcribe_file(self, input_wav: Path, device: str, model_id: str | None = None) -> list[NoteEvent]:
        self.calls.append((input_wav, device, model_id))
        return self.notes


def make_transcription_job(source_path: Path, **overrides: object) -> AudioJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        separate=True,
        separation_model="inst_hq_3",
        transcribe_stems=["vocals"],
        output_format="wav",
    )
    fields.update(overrides)
    return AudioJob(**fields)


# ---------------------------------------------------------------------------
# Manager: validacion de la seleccion de transcripcion
# ---------------------------------------------------------------------------


async def test_transcribe_stems_require_separate(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="separate"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            transcribe_stems=["vocals"],
        )


async def test_transcribe_stems_reject_drums_explicitly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    install_fake_transcription_model(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="drums"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="umx_4stem",
            transcribe_stems=["drums"],
        )


async def test_transcribe_stems_reject_an_unknown_stem_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    install_fake_transcription_model(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="guitar"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="umx_4stem",
            transcribe_stems=["guitar"],
        )


async def test_transcribe_stems_reject_a_derived_minus_one_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    install_fake_transcription_model(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="minus_drums"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="umx_4stem",
            transcribe_stems=["minus_drums"],
        )


async def test_transcribe_stems_require_the_pack_to_be_installed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)  # el modelo de separacion SI esta, el pack de transcripcion NO
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="transcripcion"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="umx_4stem",
            transcribe_stems=["vocals"],
        )


async def test_valid_transcribe_selection_lands_on_the_job_deduplicated(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    install_fake_transcription_model(settings)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        separate=True,
        separation_model="umx_4stem",
        transcribe_stems=["bass", "vocals", "bass"],
    )

    assert job.transcribe_stems == ["bass", "vocals"]


# ---------------------------------------------------------------------------
# Pipeline: MIDI + MusicXML (+ tab si guitar/bass) por stem pedido
# ---------------------------------------------------------------------------


def make_transcription_pipeline(
    settings: Settings, separator: FakeSeparator, transcription_engine: FakeTranscriptionEngine, architecture: str = "mdx"
) -> CommandRecordingPipeline:
    return CommandRecordingPipeline(
        settings, {}, {}, separators={architecture: separator}, transcription_engine=transcription_engine
    )


async def test_pipeline_writes_midi_and_musicxml_next_to_the_stems(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator = FakeSeparator()
    transcription_engine = FakeTranscriptionEngine()
    pipeline = make_transcription_pipeline(settings, separator, transcription_engine)
    job = make_transcription_job(write_upload_source(settings))

    await pipeline.run(job)

    assert set(job.transcription_output_paths) == {"vocals"}
    artifacts = job.transcription_output_paths["vocals"]
    # "vocals" no es guitar/bass: sin tab.
    assert set(artifacts) == {"midi", "musicxml"}
    assert artifacts["midi"].name == f"{job.id}.vocals.mid"
    assert artifacts["midi"].exists()
    assert artifacts["musicxml"].name == f"{job.id}.vocals.musicxml"
    assert artifacts["musicxml"].exists()
    assert len(transcription_engine.calls) == 1
    called_path, _device, _model_id = transcription_engine.calls[0]
    # El motor recibe el WAV DECODIFICADO del stem, no el mp3/flac final.
    assert called_path.name == "vocals.wav"


async def test_pipeline_writes_a_tab_for_bass_stems(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator = FakeSeparator()
    transcription_engine = FakeTranscriptionEngine()
    pipeline = make_transcription_pipeline(settings, separator, transcription_engine, architecture="umx")
    job = make_transcription_job(
        write_upload_source(settings), separation_model="umx_4stem", transcribe_stems=["bass"]
    )

    await pipeline.run(job)

    artifacts = job.transcription_output_paths["bass"]
    assert set(artifacts) == {"midi", "musicxml", "tab"}
    assert artifacts["tab"].name == f"{job.id}.bass.tab.txt"
    assert artifacts["tab"].exists()


async def test_pipeline_without_transcribe_stems_produces_no_artifacts(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator = FakeSeparator()
    transcription_engine = FakeTranscriptionEngine()
    pipeline = make_transcription_pipeline(settings, separator, transcription_engine)
    job = make_transcription_job(write_upload_source(settings), transcribe_stems=[])

    await pipeline.run(job)

    assert job.transcription_output_paths == {}
    assert transcription_engine.calls == []


# ---------------------------------------------------------------------------
# API: form CSV, respuesta, y descarga por ?stem=&fmt=
# ---------------------------------------------------------------------------


async def test_create_route_parses_transcribe_stems_field(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    install_fake_transcription_model(settings)
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
        separation_model="inst_hq_3",
        transcribe_stems="vocals",
        audio_jobs=manager,
        storage=storage,
        settings=settings,
    )

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.transcribe_stems == ["vocals"]


async def test_create_route_rejects_drums_with_400(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    install_fake_transcription_model(settings)
    storage = StorageService(settings)
    manager = make_manager(settings)

    with pytest.raises(HTTPException) as excinfo:
        await create_audio_job(
            request=None,
            file=make_upload("song.wav", b"fake-audio-bytes"),
            denoise=None,
            restore=None,
            device=None,
            output_format="flac",
            separate=True,
            separation_model="umx_4stem",
            transcribe_stems="drums",
            audio_jobs=manager,
            storage=storage,
            settings=settings,
        )
    assert excinfo.value.status_code == 400


def make_completed_transcription_job(tmp_path: Path) -> AudioJob:
    job = make_transcription_job(tmp_path / "song.wav")
    job.status = JobStatus.completed
    stem_outputs: dict[str, Path] = {}
    for stem_id in ("instrumental", "vocals"):
        path = tmp_path / f"{job.id}.{stem_id}.wav"
        path.write_bytes(stem_id.encode())
        stem_outputs[stem_id] = path
    job.output_path = stem_outputs["instrumental"]
    job.stem_output_paths = stem_outputs

    midi_path = tmp_path / f"{job.id}.vocals.mid"
    midi_path.write_bytes(b"fake-midi")
    musicxml_path = tmp_path / f"{job.id}.vocals.musicxml"
    musicxml_path.write_text("<score-partwise/>", encoding="utf-8")
    job.transcription_output_paths = {"vocals": {"midi": midi_path, "musicxml": musicxml_path}}
    return job


def test_response_echoes_transcribe_selection_and_lists_produced_downloads(
    tmp_path: Path,
) -> None:
    job = make_completed_transcription_job(tmp_path)

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    assert serialized["transcribeStems"] == ["vocals"]
    entries = {(item["stemId"], item["format"]): item["url"] for item in serialized["transcriptions"]}
    base = f"/api/v1/audio/jobs/{job.id}/download"
    assert entries[("vocals", "midi")] == f"{base}?stem=vocals&fmt=midi"
    assert entries[("vocals", "musicxml")] == f"{base}?stem=vocals&fmt=musicxml"
    # "vocals" nunca produce tab: no aparece en la lista.
    assert ("vocals", "tab") not in entries


async def test_download_serves_a_produced_transcription_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = make_completed_transcription_job(tmp_path)
    manager.jobs[job.id] = job

    response = await download_audio_job(job.id, stem="vocals", fmt="midi", audio_jobs=manager)

    assert Path(response.path) == job.transcription_output_paths["vocals"]["midi"]


async def test_download_rejects_an_unknown_fmt_with_400_listing_valid_ones(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = make_completed_transcription_job(tmp_path)
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="vocals", fmt="pdf", audio_jobs=manager)

    assert excinfo.value.status_code == 400
    assert "midi" in excinfo.value.detail
    assert "musicxml" in excinfo.value.detail
    assert "tab" in excinfo.value.detail


async def test_download_returns_404_for_a_format_that_was_not_produced(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = make_completed_transcription_job(tmp_path)  # sin "tab" para vocals
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="vocals", fmt="tab", audio_jobs=manager)

    assert excinfo.value.status_code == 404


async def test_download_transcription_requires_a_completed_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = make_transcription_job(write_upload_source(settings))  # status queued
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="vocals", fmt="midi", audio_jobs=manager)

    assert excinfo.value.status_code == 409


# ---------------------------------------------------------------------------
# Enforcement del pack: capability + missing_pack
# ---------------------------------------------------------------------------


def test_capability_needs_setup_without_the_model(tmp_path: Path) -> None:
    from app.services.capabilities import resolve_capabilities
    from app.services.model_registry import ModelRegistry

    settings = make_settings(tmp_path)
    resolved = {c.id: c for c in resolve_capabilities(settings, ModelRegistry(settings))}

    capability = resolved["audio.stemTranscription"]
    assert capability.status == "needs_setup"
    assert "music-transcription" in capability.missing_packs


def test_capability_is_available_with_the_model_present(tmp_path: Path) -> None:
    from app.services.capabilities import resolve_capabilities
    from app.services.model_registry import ModelRegistry

    settings = install_fake_transcription_model(make_settings(tmp_path))
    resolved = {c.id: c for c in resolve_capabilities(settings, ModelRegistry(settings))}

    assert resolved["audio.stemTranscription"].status == "available"


def test_missing_pack_message_names_the_transcription_model() -> None:
    from app.services.missing_pack import missing_pack_message

    message = missing_pack_message("music-transcription")

    assert "transcripcion" in message
    assert "Se baja desde la app" in message


def test_pack_provisioner_knows_the_download_script() -> None:
    from app.services.pack_provisioner import script_path

    path = script_path("music-transcription")

    assert path.name == "download-music-transcription.ps1"
    assert path.exists()
