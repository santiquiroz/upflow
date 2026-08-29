from __future__ import annotations

import io
from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import (
    audio_job_to_response,
    create_audio_job,
    download_audio_job,
)
from app.models import AudioJob, JobStatus
from app.services.engines.minus_one import (
    GUIDE_PERCENT_MAX,
    derive_minus_one,
    derive_minus_one_file,
)
from app.services.engines.separation_models import SEPARATION_MODELS
from app.services.storage import StorageService
from tests.test_audio_karaoke import (
    CommandRecordingPipeline,
    FakeSeparator,
    make_manager,
    make_pipeline,
    make_settings,
    write_upload_source,
)

# ---------------------------------------------------------------------------
# F1a - pistas de practica "minus-one": derivacion pura (mix - g*stem, clip
# guard sin normalizar), validacion en el manager (separate + modelo >=3 stems),
# derivados en stem_output_paths del pipeline y contrato de API
# (?stem=minus_<id>, practiceStems/practiceGuidePercent). Sigue la forma de
# test_audio_karaoke.py.
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100


def sine_stems(samples: int = 4410, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    stems: dict[str, np.ndarray] = {}
    for index, name in enumerate(("vocals", "drums", "bass", "other")):
        freq = 110.0 * (index + 1) + float(rng.uniform(0, 5))
        mono = 0.2 * np.sin(2 * np.pi * freq * t)
        stems[name] = np.stack([mono, mono * 0.9], axis=1).astype(np.float32)
    return stems


def mix_of(stems: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(np.stack(list(stems.values())).astype(np.float64), axis=0).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# Derivacion pura
# ---------------------------------------------------------------------------


def test_reconstruction_is_sample_accurate_without_guide() -> None:
    stems = sine_stems()
    mix = mix_of(stems)

    minus = derive_minus_one(mix, stems["drums"], guide_percent=0)

    expected = mix.astype(np.float64) - stems["drums"].astype(np.float64)
    np.testing.assert_allclose(minus, expected, atol=1e-12)


def test_guide_percent_keeps_that_fraction_of_the_stem() -> None:
    stems = sine_stems()
    mix = mix_of(stems)

    minus = derive_minus_one(mix, stems["bass"], guide_percent=30)

    expected = mix.astype(np.float64) - 0.7 * stems["bass"].astype(np.float64)
    np.testing.assert_allclose(minus, expected, atol=1e-12)


def test_clip_guard_scales_down_only_when_the_peak_exceeds_full_scale() -> None:
    t = np.arange(4410, dtype=np.float64) / SAMPLE_RATE
    mix = (0.9 * np.sin(2 * np.pi * 220.0 * t)).reshape(-1, 1).astype(np.float32)
    # Stem en contrafase: la resta SUMA y el pico sale de escala (1.4).
    stem = (-0.5 * np.sin(2 * np.pi * 220.0 * t)).reshape(-1, 1).astype(np.float32)

    minus = derive_minus_one(mix, stem, guide_percent=0)

    assert float(np.max(np.abs(minus))) == pytest.approx(1.0)
    # Escala, no recorte: la forma se conserva.
    raw = mix.astype(np.float64) - stem.astype(np.float64)
    np.testing.assert_allclose(minus, raw / np.max(np.abs(raw)), atol=1e-12)


def test_clip_guard_never_boosts_a_quiet_result() -> None:
    stems = sine_stems()
    mix = mix_of(stems)

    minus = derive_minus_one(mix, stems["vocals"], guide_percent=0)

    # El resultado es silencioso y se queda asi: normalizarlo mentiria el nivel.
    expected = mix.astype(np.float64) - stems["vocals"].astype(np.float64)
    assert float(np.max(np.abs(expected))) < 1.0
    np.testing.assert_allclose(minus, expected, atol=1e-12)


def test_length_mismatch_trims_to_the_shortest() -> None:
    stems = sine_stems()
    mix = mix_of(stems)

    minus = derive_minus_one(mix, stems["other"][:-7], guide_percent=0)

    assert minus.shape[0] == mix.shape[0] - 7


def test_file_derivation_roundtrips_through_float_wavs(tmp_path: Path) -> None:
    stems = sine_stems()
    mix = mix_of(stems)
    mix_path = tmp_path / "mix.wav"
    stem_path = tmp_path / "drums.wav"
    sf.write(mix_path, mix, SAMPLE_RATE, subtype="FLOAT")
    sf.write(stem_path, stems["drums"], SAMPLE_RATE, subtype="FLOAT")
    destination = tmp_path / "minus_drums.wav"

    derive_minus_one_file(mix_path, stem_path, destination, guide_percent=0)

    written, rate = sf.read(destination, dtype="float32", always_2d=True)
    assert rate == SAMPLE_RATE
    expected = (mix.astype(np.float64) - stems["drums"].astype(np.float64)).astype(
        np.float32
    )
    np.testing.assert_allclose(written, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# Manager: validacion de la seleccion de practica
# ---------------------------------------------------------------------------


def install_fake_umx(settings) -> None:
    # `install_fake_model` escribe solo `filename`; umxhq necesita sus CUATRO
    # grafos en disco para contar como instalado.
    model_dir = settings.karaoke_model_dir_path
    model_dir.mkdir(parents=True, exist_ok=True)
    for archivo in SEPARATION_MODELS["umx_4stem"].files:
        (model_dir / archivo).write_bytes(b"fake-onnx")


async def test_practice_stems_require_separate(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="separate"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            practice_stems=["drums"],
        )


async def test_practice_stems_reject_a_two_stem_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    from tests.test_audio_karaoke import install_fake_model

    install_fake_model(settings, "inst_hq_3")
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="3 o mas stems"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="inst_hq_3",
            practice_stems=["vocals"],
        )


async def test_practice_stems_reject_an_unknown_stem_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="guitar"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="umx_4stem",
            practice_stems=["guitar"],
        )


async def test_practice_guide_percent_is_capped(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match=str(GUIDE_PERCENT_MAX)):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="song.wav",
            separate=True,
            separation_model="umx_4stem",
            practice_stems=["drums"],
            practice_guide_percent=GUIDE_PERCENT_MAX + 10,
        )


async def test_valid_practice_selection_lands_on_the_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        separate=True,
        separation_model="umx_4stem",
        practice_stems=["drums", "bass", "drums"],
        practice_guide_percent=20,
    )

    # Deduplicado preservando orden, como ensemble_models.
    assert job.practice_stems == ["drums", "bass"]
    assert job.practice_guide_percent == 20


# ---------------------------------------------------------------------------
# Pipeline: derivados en stem_output_paths con nombre {job.id}.minus_<stem>
# ---------------------------------------------------------------------------


class WavWritingSeparator(FakeSeparator):
    """Escribe wavs REALES por stem: la derivacion los lee con soundfile."""

    def __init__(self, stems: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.stem_audio = stems

    async def run(
        self,
        input_wav: Path,
        outputs: Sequence[Path],
        device: str,
        model_id: str = "umx_4stem",
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        outputs = tuple(outputs)
        self.calls.append((input_wav, outputs, device, model_id))
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, self.stem_audio[output.stem], SAMPLE_RATE, subtype="FLOAT")


class WavDecodingPipeline(CommandRecordingPipeline):
    """El decode fake escribe un wav REAL (la mezcla), no bytes de mentira."""

    def __init__(self, *args: object, mix: np.ndarray, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.mix = mix

    async def _run_process(self, command: list[str], failure_message: str) -> None:
        self.commands.append(command)
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".wav":
            sf.write(output_path, self.mix, SAMPLE_RATE, subtype="FLOAT")
        else:
            output_path.write_bytes(b"fake-audio-bytes")


def make_practice_job(source_path: Path, **overrides: object) -> AudioJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        separate=True,
        separation_model="umx_4stem",
        practice_stems=["drums"],
        output_format="wav",
    )
    fields.update(overrides)
    return AudioJob(**fields)


async def test_pipeline_derives_minus_one_next_to_the_stems(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    stems = sine_stems()
    separator = WavWritingSeparator(stems)
    pipeline = WavDecodingPipeline(settings, {}, {}, separators={"umx": separator}, mix=mix_of(stems))
    job = make_practice_job(write_upload_source(settings))

    output_path = await pipeline.run(job)

    assert set(job.stem_output_paths) == {"vocals", "drums", "bass", "other", "minus_drums"}
    minus = job.stem_output_paths["minus_drums"]
    assert minus.name == f"{job.id}.minus_drums.wav"
    assert minus.exists()
    # El principal sigue siendo el main stem del catalogo, no un derivado.
    assert output_path.name == f"{job.id}.vocals.wav"


async def test_pipeline_minus_one_subtracts_the_stem_from_the_decoded_mix(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    stems = sine_stems()
    mix = mix_of(stems)
    separator = WavWritingSeparator(stems)
    pipeline = WavDecodingPipeline(settings, {}, {}, separators={"umx": separator}, mix=mix)
    job = make_practice_job(write_upload_source(settings), practice_guide_percent=10)

    await pipeline.run(job)

    written, _ = sf.read(
        job.stem_output_paths["minus_drums"], dtype="float32", always_2d=True
    )
    expected = (
        mix.astype(np.float64) - 0.9 * stems["drums"].astype(np.float64)
    ).astype(np.float32)
    np.testing.assert_allclose(written, expected, atol=1e-7)


async def test_pipeline_without_practice_stems_derives_nothing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    stems = sine_stems()
    separator = WavWritingSeparator(stems)
    pipeline = WavDecodingPipeline(settings, {}, {}, separators={"umx": separator}, mix=mix_of(stems))
    job = make_practice_job(write_upload_source(settings), practice_stems=[])

    await pipeline.run(job)

    assert set(job.stem_output_paths) == {"vocals", "drums", "bass", "other"}


# ---------------------------------------------------------------------------
# API: form CSV, respuesta y descarga por stem derivado
# ---------------------------------------------------------------------------


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def test_create_route_parses_practice_fields(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_umx(settings)
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
        separation_model="umx_4stem",
        practice_stems="drums,bass",
        practice_guide_percent=20,
        audio_jobs=manager,
        storage=storage,
        settings=settings,
    )

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.practice_stems == ["drums", "bass"]
    assert job.practice_guide_percent == 20


async def test_create_route_rejects_practice_stems_on_a_two_stem_model(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    from tests.test_audio_karaoke import install_fake_model

    install_fake_model(settings, "inst_hq_3")
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
            separation_model="inst_hq_3",
            practice_stems="vocals",
            audio_jobs=manager,
            storage=storage,
            settings=settings,
        )
    assert excinfo.value.status_code == 400


def make_completed_practice_job(tmp_path: Path) -> AudioJob:
    job = make_practice_job(tmp_path / "song.wav", practice_guide_percent=10)
    job.status = JobStatus.completed
    outputs: dict[str, Path] = {}
    for stem_id in ("vocals", "drums", "bass", "other", "minus_drums"):
        path = tmp_path / f"{job.id}.{stem_id}.wav"
        path.write_bytes(stem_id.encode())
        outputs[stem_id] = path
    job.output_path = outputs["vocals"]
    job.stem_output_paths = outputs
    return job


def test_response_lists_the_minus_stem_download_and_echoes_the_selection(
    tmp_path: Path,
) -> None:
    job = make_completed_practice_job(tmp_path)

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    assert serialized["practiceStems"] == ["drums"]
    assert serialized["practiceGuidePercent"] == 10
    by_id = {stem["id"]: stem for stem in serialized["stems"]}
    assert by_id["minus_drums"]["labelKey"] == "audio.stem.minus_drums"
    assert by_id["minus_drums"]["url"] == (
        f"/api/v1/audio/jobs/{job.id}/download?stem=minus_drums"
    )


async def test_download_serves_a_requested_minus_stem(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = make_completed_practice_job(tmp_path)
    manager.jobs[job.id] = job

    response = await download_audio_job(job.id, stem="minus_drums", audio_jobs=manager)

    assert Path(response.path) == job.stem_output_paths["minus_drums"]


async def test_download_rejects_an_unrequested_minus_stem_with_400(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = make_completed_practice_job(tmp_path)
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="minus_bass", audio_jobs=manager)
    assert excinfo.value.status_code == 400
