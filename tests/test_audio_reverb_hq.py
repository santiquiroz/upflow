from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.routes import audio_job_to_response, download_audio_job
from app.models import JobStatus
from app.services.engines.separation_models import SEPARATION_MODELS
from app.services.engines.separation_spec import RESIDUAL
from app.services.engines.mdx_separator import MdxSeparator
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from tests.test_audio_karaoke import (
    FakeSeparator,
    install_fake_model,
    make_manager,
    make_pipeline,
    make_separate_job,
    make_settings,
    write_upload_source,
)
from tests.test_mdx_separator import (
    FakeIdentitySession,
    interior,
    si_sdr,
    write_stereo_wav,
)

# ---------------------------------------------------------------------------
# v0.60 - Reverb HQ (de-reverb, FoxJoy) en el catalogo de separacion: la pasada
# de limpieza post-karaoke. El modelo saca el stem WET (Reverb); el que el
# usuario quiere es la resta (dry), asi que el catalogo declara los stems por
# modelo con labels y orden (primero = downloadUrl) y la API valida los ids de
# stem contra el modelo del job. Datos duros del spike 2026-08-09
# (spike-deecho/reverb_hq_test.py).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Catalogo
# ---------------------------------------------------------------------------


def test_catalog_still_has_the_three_mdx_models() -> None:
    from app.services.engines.mdx_models import MDX_MODELS

    assert set(MDX_MODELS) == {"inst_hq_3", "voc_ft", "reverb_hq"}


def test_reverb_hq_spec_matches_the_validated_spike() -> None:
    spec = SEPARATION_MODELS["reverb_hq"]

    assert spec.filename == "Reverb_HQ_By_FoxJoy.onnx"
    assert spec.uvr_hash == "cd5b2989ad863f116c855db1dfe24e39"
    assert spec.sha256 == (
        "233bb5c6aaa365e568659a0a81211746fa881f8f47f82d9e864fce1f7692db80"
    )
    assert (spec.n_fft, spec.dim_f, spec.dim_t) == (6144, 3072, 512)
    assert spec.compensate == 1.035
    assert spec.primary_stem == "Reverb"
    assert spec.category == "cleanup"


def test_every_spec_maps_each_stem_to_a_distinct_source() -> None:
    # El numero de stems lo pone cada modelo (dos en karaoke y limpieza,
    # cuatro en los multi-stem); lo que NO puede pasar en ninguno es que dos
    # stems compartan origen, porque uno de los dos seria inalcanzable.
    for spec in SEPARATION_MODELS.values():
        sources = [stem.source for stem in spec.stems]
        assert len(set(sources)) == len(sources), spec.id
        assert len({stem.id for stem in spec.stems}) == len(spec.stems), spec.id
        assert all(stem.label_key for stem in spec.stems), spec.id
        # Cada indice tiene que apuntar a una salida que exista. El ORDEN no
        # se exige: el catalogo ordena por lo que el usuario quiere primero, y
        # denoise invierte el par a proposito.
        indices = [source for source in sources if source != RESIDUAL]
        assert all(0 <= index < len(sources) for index in indices), spec.id


def test_karaoke_models_put_the_instrumental_first() -> None:
    # El orden ES el contrato: stems[0] es lo que downloadUrl sirve.
    for model_id in ("inst_hq_3", "voc_ft"):
        assert SEPARATION_MODELS[model_id].stem_ids() == ("instrumental", "vocals")


def test_reverb_hq_puts_the_dry_stem_first_and_wet_is_the_model_output() -> None:
    spec = SEPARATION_MODELS["reverb_hq"]

    assert spec.stem_ids() == ("dry", "wet")
    assert spec.main_stem.source == RESIDUAL  # dry = mezcla - wet * 1.035
    assert spec.stems[1].source == 0


# ---------------------------------------------------------------------------
# Motor: primary "Reverb" -> la resta compensada va PRIMERA (invertida)
# ---------------------------------------------------------------------------


def _make_separator(settings) -> MdxSeparator:
    separator = MdxSeparator(settings, GpuSessionCoordinator())
    fake = FakeIdentitySession()
    separator._create_session = lambda device, model_id: fake  # type: ignore[method-assign]
    return separator


def test_reverb_identity_model_puts_wet_second_and_dry_is_the_residual(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "reverb_hq")
    separator = _make_separator(settings)
    spec = SEPARATION_MODELS["reverb_hq"]
    mix = write_stereo_wav(tmp_path / "mix.wav", spec.gen_samples // 2)

    dry_path = tmp_path / "out" / "dry.wav"
    wet_path = tmp_path / "out" / "wet.wav"
    asyncio.run(
        separator.run(
            tmp_path / "mix.wav", (dry_path, wet_path), "cpu", model_id="reverb_hq"
        )
    )

    import soundfile as sf

    dry = sf.read(str(dry_path), dtype="float32", always_2d=True)[0].T
    wet = sf.read(str(wet_path), dtype="float32", always_2d=True)[0].T
    # Con el modelo identidad el primario (wet) es la mezcla misma...
    assert si_sdr(wet, mix) > 40.0
    assert np.allclose(interior(wet, spec.n_fft), interior(mix, spec.n_fft), atol=2e-4)
    # ...y el PRIMER archivo (el que el usuario quiere) es la resta invertida.
    assert np.allclose(dry, mix - wet * spec.compensate, atol=1e-4)


# ---------------------------------------------------------------------------
# Pipeline: salidas nombradas por stem del modelo
# ---------------------------------------------------------------------------


async def test_pipeline_names_reverb_outputs_dry_and_wet(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())
    job = make_separate_job(
        write_upload_source(settings), separation_model="reverb_hq", output_format="flac"
    )

    output_path = await pipeline.run(job)

    assert output_path.name == f"{job.id}.dry.flac"
    assert set(job.stem_output_paths) == {"dry", "wet"}
    assert job.stem_output_paths["wet"].name == f"{job.id}.wet.flac"
    assert job.stem_output_paths["dry"] == output_path


async def test_manager_accepts_reverb_hq_when_installed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "reverb_hq")
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="instrumental.flac",
        separate=True,
        separation_model="reverb_hq",
    )

    assert job.separation_model == "reverb_hq"


# ---------------------------------------------------------------------------
# API: respuesta del job con stems ordenados y compat vocalsDownloadUrl
# ---------------------------------------------------------------------------


def _completed_reverb_job(tmp_path: Path):
    job = make_separate_job(tmp_path / "song.wav", separation_model="reverb_hq")
    job.status = JobStatus.completed
    job.output_path = tmp_path / f"{job.id}.dry.flac"
    job.output_path.write_bytes(b"dry")
    wet = tmp_path / f"{job.id}.wet.flac"
    wet.write_bytes(b"wet")
    job.stem_output_paths = {"dry": job.output_path, "wet": wet}
    return job


def test_reverb_response_lists_dry_first_and_has_no_vocals_url(tmp_path: Path) -> None:
    job = _completed_reverb_job(tmp_path)

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    base = f"/api/v1/audio/jobs/{job.id}/download"
    assert serialized["downloadUrl"] == base
    assert serialized["stems"] == [
        {"id": "dry", "labelKey": "audio.stem.dry", "url": f"{base}?stem=dry"},
        {"id": "wet", "labelKey": "audio.stem.wet", "url": f"{base}?stem=wet"},
    ]
    # reverb_hq no tiene voz que ofrecer: el campo compat queda vacio.
    assert serialized["vocalsDownloadUrl"] is None


def test_karaoke_response_keeps_vocals_url_and_gains_stems(tmp_path: Path) -> None:
    job = make_separate_job(tmp_path / "song.wav")  # inst_hq_3
    job.status = JobStatus.completed
    job.output_path = tmp_path / "instrumental.flac"
    job.stem_output_paths = {
        "instrumental": job.output_path,
        "vocals": tmp_path / "vocals.flac",
    }

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    base = f"/api/v1/audio/jobs/{job.id}/download"
    assert serialized["vocalsDownloadUrl"] == f"{base}?stem=vocals"
    assert [stem["id"] for stem in serialized["stems"]] == ["instrumental", "vocals"]
    assert [stem["labelKey"] for stem in serialized["stems"]] == [
        "audio.stem.instrumental",
        "audio.stem.vocals",
    ]


def test_incomplete_separation_job_has_no_stems(tmp_path: Path) -> None:
    job = make_separate_job(tmp_path / "song.wav", separation_model="reverb_hq")

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    assert serialized["stems"] is None
    assert serialized["vocalsDownloadUrl"] is None


# ---------------------------------------------------------------------------
# API: descarga con ids de stem validados contra el modelo del job
# ---------------------------------------------------------------------------


def _manager_with_reverb_job(tmp_path: Path):
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = _completed_reverb_job(tmp_path)
    manager.jobs[job.id] = job
    return manager, job


async def test_download_default_serves_the_dry_stem(tmp_path: Path) -> None:
    manager, job = _manager_with_reverb_job(tmp_path)

    response = await download_audio_job(job.id, audio_jobs=manager)

    assert Path(response.path) == job.output_path


async def test_download_serves_each_reverb_stem_by_id(tmp_path: Path) -> None:
    manager, job = _manager_with_reverb_job(tmp_path)

    dry = await download_audio_job(job.id, stem="dry", audio_jobs=manager)
    wet = await download_audio_job(job.id, stem="wet", audio_jobs=manager)

    assert Path(dry.path) == job.output_path
    assert Path(wet.path) == job.stem_output_paths["wet"]


async def test_download_rejects_a_karaoke_stem_on_a_reverb_job(tmp_path: Path) -> None:
    manager, job = _manager_with_reverb_job(tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="vocals", audio_jobs=manager)

    assert excinfo.value.status_code == 400
    assert "dry" in str(excinfo.value.detail) and "wet" in str(excinfo.value.detail)


async def test_download_rejects_a_reverb_stem_on_a_karaoke_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = make_separate_job(tmp_path / "song.wav")  # inst_hq_3
    job.status = JobStatus.completed
    job.output_path = tmp_path / "instrumental.flac"
    job.output_path.write_bytes(b"instrumental")
    vocals = tmp_path / "vocals.flac"
    vocals.write_bytes(b"vocals")
    job.stem_output_paths = {"instrumental": job.output_path, "vocals": vocals}
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="dry", audio_jobs=manager)

    assert excinfo.value.status_code == 400
    assert "instrumental" in str(excinfo.value.detail)


# ---------------------------------------------------------------------------
# Capabilities: el catalogo viaja con categoria, descripcion y stems
# ---------------------------------------------------------------------------


async def test_capabilities_expose_labels_category_and_stem_order(tmp_path: Path) -> None:
    from app.api.routes import audio_capabilities

    settings = make_settings(tmp_path)
    install_fake_model(settings, "reverb_hq")

    response = await audio_capabilities(settings=settings)
    by_id = {model.id: model for model in response.separation_models}

    assert {"inst_hq_3", "voc_ft", "reverb_hq"} <= set(by_id)
    reverb = by_id["reverb_hq"]
    assert reverb.installed is True
    assert reverb.category == "cleanup"
    assert reverb.description_key == "audio.karaoke.model.reverb_hq.description"
    assert [(stem.id, stem.label_key) for stem in reverb.stems] == [
        ("dry", "audio.stem.dry"),
        ("wet", "audio.stem.wet"),
    ]
    assert by_id["inst_hq_3"].category == "karaoke"
    assert [stem.id for stem in by_id["inst_hq_3"].stems] == ["instrumental", "vocals"]
    assert by_id["voc_ft"].installed is False
