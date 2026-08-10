from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.routes import audio_capabilities, audio_job_to_response, download_audio_job
from app.config import Settings
from app.models import JobStatus
from app.services.engines.separation_models import SEPARATION_MODELS
from app.services.engines.vr_deecho import MODEL_SPECS as DRIVER_MODEL_SPECS
from app.services.engines.vr_deecho import multiband
from app.services.engines.vr_deecho_separator import (
    VrDeEchoSeparator,
    block_ranges,
    combined_spec_frames,
    crossfade_window,
    graph_windows,
)
from app.services.engines.vr_models import VR_MODELS
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from tests.test_audio_karaoke import (
    FakeSeparator,
    make_manager,
    make_pipeline,
    make_separate_job,
    write_upload_source,
)

# ---------------------------------------------------------------------------
# v0.61.0 - los 3 modelos De-Echo/De-Reverb (arquitectura VR 5.1 CascadedNet,
# FoxJoy) en el grupo "Limpieza" del catalogo de separacion. El catalogo pasa a
# tener dos arquitecturas detras de UNA lista, el pipeline elige motor por la
# arquitectura del modelo del job, y el motor nuevo acota memoria por bloques
# reportando progreso y mirando la cancelacion en cada ventana del grafo.
# ---------------------------------------------------------------------------

SR = 44100


class FakeMaskSession:
    """Devuelve una mascara constante del shape de la entrada."""

    class _Meta:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [self._Meta("mag")]

    def get_outputs(self):
        return [self._Meta("mask")]

    def run(self, output_names, feeds):
        self.calls += 1
        window = feeds["mag"]
        self.shapes.append(tuple(window.shape))
        return [np.full(window.shape, self.value, dtype=np.float32)]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        KARAOKE_MODEL_DIR=str(tmp_path / "karaoke"),
        **overrides,
    )


def install_fake_model(settings: Settings, model_id: str) -> Path:
    model_dir = settings.karaoke_model_dir_path
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / SEPARATION_MODELS[model_id].filename
    path.write_bytes(b"fake-onnx")
    return path


def make_separator(
    settings: Settings, session: FakeMaskSession | None = None
) -> tuple[VrDeEchoSeparator, FakeMaskSession]:
    separator = VrDeEchoSeparator(settings, GpuSessionCoordinator())
    fake = session or FakeMaskSession()
    separator._create_session = lambda device, model_id: fake  # type: ignore[method-assign]
    return separator, fake


def write_stereo_wav(path: Path, samples: int) -> np.ndarray:
    import soundfile as sf

    t = np.arange(samples) / SR
    mix = np.stack(
        [0.4 * np.sin(2 * np.pi * 220.0 * t), 0.3 * np.sin(2 * np.pi * 330.0 * t)]
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mix.T, SR)
    return mix


def read_stereo_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    return sf.read(str(path), dtype="float32", always_2d=True)[0].T


# ---------------------------------------------------------------------------
# Catalogo: UNA lista, dos arquitecturas
# ---------------------------------------------------------------------------


def test_catalog_has_six_models_across_two_architectures() -> None:
    assert set(SEPARATION_MODELS) == {
        "inst_hq_3",
        "voc_ft",
        "reverb_hq",
        "deecho_normal",
        "deecho_aggressive",
        "deecho_dereverb",
    }
    architectures = {model_id: spec.architecture for model_id, spec in SEPARATION_MODELS.items()}
    assert architectures == {
        "inst_hq_3": "mdx",
        "voc_ft": "mdx",
        "reverb_hq": "mdx",
        "deecho_normal": "vr",
        "deecho_aggressive": "vr",
        "deecho_dereverb": "vr",
    }


def test_every_vr_model_is_a_cleanup_model_with_an_ordered_stem_pair() -> None:
    for spec in VR_MODELS.values():
        assert spec.category == "cleanup", spec.id
        # Estos modelos predicen el DRY directo: el stem que el usuario quiere
        # es la salida del modelo, no la resta.
        assert spec.main_stem.source == "primary", spec.id
        assert spec.other_stem.source == "secondary", spec.id
        assert len({stem.id for stem in spec.stems}) == 2, spec.id
        assert all(stem.label_key for stem in spec.stems), spec.id


def test_vr_stem_ids_name_what_the_user_gets() -> None:
    assert VR_MODELS["deecho_normal"].stem_ids() == ("no_echo", "echo")
    assert VR_MODELS["deecho_aggressive"].stem_ids() == ("no_echo", "echo")
    assert VR_MODELS["deecho_dereverb"].stem_ids() == ("no_reverb", "reverb")


def test_vr_catalog_matches_the_vendored_driver_specs() -> None:
    # Anti-drift contra app/services/engines/vr_deecho/: si un re-vendoreo
    # cambia nombres, primary_stem o nout, esto falla en vez de inferir mal.
    for spec in VR_MODELS.values():
        driver_spec = DRIVER_MODEL_SPECS[spec.vr_model_name]
        assert spec.primary_stem == driver_spec["primary_stem"], spec.id
        assert spec.nout == driver_spec["nout"], spec.id
        assert spec.filename == f"{spec.vr_model_name}.onnx", spec.id


def test_vr_models_pin_the_port_release_and_a_full_sha256() -> None:
    for spec in VR_MODELS.values():
        assert spec.url.startswith(
            "https://github.com/santiquiroz/port-uvr-deecho-onnx/releases/download/models-v1.0/"
        )
        assert len(spec.sha256) == 64
        # Procedencia del checkpoint de origen (hash UVR del .pth), no del onnx.
        assert len(spec.source_uvr_hash) == 32


# ---------------------------------------------------------------------------
# Geometria de bloques y conteo de ventanas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seconds", [1.0, 4.18, 10.0, 17.0])
def test_frame_and_window_counts_match_the_driver(seconds: float) -> None:
    # El motor cuenta ventanas por adelantado para reportar progreso; si el
    # driver cambia su ventaneo, este test lo detecta antes que el usuario.
    samples = int(SR * seconds)
    mix = np.zeros((2, samples), dtype=np.float32)

    spec = multiband.wave_to_combined_spec(mix)

    assert combined_spec_frames(samples) == spec.shape[2]


def test_block_ranges_cover_everything_and_overlap_by_the_fade() -> None:
    ranges = block_ranges(1000, 300, 60, 20)

    assert [keep for _, _, keep, _ in ranges][0] == 0
    assert ranges[-1][3] == 1000
    for previous, current in zip(ranges, ranges[1:]):
        # El bloque siguiente arranca `fade` antes de donde termina el anterior.
        assert current[2] == previous[3] - 20
        assert current[0] <= current[2] and current[1] >= current[3]


def test_a_file_shorter_than_a_block_runs_in_one_pass() -> None:
    # Ese camino es EXACTAMENTE el que el port valido contra audio-separator.
    assert block_ranges(1000, 5000, 60, 20) == [(0, 1000, 0, 1000)]


@pytest.mark.parametrize("samples", [SR, SR * 2 + 7, SR * 6, 88200, 264600])
def test_the_tail_pad_is_enough_to_cover_the_whole_block(samples: int) -> None:
    # Sin esta cola de silencio el ultimo bloque quedaba relleno con ceros (~8 ms
    # de corte seco al final del archivo). Gatea el valor de _TAIL_PAD_SAMPLES.
    from app.services.engines.vr_deecho.pipeline import DeEchoDriver
    from app.services.engines.vr_deecho_separator import _TAIL_PAD_SAMPLES

    driver = DeEchoDriver(lambda window: np.ones(window.shape, dtype=np.float32))
    # Señal real y no silencio: el driver normaliza por el maximo del
    # espectrograma y un bloque todo en cero divide por cero.
    tone = np.sin(2 * np.pi * 220.0 * np.arange(samples + _TAIL_PAD_SAMPLES) / SR)
    padded = np.stack([tone, tone]).astype(np.float32)

    primary, _ = driver.separate(padded)

    assert primary.shape[1] >= samples


def test_contiguous_crossfade_windows_sum_to_one() -> None:
    fade = 16
    falling = crossfade_window(100, 0, fade)[-fade:]
    rising = crossfade_window(100, fade, 0)[:fade]

    assert np.allclose(falling + rising, 1.0)


# ---------------------------------------------------------------------------
# Motor: mascara, stems, bloques, progreso y cancelacion
# ---------------------------------------------------------------------------


def _separate(
    separator: VrDeEchoSeparator, tmp_path: Path, samples: int, model_id: str, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mix = write_stereo_wav(tmp_path / "mix.wav", samples)
    main_path = tmp_path / "out" / "main.wav"
    other_path = tmp_path / "out" / "other.wav"
    asyncio.run(
        separator.run(tmp_path / "mix.wav", main_path, other_path, "cpu", model_id=model_id, **kwargs)
    )
    return mix, read_stereo_wav(main_path), read_stereo_wav(other_path)


def test_a_pass_through_mask_returns_the_mix_as_the_clean_stem(tmp_path: Path) -> None:
    # mascara = 1 -> el primario es la mezcla y el eco extraido es silencio;
    # el residuo que queda es el piso del round-trip multibanda del driver.
    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_normal")
    separator, _ = make_separator(settings, FakeMaskSession(1.0))

    mix, no_echo, echo = _separate(separator, tmp_path, SR * 6, "deecho_normal")

    assert no_echo.shape == mix.shape
    interior = slice(SR, -SR)
    assert np.max(np.abs(no_echo[:, interior] - mix[:, interior])) < 5e-3
    assert np.max(np.abs(echo)) == 0.0


def test_an_empty_mask_puts_everything_in_the_echo_stem(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_normal")
    separator, _ = make_separator(settings, FakeMaskSession(0.0))

    mix, no_echo, echo = _separate(separator, tmp_path, SR * 6, "deecho_normal")

    assert np.max(np.abs(no_echo)) == 0.0
    interior = slice(SR, -SR)
    assert np.max(np.abs(echo[:, interior] - mix[:, interior])) < 5e-3


def test_blocking_a_long_file_reconstructs_like_a_single_pass(tmp_path: Path) -> None:
    # La costura entre bloques se cruza con rampas complementarias: procesar por
    # bloques no puede cambiar el resultado de una pasada entera.
    whole = make_settings(tmp_path / "whole", VR_SEPARATION_BLOCK_SECONDS=60.0)
    blocked = make_settings(
        tmp_path / "blocked",
        VR_SEPARATION_BLOCK_SECONDS=4.0,
        VR_SEPARATION_MARGIN_SECONDS=1.0,
    )
    for settings in (whole, blocked):
        install_fake_model(settings, "deecho_normal")

    whole_separator, whole_session = make_separator(whole, FakeMaskSession(1.0))
    blocked_separator, blocked_session = make_separator(blocked, FakeMaskSession(1.0))
    _, whole_main, _ = _separate(whole_separator, tmp_path / "a", SR * 13, "deecho_normal")
    _, blocked_main, _ = _separate(blocked_separator, tmp_path / "b", SR * 13, "deecho_normal")

    assert blocked_session.calls > whole_session.calls  # de verdad hubo bloques
    assert np.max(np.abs(whole_main - blocked_main)) < 1e-3


def test_every_graph_call_gets_the_window_shape_the_models_expect(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_dereverb")
    separator, session = make_separator(settings, FakeMaskSession(0.5))

    _separate(separator, tmp_path, SR * 6, "deecho_dereverb")

    assert session.shapes and all(shape == (1, 2, 673, 512) for shape in session.shapes)


def test_progress_is_reported_once_per_window_and_ends_complete(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, VR_SEPARATION_BLOCK_SECONDS=4.0, VR_SEPARATION_MARGIN_SECONDS=1.0
    )
    install_fake_model(settings, "deecho_normal")
    separator, session = make_separator(settings, FakeMaskSession(1.0))
    seen: list[tuple[int, int]] = []

    _separate(
        separator,
        tmp_path,
        SR * 13,
        "deecho_normal",
        on_chunk=lambda done, total: seen.append((done, total)),
    )

    assert len(seen) == session.calls
    assert [done for done, _ in seen] == list(range(1, session.calls + 1))
    assert seen[-1] == (session.calls, session.calls)


def test_the_declared_window_total_matches_the_real_graph_calls(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, VR_SEPARATION_BLOCK_SECONDS=4.0, VR_SEPARATION_MARGIN_SECONDS=1.0
    )
    install_fake_model(settings, "deecho_normal")
    separator, session = make_separator(settings, FakeMaskSession(1.0))
    seen: list[tuple[int, int]] = []

    _separate(
        separator,
        tmp_path,
        SR * 13,
        "deecho_normal",
        on_chunk=lambda done, total: seen.append((done, total)),
    )

    from app.services.engines.vr_deecho_separator import _TAIL_PAD_SAMPLES

    expected = sum(
        graph_windows(hi - lo + _TAIL_PAD_SAMPLES)
        for lo, hi, _, _ in block_ranges(SR * 13, SR * 4, SR * 1, SR * 1)
    )
    assert session.calls == expected
    assert {total for _, total in seen} == {expected}


def test_cancelling_stops_between_windows_without_leaving_the_thread_running(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_normal")
    entered = threading.Event()
    release = threading.Event()

    class SlowSession(FakeMaskSession):
        def run(self, output_names, feeds):
            entered.set()
            release.wait(timeout=10)
            return super().run(output_names, feeds)

    separator, session = make_separator(settings, SlowSession(1.0))
    write_stereo_wav(tmp_path / "mix.wav", SR * 20)

    async def scenario() -> None:
        task = asyncio.create_task(
            separator.run(
                tmp_path / "mix.wav",
                tmp_path / "out" / "a.wav",
                tmp_path / "out" / "b.wav",
                "cpu",
                model_id="deecho_normal",
            )
        )
        await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    # Freno en la SIGUIENTE ventana: ni corrio todas ni escribio salidas.
    assert session.calls < graph_windows(SR * 20)
    assert not (tmp_path / "out" / "a.wav").exists()


def test_mono_input_is_duplicated_to_stereo(tmp_path: Path) -> None:
    import soundfile as sf

    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_normal")
    separator, _ = make_separator(settings, FakeMaskSession(1.0))
    samples = SR * 6
    mono = (0.3 * np.sin(2 * np.pi * 220.0 * np.arange(samples) / SR)).astype(np.float32)
    sf.write(str(tmp_path / "mono.wav"), mono, SR)

    asyncio.run(
        separator.run(
            tmp_path / "mono.wav",
            tmp_path / "out" / "a.wav",
            tmp_path / "out" / "b.wav",
            "cpu",
            model_id="deecho_normal",
        )
    )

    out = read_stereo_wav(tmp_path / "out" / "a.wav")
    assert out.shape == (2, samples)
    assert np.allclose(out[0], out[1])


def test_the_vr_engine_refuses_an_mdx_model_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator, _ = make_separator(settings)
    write_stereo_wav(tmp_path / "mix.wav", SR)

    with pytest.raises(RuntimeError, match="inst_hq_3"):
        asyncio.run(
            separator.run(
                tmp_path / "mix.wav",
                tmp_path / "a.wav",
                tmp_path / "b.wav",
                "cpu",
                model_id="inst_hq_3",
            )
        )


def test_the_vr_engine_reports_a_missing_model_as_a_missing_pack(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator, _ = make_separator(settings)
    write_stereo_wav(tmp_path / "mix.wav", SR)

    with pytest.raises(RuntimeError, match="deecho_dereverb"):
        asyncio.run(
            separator.run(
                tmp_path / "mix.wav",
                tmp_path / "a.wav",
                tmp_path / "b.wav",
                "cpu",
                model_id="deecho_dereverb",
            )
        )


def test_sessions_are_cached_per_device_and_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_normal")
    separator, _ = make_separator(settings)

    first = separator._get_session("cpu", "deecho_normal")
    assert separator._get_session("cpu", "deecho_normal") is first
    separator.release_device("cpu")
    assert separator._session_cache == {}


# ---------------------------------------------------------------------------
# Pipeline: el motor lo elige la ARQUITECTURA del modelo del job
# ---------------------------------------------------------------------------


async def test_pipeline_routes_a_vr_model_to_the_vr_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    mdx, vr = FakeSeparator(), FakeSeparator()
    pipeline = make_pipeline(settings, mdx)
    pipeline.separators["vr"] = vr
    job = make_separate_job(
        write_upload_source(settings), separation_model="deecho_normal"
    )

    output_path = await pipeline.run(job)

    assert vr.calls and not mdx.calls
    assert vr.calls[0][4] == "deecho_normal"
    assert output_path.name == f"{job.id}.no_echo.flac"
    assert job.secondary_output_path is not None
    assert job.secondary_output_path.name == f"{job.id}.echo.flac"


async def test_pipeline_still_routes_an_mdx_model_to_the_mdx_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    mdx, vr = FakeSeparator(), FakeSeparator()
    pipeline = make_pipeline(settings, mdx)
    pipeline.separators["vr"] = vr
    job = make_separate_job(write_upload_source(settings), separation_model="reverb_hq")

    await pipeline.run(job)

    assert mdx.calls and not vr.calls


async def test_pipeline_names_dereverb_outputs_no_reverb_and_reverb(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator(), architecture="vr")
    job = make_separate_job(
        write_upload_source(settings), separation_model="deecho_dereverb"
    )

    output_path = await pipeline.run(job)

    assert output_path.name == f"{job.id}.no_reverb.flac"
    assert job.secondary_output_path is not None
    assert job.secondary_output_path.name == f"{job.id}.reverb.flac"


async def test_a_vr_job_without_a_vr_engine_fails_loudly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())  # solo mdx
    job = make_separate_job(
        write_upload_source(settings), separation_model="deecho_normal"
    )

    with pytest.raises(RuntimeError, match="arquitectura vr"):
        await pipeline.run(job)


async def test_manager_accepts_a_vr_model_when_installed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_aggressive")
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="voz.wav",
        separate=True,
        separation_model="deecho_aggressive",
    )

    assert job.separation_model == "deecho_aggressive"


# ---------------------------------------------------------------------------
# API: capabilities, stems y validacion cruzada de ?stem=
# ---------------------------------------------------------------------------


async def test_capabilities_expose_one_list_with_the_architecture(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "deecho_normal")

    response = await audio_capabilities(settings=settings)
    by_id = {model.id: model for model in response.separation_models}

    assert len(response.separation_models) == 6
    assert by_id["deecho_normal"].architecture == "vr"
    assert by_id["deecho_normal"].installed is True
    assert by_id["deecho_aggressive"].installed is False
    assert by_id["inst_hq_3"].architecture == "mdx"
    # Los tres nuevos caen en el grupo Limpieza junto a reverb_hq.
    cleanup = [model.id for model in response.separation_models if model.category == "cleanup"]
    assert cleanup == ["reverb_hq", "deecho_normal", "deecho_aggressive", "deecho_dereverb"]


def _completed_vr_job(tmp_path: Path, model_id: str, main: str, other: str):
    job = make_separate_job(tmp_path / "voz.wav", separation_model=model_id)
    job.status = JobStatus.completed
    job.output_path = tmp_path / f"{job.id}.{main}.flac"
    job.output_path.write_bytes(b"main")
    job.secondary_output_path = tmp_path / f"{job.id}.{other}.flac"
    job.secondary_output_path.write_bytes(b"other")
    return job


def test_a_deecho_job_lists_the_clean_stem_first_and_has_no_vocals_url(
    tmp_path: Path,
) -> None:
    job = _completed_vr_job(tmp_path, "deecho_normal", "no_echo", "echo")

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    base = f"/api/v1/audio/jobs/{job.id}/download"
    assert serialized["downloadUrl"] == base
    assert serialized["stems"] == [
        {"id": "no_echo", "labelKey": "audio.stem.no_echo", "url": f"{base}?stem=no_echo"},
        {"id": "echo", "labelKey": "audio.stem.echo", "url": f"{base}?stem=echo"},
    ]
    assert serialized["vocalsDownloadUrl"] is None


async def test_download_serves_each_deecho_stem_by_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = _completed_vr_job(tmp_path, "deecho_dereverb", "no_reverb", "reverb")
    manager.jobs[job.id] = job

    clean = await download_audio_job(job.id, stem="no_reverb", audio_jobs=manager)
    tail = await download_audio_job(job.id, stem="reverb", audio_jobs=manager)
    default = await download_audio_job(job.id, audio_jobs=manager)

    assert Path(clean.path) == job.output_path
    assert Path(tail.path) == job.secondary_output_path
    assert Path(default.path) == job.output_path


@pytest.mark.parametrize(
    ("model_id", "main", "other", "wrong_stem"),
    [
        ("deecho_normal", "no_echo", "echo", "dry"),
        ("deecho_dereverb", "no_reverb", "reverb", "no_echo"),
    ],
)
async def test_download_rejects_a_stem_from_another_model(
    tmp_path: Path, model_id: str, main: str, other: str, wrong_stem: str
) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = _completed_vr_job(tmp_path, model_id, main, other)
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem=wrong_stem, audio_jobs=manager)

    assert excinfo.value.status_code == 400
    assert main in str(excinfo.value.detail) and other in str(excinfo.value.detail)


async def test_download_rejects_a_deecho_stem_on_a_karaoke_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    job = _completed_vr_job(tmp_path, "inst_hq_3", "instrumental", "vocals")
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as excinfo:
        await download_audio_job(job.id, stem="no_echo", audio_jobs=manager)

    assert excinfo.value.status_code == 400


# ---------------------------------------------------------------------------
# Pack: los ids nuevos son variantes validas del pack karaoke
# ---------------------------------------------------------------------------


def test_the_provisioner_accepts_every_vr_variant() -> None:
    from app.services.pack_provisioner import build_command

    for model_id in VR_MODELS:
        command = build_command("karaoke", model_id)
        assert command[-2:] == ["-Model", model_id]
