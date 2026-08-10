from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from app.api.routes import audio_capabilities
from app.config import Settings
from app.services.engines.roformer import RoformerSpec
from app.services.engines.roformer.chunking import plan_chunks
from app.services.engines.roformer_models import ROFORMER_MODELS
from app.services.engines.roformer_separator import (
    RoformerSeparator,
    driver_spec,
    graph_chunks,
    stems_from_vocals,
)
from app.services.engines.separation_models import (
    DEFAULT_SEPARATION_MODEL,
    SEPARATION_MODELS,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from tests.test_audio_karaoke import (
    FakeSeparator,
    make_manager,
    make_pipeline,
    make_separate_job,
    write_upload_source,
)

# ---------------------------------------------------------------------------
# Mel-Band RoFormer: el carril de MAXIMA CALIDAD del catalogo de separacion.
# Tercera arquitectura detras de la misma lista unica, con tres cosas que no
# existen en MDX ni en VR y que por eso se gatean aca:
#
#   * el stem residual es una resta SIMPLE, sin `compensate`;
#   * el chunk es FIJO (el rotary embedding hornea el eje temporal al trazar);
#   * pide ~2,3 GB libres, asi que hay admision de memoria antes de cargar.
# ---------------------------------------------------------------------------

SR = 44100
MODEL_ID = "mel_band_roformer_kim"
SPEC = ROFORMER_MODELS[MODEL_ID]


class FakeMaskSession:
    """Devuelve una mascara compleja constante con el shape que el grafo emite.

    spec [1, F*C, T, 2] -> mask [1, N, F*C, T, 2]. `real`/`imag` son la mascara
    compleja: (1, 0) es identidad y deja pasar la mezcla entera.
    """

    class _Meta:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self, real: float = 1.0, imag: float = 0.0, stems: int = 1) -> None:
        self.real = real
        self.imag = imag
        self.stems = stems
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def get_inputs(self):
        return [self._Meta("spec")]

    def get_outputs(self):
        return [self._Meta("mask")]

    def run(self, output_names, feeds):
        self.calls += 1
        spec_input = feeds["spec"]
        self.shapes.append(tuple(spec_input.shape))
        mask = np.empty((1, self.stems, *spec_input.shape[1:]), dtype=np.float32)
        mask[..., 0] = self.real
        mask[..., 1] = self.imag
        return [mask]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        KARAOKE_MODEL_DIR=str(tmp_path / "karaoke"),
        **overrides,
    )


def install_fake_model(settings: Settings, model_id: str = MODEL_ID) -> Path:
    model_dir = settings.karaoke_model_dir_path
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / SEPARATION_MODELS[model_id].filename
    path.write_bytes(b"fake-onnx")
    return path


def make_separator(
    settings: Settings, session: FakeMaskSession | None = None
) -> tuple[RoformerSeparator, FakeMaskSession]:
    separator = RoformerSeparator(settings, GpuSessionCoordinator())
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


def _separate(
    separator: RoformerSeparator, tmp_path: Path, samples: int, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mix = write_stereo_wav(tmp_path / "mix.wav", samples)
    main_path = tmp_path / "out" / "main.wav"
    other_path = tmp_path / "out" / "other.wav"
    asyncio.run(
        separator.run(
            tmp_path / "mix.wav", main_path, other_path, "cpu", model_id=MODEL_ID, **kwargs
        )
    )
    return mix, read_stereo_wav(main_path), read_stereo_wav(other_path)


# ---------------------------------------------------------------------------
# Catalogo: tercera arquitectura, misma lista
# ---------------------------------------------------------------------------


def test_the_roformer_model_joins_the_karaoke_group_without_becoming_the_default() -> None:
    # El carril caro se OFRECE, no se impone: cambiar el default a este modelo
    # multiplicaria por ~9 el tiempo de todo trabajo que hoy no elige nada.
    assert DEFAULT_SEPARATION_MODEL == "inst_hq_3"
    assert SPEC.category == "karaoke"
    karaoke = [spec.id for spec in SEPARATION_MODELS.values() if spec.category == "karaoke"]
    assert karaoke == ["inst_hq_3", "voc_ft", MODEL_ID]


def test_the_roformer_entry_declares_its_own_architecture() -> None:
    assert SPEC.architecture == "roformer"
    assert SEPARATION_MODELS[MODEL_ID] is SPEC


def test_the_user_gets_the_instrumental_first_and_the_vocals_second() -> None:
    # El modelo infiere la VOZ, pero en karaoke lo que el usuario se lleva
    # primero es la instrumental — mismo par que voc_ft, mismo motivo.
    assert SPEC.stem_ids() == ("instrumental", "vocals")
    assert SPEC.main_stem.source == "secondary"
    assert SPEC.other_stem.source == "primary"
    assert {stem.source for stem in SPEC.stems} == {"primary", "secondary"}
    assert all(stem.label_key for stem in SPEC.stems)


def test_the_roformer_entry_pins_the_port_release_and_a_full_sha256() -> None:
    assert SPEC.url.startswith(
        "https://github.com/santiquiroz/port-bs-roformer-onnx/releases/download/models-v1.0/"
    )
    assert SPEC.url.endswith(SPEC.filename)
    assert len(SPEC.sha256) == 64
    # El hash UVR no aplica: no es un modelo del catalogo de UVR.
    assert not hasattr(SPEC, "uvr_hash")


def test_only_the_slow_lane_carries_a_warning() -> None:
    # La advertencia es lo que la UI muestra ANTES de elegir. Si algun dia otro
    # modelo la necesita, este test dice donde actualizar la expectativa.
    warned = [spec.id for spec in SEPARATION_MODELS.values() if spec.warning_key]
    assert warned == [MODEL_ID]
    assert SPEC.warning_key == "audio.karaoke.model.mel_band_roformer_kim.warning"


def test_the_catalog_numbers_match_the_vendored_driver_defaults() -> None:
    # Anti-drift contra app/services/engines/roformer/: el eje temporal esta
    # HORNEADO en el grafo, asi que un numero que se mueva no da error, da
    # audio mal reconstruido.
    defaults = RoformerSpec()
    built = driver_spec(SPEC)

    assert (built.n_fft, built.hop_length) == (defaults.n_fft, defaults.hop_length)
    assert built.chunk_size == defaults.chunk_size
    assert built.num_overlap == defaults.num_overlap
    assert built.stems == defaults.stems == ("vocals",)
    assert built.frames == SPEC.frames == 801  # T del grafo publicado
    assert built.audio_channels == 2
    assert built.sample_rate == SR


def test_the_declared_memory_budget_covers_graph_plus_intermediate() -> None:
    assert SPEC.required_free_mb == SPEC.graph_mb + SPEC.intermediate_mb
    assert SPEC.chunk_seconds == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Resta SIMPLE: la diferencia que separa este motor de MdxSeparator
# ---------------------------------------------------------------------------


def test_the_residual_is_a_plain_subtraction_with_no_compensate_factor() -> None:
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, 512))
    vocals = rng.standard_normal((2, 512))

    instrumental, returned_vocals = stems_from_vocals(mix, vocals, SPEC)

    assert np.array_equal(returned_vocals, vocals)
    # Exacto: mezcla - voz. Cualquier factor rompe la igualdad bit a bit.
    assert np.array_equal(instrumental, mix - vocals)
    assert np.allclose(instrumental + returned_vocals, mix)


def test_no_mdx_style_compensate_leaked_into_the_roformer_spec() -> None:
    assert not hasattr(SPEC, "compensate")


def test_the_two_stems_add_back_up_to_the_mix(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator, _ = make_separator(settings, FakeMaskSession(0.5, 0.0))

    mix, instrumental, vocals = _separate(separator, tmp_path, SR * 6)

    assert instrumental.shape == vocals.shape == mix.shape
    assert np.max(np.abs(instrumental + vocals - mix)) < 1e-3


def test_an_identity_mask_puts_the_whole_mix_in_the_vocals_stem(tmp_path: Path) -> None:
    # mascara = 1+0j -> el stem inferido es la mezcla y el residual es silencio.
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator, _ = make_separator(settings, FakeMaskSession(1.0, 0.0))

    mix, instrumental, vocals = _separate(separator, tmp_path, SR * 6)

    interior = slice(SR, -SR)
    assert np.max(np.abs(vocals[:, interior] - mix[:, interior])) < 5e-3
    assert np.max(np.abs(instrumental[:, interior])) < 5e-3


def test_an_empty_mask_puts_the_whole_mix_in_the_instrumental_stem(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator, _ = make_separator(settings, FakeMaskSession(0.0, 0.0))

    mix, instrumental, vocals = _separate(separator, tmp_path, SR * 6)

    assert np.max(np.abs(vocals)) == 0.0
    interior = slice(SR, -SR)
    assert np.max(np.abs(instrumental[:, interior] - mix[:, interior])) < 5e-3


# ---------------------------------------------------------------------------
# Chunk fijo, bloques, progreso y cancelacion
# ---------------------------------------------------------------------------


def test_every_graph_call_gets_the_fixed_shape_the_export_baked_in(tmp_path: Path) -> None:
    # T=801 no es negociable: el rotary embedding cacheo su tabla por seq_len.
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator, session = make_separator(settings, FakeMaskSession(0.5, 0.1))

    _separate(separator, tmp_path, SR * 6)

    assert session.shapes and all(shape == (1, 2050, 801, 2) for shape in session.shapes)


def test_the_chunk_count_comes_from_the_drivers_own_planner() -> None:
    for seconds in (1, 8, 12, 30):
        samples = SR * seconds
        assert graph_chunks(samples, SPEC) == plan_chunks(
            samples, SPEC.chunk_size, SPEC.num_overlap
        ).num_chunks


def test_a_file_shorter_than_a_block_runs_in_one_driver_pass(tmp_path: Path) -> None:
    # El camino de una sola pasada es EXACTAMENTE el que el port valido.
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator, session = make_separator(settings, FakeMaskSession(0.5))

    _separate(separator, tmp_path, SR * 12)

    assert session.calls == graph_chunks(SR * 12, SPEC)


def test_blocking_a_long_file_reconstructs_like_a_single_pass(tmp_path: Path) -> None:
    # La costura entre bloques se cruza con rampas complementarias: trocear no
    # puede cambiar el resultado de una pasada entera.
    whole = make_settings(tmp_path / "whole")
    blocked = make_settings(
        tmp_path / "blocked",
        ROFORMER_SEPARATION_BLOCK_SECONDS=16.0,
        ROFORMER_SEPARATION_MARGIN_SECONDS=8.0,
    )
    for settings in (whole, blocked):
        install_fake_model(settings)

    whole_separator, whole_session = make_separator(whole, FakeMaskSession(0.6, 0.2))
    blocked_separator, blocked_session = make_separator(blocked, FakeMaskSession(0.6, 0.2))
    _, whole_main, _ = _separate(whole_separator, tmp_path / "a", SR * 40)
    _, blocked_main, _ = _separate(blocked_separator, tmp_path / "b", SR * 40)

    assert blocked_session.calls > whole_session.calls  # de verdad hubo bloques
    assert np.max(np.abs(whole_main - blocked_main)) < 1e-3


def test_progress_is_reported_once_per_chunk_and_ends_complete(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        ROFORMER_SEPARATION_BLOCK_SECONDS=16.0,
        ROFORMER_SEPARATION_MARGIN_SECONDS=8.0,
    )
    install_fake_model(settings)
    separator, session = make_separator(settings, FakeMaskSession(0.5))
    seen: list[tuple[int, int]] = []

    _separate(separator, tmp_path, SR * 40, on_chunk=lambda d, t: seen.append((d, t)))

    assert len(seen) == session.calls
    # Un solo contador para el archivo entero: el `on_chunk` del driver se
    # reinicia por bloque y mostraria el porcentaje volviendo para atras.
    assert [done for done, _ in seen] == list(range(1, session.calls + 1))
    assert seen[-1] == (session.calls, session.calls)


def test_the_declared_chunk_total_matches_the_real_graph_calls(tmp_path: Path) -> None:
    from app.services.engines.separation_blocks import block_ranges

    settings = make_settings(
        tmp_path,
        ROFORMER_SEPARATION_BLOCK_SECONDS=16.0,
        ROFORMER_SEPARATION_MARGIN_SECONDS=8.0,
    )
    install_fake_model(settings)
    separator, session = make_separator(settings, FakeMaskSession(0.5))
    seen: list[tuple[int, int]] = []

    _separate(separator, tmp_path, SR * 40, on_chunk=lambda d, t: seen.append((d, t)))

    expected = sum(
        graph_chunks(hi - lo, SPEC)
        for lo, hi, _, _ in block_ranges(SR * 40, SR * 16, SR * 8, SR * 8)
    )
    assert session.calls == expected
    assert {total for _, total in seen} == {expected}


def test_cancelling_stops_between_chunks_without_leaving_the_thread_running(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    entered = threading.Event()
    release = threading.Event()

    class SlowSession(FakeMaskSession):
        def run(self, output_names, feeds):
            entered.set()
            release.wait(timeout=10)
            return super().run(output_names, feeds)

    separator, session = make_separator(settings, SlowSession(0.5))
    write_stereo_wav(tmp_path / "mix.wav", SR * 40)

    async def scenario() -> None:
        task = asyncio.create_task(
            separator.run(
                tmp_path / "mix.wav",
                tmp_path / "out" / "a.wav",
                tmp_path / "out" / "b.wav",
                "cpu",
                model_id=MODEL_ID,
            )
        )
        await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    # Freno en el SIGUIENTE chunk: ni corrio todos ni escribio salidas.
    assert session.calls < graph_chunks(SR * 40, SPEC)
    assert not (tmp_path / "out" / "a.wav").exists()


def test_mono_input_is_duplicated_to_stereo(tmp_path: Path) -> None:
    import soundfile as sf

    settings = make_settings(tmp_path)
    install_fake_model(settings)
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
            model_id=MODEL_ID,
        )
    )

    out = read_stereo_wav(tmp_path / "out" / "a.wav")
    assert out.shape == (2, samples)
    assert np.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# Admision de memoria: 931 MB de grafo + ~1,3 GB de intermedio
# ---------------------------------------------------------------------------


def test_loading_is_refused_with_an_actionable_message_when_memory_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.engines.roformer_separator as module

    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator = RoformerSeparator(settings, GpuSessionCoordinator())
    monkeypatch.setattr(module, "free_memory_mb", lambda device: 900)

    with pytest.raises(RuntimeError) as excinfo:
        separator._create_session("dml:0", MODEL_ID)

    message = str(excinfo.value)
    assert "900 MB" in message
    assert str(SPEC.required_free_mb) in message
    # Accionable: dice que hacer, no solo que fallo.
    assert "Inst HQ 3" in message


def test_an_unmeasurable_device_fails_open_like_the_shared_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.engines.roformer_separator as module

    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator = RoformerSeparator(settings, GpuSessionCoordinator())
    monkeypatch.setattr(module, "free_memory_mb", lambda device: None)
    # El preflight no debe bloquear; que la carga real falle despues es otra
    # cosa (el .onnx de este test es un archivo trucho).
    monkeypatch.setattr(
        RoformerSeparator, "_require_free_memory", RoformerSeparator._require_free_memory
    )

    separator._require_free_memory("dml:0", SPEC)  # no levanta


def test_enough_free_memory_admits_the_load(tmp_path: Path, monkeypatch) -> None:
    import app.services.engines.roformer_separator as module

    settings = make_settings(tmp_path)
    separator = RoformerSeparator(settings, GpuSessionCoordinator())
    monkeypatch.setattr(module, "free_memory_mb", lambda device: SPEC.required_free_mb)

    separator._require_free_memory("dml:0", SPEC)  # no levanta


def test_the_previous_session_for_the_device_is_dropped_before_building_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Con grafos de 931 MB el solapamiento de dos sesiones es justo el pico que
    # hay que evitar; la base recicla su LRU recien DESPUES de construir.
    import app.services.engines.roformer_separator as module

    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator = RoformerSeparator(settings, GpuSessionCoordinator())
    monkeypatch.setattr(module, "free_memory_mb", lambda device: None)
    stale = FakeMaskSession()
    separator._session_cache["dml:0::otro_modelo"] = stale
    monkeypatch.setattr(
        RoformerSeparator.__mro__[1], "_create_session", lambda self, d, m: FakeMaskSession()
    )

    separator._create_session("dml:0", MODEL_ID)

    assert "dml:0::otro_modelo" not in separator._session_cache


def test_sessions_are_cached_per_device_and_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    separator, _ = make_separator(settings)

    first = separator._get_session("cpu", MODEL_ID)
    assert separator._get_session("cpu", MODEL_ID) is first
    separator.release_device("cpu")
    assert separator._session_cache == {}


# ---------------------------------------------------------------------------
# Ruteo por arquitectura: ahora son TRES motores detras de una lista
# ---------------------------------------------------------------------------


def test_the_roformer_engine_refuses_a_model_of_another_architecture(
    tmp_path: Path,
) -> None:
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


def test_the_roformer_engine_reports_a_missing_model_as_a_missing_pack(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    separator, _ = make_separator(settings)
    write_stereo_wav(tmp_path / "mix.wav", SR)

    with pytest.raises(RuntimeError, match=MODEL_ID):
        asyncio.run(
            separator.run(
                tmp_path / "mix.wav",
                tmp_path / "a.wav",
                tmp_path / "b.wav",
                "cpu",
                model_id=MODEL_ID,
            )
        )


async def test_pipeline_routes_the_roformer_model_to_the_roformer_engine(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    mdx, vr, roformer = FakeSeparator(), FakeSeparator(), FakeSeparator()
    pipeline = make_pipeline(settings, mdx)
    pipeline.separators["vr"] = vr
    pipeline.separators["roformer"] = roformer
    job = make_separate_job(write_upload_source(settings), separation_model=MODEL_ID)

    output_path = await pipeline.run(job)

    assert roformer.calls and not mdx.calls and not vr.calls
    assert roformer.calls[0][4] == MODEL_ID
    assert output_path.name == f"{job.id}.instrumental.flac"
    assert job.secondary_output_path is not None
    assert job.secondary_output_path.name == f"{job.id}.vocals.flac"


async def test_a_roformer_job_without_its_engine_fails_loudly(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = make_pipeline(settings, FakeSeparator())  # solo mdx
    job = make_separate_job(write_upload_source(settings), separation_model=MODEL_ID)

    with pytest.raises(RuntimeError, match="arquitectura roformer"):
        await pipeline.run(job)


async def test_manager_accepts_the_roformer_model_when_installed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="song.wav",
        separate=True,
        separation_model=MODEL_ID,
    )

    assert job.separation_model == MODEL_ID


# ---------------------------------------------------------------------------
# API: la advertencia tiene que llegar a la UI
# ---------------------------------------------------------------------------


async def test_capabilities_expose_the_warning_only_for_the_slow_model(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings)

    response = await audio_capabilities(settings=settings)
    by_id = {model.id: model for model in response.separation_models}

    assert by_id[MODEL_ID].architecture == "roformer"
    assert by_id[MODEL_ID].installed is True
    assert by_id[MODEL_ID].warning_key == SPEC.warning_key
    assert by_id["inst_hq_3"].warning_key is None
    assert [stem.id for stem in by_id[MODEL_ID].stems] == ["instrumental", "vocals"]
    karaoke = [model.id for model in response.separation_models if model.category == "karaoke"]
    assert karaoke == ["inst_hq_3", "voc_ft", MODEL_ID]
