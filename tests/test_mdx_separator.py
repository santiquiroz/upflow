from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.mdx_models import (
    DEFAULT_SEPARATION_MODEL,
    SEPARATION_MODELS,
    installed_model_ids,
)
from app.services.engines.mdx_separator import MdxSeparator
from app.services.gpu_session_coordinator import GpuSessionCoordinator

# ---------------------------------------------------------------------------
# v0.59.0 - motor de separacion voz/instrumental (MDX-Net/UVR). Porta las
# aserciones numericas del spike (spike-stems, validado DML == CPU el
# 2026-08-09) con una sesion fake identidad: el STFT -> modelo -> iSTFT ->
# trim -> concatenar tiene reconstruccion casi perfecta, asi que con un modelo
# que devuelve su entrada el "stem primario" tiene que ser la mezcla misma y el
# secundario la resta compensada.
# ---------------------------------------------------------------------------

SR = 44100


class FakeIdentitySession:
    """Devuelve el tensor de entrada tal cual: primario == mezcla (menos los 3
    bins graves que el pipeline pone a cero antes de inferir)."""

    class _Meta:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self) -> None:
        self.run_calls = 0

    def get_inputs(self):
        return [self._Meta("input")]

    def get_outputs(self):
        return [self._Meta("output")]

    def run(self, output_names, feeds):
        self.run_calls += 1
        return [feeds["input"]]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        KARAOKE_MODEL_DIR=str(tmp_path / "karaoke"),
    )


def install_fake_model(settings: Settings, model_id: str) -> Path:
    model_dir = settings.karaoke_model_dir_path
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / SEPARATION_MODELS[model_id].filename
    path.write_bytes(b"fake-onnx")
    return path


def make_separator(
    settings: Settings, session: FakeIdentitySession | None = None
) -> tuple[MdxSeparator, FakeIdentitySession]:
    separator = MdxSeparator(settings, GpuSessionCoordinator())
    fake = session or FakeIdentitySession()
    separator._create_session = lambda device, model_id: fake  # type: ignore[method-assign]
    return separator, fake


def write_stereo_wav(path: Path, samples: int) -> np.ndarray:
    import soundfile as sf

    t = np.arange(samples) / SR
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.25 * np.sin(2 * np.pi * 554.0 * t)
    mix = np.stack([left, right]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mix.T, SR)
    return mix


def read_stereo_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    assert rate == SR
    return data.T


def run_separation(
    separator: MdxSeparator, tmp_path: Path, mix_path: Path, model_id: str
) -> tuple[np.ndarray, np.ndarray]:
    instrumental = tmp_path / "out" / "instrumental.wav"
    vocals = tmp_path / "out" / "vocals.wav"
    asyncio.run(separator.run(mix_path, instrumental, vocals, "cpu", model_id=model_id))
    return read_stereo_wav(instrumental), read_stereo_wav(vocals)


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    # Portado del spike: la metrica global con la que se valido el pipeline.
    est, ref = estimate.ravel(), reference.ravel()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target**2) + 1e-12) / (np.sum(noise**2) + 1e-12)))


def interior(signal: np.ndarray, n_fft: int) -> np.ndarray:
    # Los 3 bins graves que UVR pone a cero + el reflect de los bordes dejan un
    # residuo LOCAL en los extremos (medido: <0.035); el interior reconstruye a
    # ~1e-7. La verificacion fina va sobre el interior, la global por SI-SDR.
    return signal[:, 2 * n_fft : signal.shape[1] - 2 * n_fft]


# ---------------------------------------------------------------------------
# Numerica: reconstruccion, compensacion, inversion por primary_stem
# ---------------------------------------------------------------------------


def test_identity_model_reconstructs_the_mix_as_primary_stem(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    separator, _ = make_separator(settings)
    samples = SEPARATION_MODELS["inst_hq_3"].gen_samples + 1000
    mix = write_stereo_wav(tmp_path / "mix.wav", samples)

    instrumental, _vocals = run_separation(separator, tmp_path, tmp_path / "mix.wav", "inst_hq_3")

    # inst_hq_3 saca la instrumental: con el modelo identidad debe ser la
    # mezcla misma (reconstruccion STFT/iSTFT, spike 2026-08-09; medido
    # SI-SDR ~59 dB e interior exacto a ~1e-7).
    assert instrumental.shape == mix.shape
    assert si_sdr(instrumental, mix) > 40.0
    n_fft = SEPARATION_MODELS["inst_hq_3"].n_fft
    assert np.allclose(interior(instrumental, n_fft), interior(mix, n_fft), atol=2e-4)


def test_vocals_are_the_compensated_residual(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    separator, _ = make_separator(settings)
    samples = SEPARATION_MODELS["inst_hq_3"].gen_samples // 2
    mix = write_stereo_wav(tmp_path / "mix.wav", samples)

    instrumental, vocals = run_separation(separator, tmp_path, tmp_path / "mix.wav", "inst_hq_3")

    compensate = SEPARATION_MODELS["inst_hq_3"].compensate
    assert np.allclose(vocals, mix - instrumental * compensate, atol=1e-4)


def test_primary_stem_vocals_inverts_the_mapping(tmp_path: Path) -> None:
    # voc_ft saca la VOZ: con identidad, vocals == mezcla y la instrumental es
    # la resta compensada. La logica es generica por primary_stem del catalogo.
    settings = make_settings(tmp_path)
    install_fake_model(settings, "voc_ft")
    separator, _ = make_separator(settings)
    spec = SEPARATION_MODELS["voc_ft"]
    mix = write_stereo_wav(tmp_path / "mix.wav", spec.gen_samples // 2)

    instrumental, vocals = run_separation(separator, tmp_path, tmp_path / "mix.wav", "voc_ft")

    assert si_sdr(vocals, mix) > 40.0
    assert np.allclose(interior(vocals, spec.n_fft), interior(mix, spec.n_fft), atol=2e-4)
    assert np.allclose(instrumental, mix - vocals * spec.compensate, atol=1e-4)


def test_mono_input_is_duplicated_to_stereo(tmp_path: Path) -> None:
    import soundfile as sf

    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    separator, _ = make_separator(settings)
    t = np.arange(SR) / SR
    mono = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    mix_path = tmp_path / "mono.wav"
    sf.write(str(mix_path), mono, SR)

    instrumental, vocals = run_separation(separator, tmp_path, mix_path, "inst_hq_3")

    assert instrumental.shape[0] == 2
    assert vocals.shape[0] == 2
    assert np.allclose(instrumental[0], instrumental[1], atol=1e-6)


def test_overshooting_secondary_is_peak_normalized_and_saved_as_float(tmp_path: Path) -> None:
    import soundfile as sf

    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")

    class OutOfPhaseSession(FakeIdentitySession):
        """Primario en contrafase: el secundario = mezcla*(1+compensate) supera
        +-1.0 en una mezcla caliente, el caso que el PCM_16 default recortaba."""

        def run(self, output_names, feeds):
            self.run_calls += 1
            return [-feeds["input"]]

    separator, _ = make_separator(settings, OutOfPhaseSession())
    spec = SEPARATION_MODELS["inst_hq_3"]
    samples = spec.gen_samples // 2
    t = np.arange(samples) / SR
    hot = np.stack(
        [0.9 * np.sin(2 * np.pi * 440.0 * t), 0.9 * np.sin(2 * np.pi * 554.0 * t)]
    ).astype(np.float32)
    mix_path = tmp_path / "mix.wav"
    sf.write(str(mix_path), hot.T, SR, subtype="FLOAT")

    saved: dict[str, np.ndarray] = {}
    original_save = separator._save_stereo

    def spy_save(path: Path, audio: np.ndarray) -> None:
        saved[path.name] = audio.copy()
        original_save(path, audio)

    separator._save_stereo = spy_save  # type: ignore[method-assign]

    instrumental, vocals = run_separation(separator, tmp_path, mix_path, "inst_hq_3")

    # Hubo overshoot y el factor compartido dejo el pico mayor exacto en 1.0...
    assert float(np.max(np.abs(saved["vocals.wav"]))) == pytest.approx(1.0, abs=1e-9)
    # ...y escalo la instrumental con el MISMO factor (no un factor por stem).
    expected_instrumental_peak = 0.9 / (0.9 * (1.0 + spec.compensate))
    assert float(np.max(np.abs(saved["instrumental.wav"]))) == pytest.approx(
        expected_instrumental_peak, rel=0.05
    )
    # WAV intermedio en float: el disco reproduce lo guardado a 1e-6, por debajo
    # del paso de cuantizacion de 16 bits (3.05e-5) — una regresion a PCM_16
    # falla fuerte aca.
    assert float(np.max(np.abs(instrumental))) <= 1.0
    assert float(np.max(np.abs(vocals))) <= 1.0
    assert np.allclose(instrumental, saved["instrumental.wav"], atol=1e-6)
    assert np.allclose(vocals, saved["vocals.wav"], atol=1e-6)


# ---------------------------------------------------------------------------
# Chunks y trim
# ---------------------------------------------------------------------------


def test_chunk_loop_covers_the_whole_signal(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    separator, fake = make_separator(settings)
    gen = SEPARATION_MODELS["inst_hq_3"].gen_samples
    samples = gen + 1000  # fuerza DOS chunks con padding
    mix = write_stereo_wav(tmp_path / "mix.wav", samples)

    instrumental, _ = run_separation(separator, tmp_path, tmp_path / "mix.wav", "inst_hq_3")

    assert fake.run_calls == 2
    # El padding del ultimo chunk no puede alargar la salida.
    assert instrumental.shape[1] == mix.shape[1]


def test_output_length_matches_input_for_short_clips(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    separator, fake = make_separator(settings)
    mix = write_stereo_wav(tmp_path / "mix.wav", 12345)

    instrumental, vocals = run_separation(separator, tmp_path, tmp_path / "mix.wav", "inst_hq_3")

    assert fake.run_calls == 1
    assert instrumental.shape[1] == mix.shape[1]
    assert vocals.shape[1] == mix.shape[1]


def test_run_reports_chunk_progress_with_exact_totals(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    separator, _ = make_separator(settings)
    gen = SEPARATION_MODELS["inst_hq_3"].gen_samples
    write_stereo_wav(tmp_path / "mix.wav", gen + 1000)  # fuerza DOS chunks
    calls: list[tuple[int, int]] = []

    instrumental = tmp_path / "out" / "instrumental.wav"
    vocals = tmp_path / "out" / "vocals.wav"
    asyncio.run(
        separator.run(
            tmp_path / "mix.wav",
            instrumental,
            vocals,
            "cpu",
            model_id="inst_hq_3",
            on_chunk=lambda done, total: calls.append((done, total)),
        )
    )

    assert calls == [(1, 2), (2, 2)]


# ---------------------------------------------------------------------------
# Cancelacion
# ---------------------------------------------------------------------------


def test_a_second_cancel_still_waits_for_the_worker_thread(tmp_path: Path) -> None:
    # Regresion: suppress(Exception) no cubria el segundo CancelledError y run()
    # propagaba con el hilo vivo — el rmtree del caller corria en paralelo con
    # un save_wav rezagado que resucitaba el work dir.
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    entered = threading.Event()
    release = threading.Event()

    class SlowSession(FakeIdentitySession):
        def run(self, output_names, feeds):
            entered.set()
            release.wait(timeout=10)
            return super().run(output_names, feeds)

    separator, _ = make_separator(settings, SlowSession())
    write_stereo_wav(tmp_path / "mix.wav", 12345)
    instrumental = tmp_path / "out" / "instrumental.wav"
    vocals = tmp_path / "out" / "vocals.wav"

    async def scenario() -> bool:
        task = asyncio.create_task(
            separator.run(
                tmp_path / "mix.wav", instrumental, vocals, "cpu", model_id="inst_hq_3"
            )
        )
        await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        await asyncio.sleep(0.05)  # el primer cancel llega a la espera del hilo
        task.cancel()  # segundo cancel EN la espera (doble click / retry MCP)
        done, _pending = await asyncio.wait({task}, timeout=0.3)
        held_the_wait = not done
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return held_the_wait

    assert asyncio.run(scenario()) is True
    # El hilo termino ANTES de que run() propagara: sus saves ya llegaron.
    assert instrumental.exists() and vocals.exists()


# ---------------------------------------------------------------------------
# Disponibilidad y validaciones
# ---------------------------------------------------------------------------


def test_available_reflects_installed_models_on_disk(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator, _ = make_separator(settings)

    assert separator.available() is False
    install_fake_model(settings, "voc_ft")
    assert separator.available() is True
    assert installed_model_ids(settings.karaoke_model_dir_path) == ["voc_ft"]


def test_missing_model_raises_the_pack_message_naming_the_variant(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator, _ = make_separator(settings)
    write_stereo_wav(tmp_path / "mix.wav", 1000)

    with pytest.raises(RuntimeError, match=r"\(inst_hq_3\)") as excinfo:
        run_separation(separator, tmp_path, tmp_path / "mix.wav", "inst_hq_3")
    assert "boton" in str(excinfo.value)


def test_unknown_model_lists_the_catalog(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator, _ = make_separator(settings)
    write_stereo_wav(tmp_path / "mix.wav", 1000)

    with pytest.raises(RuntimeError, match="inst_hq_3"):
        run_separation(separator, tmp_path, tmp_path / "mix.wav", "no-existe")


def test_default_model_id_is_in_the_catalog() -> None:
    assert DEFAULT_SEPARATION_MODEL in SEPARATION_MODELS


# ---------------------------------------------------------------------------
# Cache de sesiones por device+modelo
# ---------------------------------------------------------------------------


def test_session_is_cached_per_device_and_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    install_fake_model(settings, "inst_hq_3")
    separator = MdxSeparator(settings, GpuSessionCoordinator())
    creations: list[tuple[str, str]] = []

    def create(device: str, model_id: str) -> FakeIdentitySession:
        creations.append((device, model_id))
        return FakeIdentitySession()

    separator._create_session = create  # type: ignore[method-assign]

    first = separator._get_session("cpu", "inst_hq_3")
    second = separator._get_session("cpu", "inst_hq_3")

    assert first is second
    assert creations == [("cpu", "inst_hq_3")]


def test_release_device_evicts_only_that_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    separator = MdxSeparator(settings, GpuSessionCoordinator())
    separator._session_cache["cpu::inst_hq_3"] = object()
    separator._session_cache["dml:0::inst_hq_3"] = object()

    separator.release_device("cpu")

    assert "cpu::inst_hq_3" not in separator._session_cache
    assert "dml:0::inst_hq_3" in separator._session_cache
