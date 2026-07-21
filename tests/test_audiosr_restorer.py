from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.services.engines.audiosr_restore import AudioSrRestorer
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.restorer_registry import build_restorers, validate_restore_mode_ready

# ---------------------------------------------------------------------------
# SP13 - AudioSrRestorer: availability gate, fake-session inference path,
# LRU(1) session cache and the registry/validation glue.
# ---------------------------------------------------------------------------


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "audiosr"
    model_dir.mkdir(parents=True)
    np.save(model_dir / "alphas_cumprod.npy", np.linspace(0.9999, 0.0, 1000))
    np.save(model_dir / "mel_basis.npy",
            np.random.default_rng(2).random((256, 1025), dtype=np.float32) * 0.01)
    manifest = {
        "scale_factor": 0.3342,
        "mel": {"basis_file": "mel_basis.npy"},
        "scheduler": {"alphas_cumprod_file": "alphas_cumprod.npy"},
        "cfg": {"guidance_scale": 3.5, "unconditional_value": -11.4981},
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in ("vocoder", "vae_decoder", "vae_feature_extract", "ddpm"):
        (model_dir / f"{name}.onnx").write_bytes(b"fake")
    return model_dir


def make_settings(tmp_path: Path, enabled: bool = True) -> Settings:
    model_dir = make_model_dir(tmp_path)
    return Settings(
        _env_file=None,
        ENABLE_AUDIOSR=enabled,
        AUDIOSR_MODEL_DIR=str(model_dir),
        AUDIOSR_DDIM_STEPS=2,
    )


class FakeSession:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.name == "vae_feature_extract":
            return [feeds["noise"] * 0.1]
        if self.name == "ddpm":
            return [feeds["x"][:, :16] * 0.5]
        if self.name == "vae_decoder":
            frames = feeds["z"].shape[2] * 8
            return [np.full((1, 1, frames, 256), -3.0, dtype=np.float32)]
        if self.name == "vocoder":
            samples = feeds["mel"].shape[2] * 480
            t = np.arange(samples, dtype=np.float32)
            return [(0.3 * np.sin(2 * np.pi * 220 * t / 48000)).reshape(1, 1, -1)]
        raise AssertionError(self.name)


def fake_sessions(_device: str) -> dict[str, Any]:
    return {name: FakeSession(name) for name in
            ("vocoder", "vae_decoder", "vae_feature_extract", "ddpm")}


def write_input_wav(path: Path, seconds: float = 1.0, rate: int = 44100) -> None:
    t = np.arange(int(rate * seconds)) / rate
    sf.write(str(path), (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)


def test_available_follows_settings_gate(tmp_path: Path) -> None:
    assert AudioSrRestorer(make_settings(tmp_path, enabled=True), GpuSessionCoordinator()).available() is True
    assert AudioSrRestorer(make_settings(tmp_path / "off", enabled=False), GpuSessionCoordinator()).available() is False


def test_run_produces_48k_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    restorer = AudioSrRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(restorer, "_create_sessions", fake_sessions)
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"
    write_input_wav(input_wav)

    asyncio.run(restorer.run(input_wav, output_wav, device="cpu"))

    data, rate = sf.read(str(output_wav))
    assert rate == 48000
    assert data.shape[-1] == 48000  # resampled 44.1k input, same duration out


def test_run_when_unavailable_raises_actionable_error(tmp_path: Path) -> None:
    restorer = AudioSrRestorer(make_settings(tmp_path, enabled=False), GpuSessionCoordinator())
    input_wav = tmp_path / "in.wav"
    write_input_wav(input_wav)

    with pytest.raises(RuntimeError, match="ENABLE_AUDIOSR"):
        asyncio.run(restorer.run(input_wav, tmp_path / "out.wav", device="cpu"))


def write_stereo_input_wav(path: Path, seconds: float = 1.0, rate: int = 44100) -> None:
    t = np.arange(int(rate * seconds)) / rate
    left = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), np.stack([left, right], axis=1), rate)


def test_run_and_save_preserves_stereo_via_multichannel_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restorer = AudioSrRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(restorer, "_create_sessions", fake_sessions)
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"
    write_stereo_input_wav(input_wav)

    asyncio.run(restorer.run(input_wav, output_wav, device="cpu"))

    result, rate = sf.read(str(output_wav), always_2d=True)
    assert result.shape[1] == 2  # sigue estereo, no colapsa a mono


def test_session_cache_keeps_single_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    restorer = AudioSrRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    built: list[str] = []

    def tracking_sessions(device: str) -> dict[str, Any]:
        built.append(device)
        return fake_sessions(device)

    monkeypatch.setattr(restorer, "_create_sessions", tracking_sessions)

    restorer._get_sessions("cpu")
    restorer._get_sessions("cpu")
    assert built == ["cpu"]  # cached

    restorer._get_sessions("dml:0")
    restorer._get_sessions("cpu")  # evicted by dml:0 -> rebuilt
    assert built == ["cpu", "dml:0", "cpu"]


def test_build_restorers_registers_both_engines(tmp_path: Path) -> None:
    restorers = build_restorers(make_settings(tmp_path), GpuSessionCoordinator())

    assert set(restorers) == {"apollo", "audiosr"}


def test_cancel_waits_for_worker_thread_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review HIGH: si run() re-lanza CancelledError sin esperar el thread, el
    # finally del pipeline borra el work dir mientras un _save_wav rezagado lo
    # resucita. El contrato es: al propagar el cancel, el thread YA terminó.
    import threading
    import time

    restorer = AudioSrRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    worker_finished = threading.Event()

    def slow_worker(input_wav: Path, output_wav: Path, device: str, cancel_event: threading.Event) -> None:
        cancel_event.wait(timeout=10)
        time.sleep(0.2)  # cola no-interrumpible simulada
        worker_finished.set()

    monkeypatch.setattr(restorer, "_run_and_save", slow_worker)

    async def scenario() -> None:
        task = asyncio.create_task(restorer.run(tmp_path / "in.wav", tmp_path / "out.wav", "cpu"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker_finished.is_set(), "run() propagó el cancel antes de que el thread terminara"

    asyncio.run(scenario())


def test_zero_sample_audio_raises_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restorer = AudioSrRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(restorer, "_create_sessions", fake_sessions)
    empty_wav = tmp_path / "empty.wav"
    sf.write(str(empty_wav), np.zeros(0, dtype=np.float32), 48000)

    with pytest.raises(RuntimeError, match="zero samples"):
        asyncio.run(restorer.run(empty_wav, tmp_path / "out.wav", "cpu"))


def test_validate_restore_mode_ready_messages(tmp_path: Path) -> None:
    disabled = make_settings(tmp_path, enabled=False)
    with pytest.raises(ValueError, match="ENABLE_AUDIOSR"):
        validate_restore_mode_ready(disabled, "audiosr")

    enabled_missing = Settings(
        _env_file=None, ENABLE_AUDIOSR=True, AUDIOSR_MODEL_DIR=str(tmp_path / "missing")
    )
    with pytest.raises(ValueError, match="download-audiosr-onnx"):
        validate_restore_mode_ready(enabled_missing, "audiosr")

    with pytest.raises(ValueError, match="Unknown restore mode"):
        validate_restore_mode_ready(disabled, "nope")


# ---------------------------------------------------------------------------
# GpuSessionCoordinator wiring (Fase 1 Task 2) - release_device evicts only
# its own device's cache entry, acquire() runs before any session is built.
# ---------------------------------------------------------------------------


def test_release_device_clears_cached_session_for_that_device_only(tmp_path: Path) -> None:
    restorer = AudioSrRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    restorer._session_cache["dml:0"] = {"fake": "session"}
    restorer._session_cache["dml:1"] = {"fake": "session-1"}

    restorer.release_device("dml:0")

    assert "dml:0" not in restorer._session_cache
    assert "dml:1" in restorer._session_cache


def test_release_device_on_empty_cache_is_a_noop(tmp_path: Path) -> None:
    restorer = AudioSrRestorer(make_settings(tmp_path), GpuSessionCoordinator())

    restorer.release_device("dml:0")  # no debe lanzar


def test_get_sessions_calls_coordinator_acquire_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_coordinator = GpuSessionCoordinator()
    restorer = AudioSrRestorer(make_settings(tmp_path), gpu_coordinator)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(restorer, "_create_sessions", lambda device: {"fake": "session"})

    restorer._get_sessions("dml:0")

    assert calls == [("dml:0", restorer)]
