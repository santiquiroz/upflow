from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.services.engines.apollo_restore import ApolloRestorer
from app.services.gpu_session_coordinator import GpuSessionCoordinator

# ---------------------------------------------------------------------------
# ApolloRestorer:
# - GpuSessionCoordinator wiring (Fase 1 Task 2 de gmfss-production-performance):
#   same contract as AudioSrRestorer (see tests/test_audiosr_restorer.py), just
#   keyed by a single cached session per device instead of a dict of graphs.
# - Availability gate y wiring de M/S (multichannel_restore) via un fake onnx
#   session (Fase B Task 6 de audio-tracks-subs-quality), mismo patron que
#   test_audiosr_restorer.py.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def make_model_file(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "apollo.onnx"
    model.write_bytes(b"fake")
    return model


def make_settings_with_model(tmp_path: Path, enabled: bool = True) -> Settings:
    model = make_model_file(tmp_path)
    return Settings(_env_file=None, ENABLE_AUDIO_RESTORE=enabled, APOLLO_RESTORE_MODEL=str(model))


def test_release_device_clears_cached_session_for_that_device_only(tmp_path: Path) -> None:
    restorer = ApolloRestorer(make_settings(tmp_path), GpuSessionCoordinator())
    restorer._session_cache["dml:0"] = "fake-session"
    restorer._session_cache["dml:1"] = "fake-session-1"

    restorer.release_device("dml:0")

    assert "dml:0" not in restorer._session_cache
    assert "dml:1" in restorer._session_cache


def test_release_device_on_empty_cache_is_a_noop(tmp_path: Path) -> None:
    restorer = ApolloRestorer(make_settings(tmp_path), GpuSessionCoordinator())

    restorer.release_device("dml:0")  # no debe lanzar


def test_get_session_calls_coordinator_acquire_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_coordinator = GpuSessionCoordinator()
    restorer = ApolloRestorer(make_settings(tmp_path), gpu_coordinator)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(restorer, "_create_session", lambda device: "fake-session")

    restorer._get_session("dml:0")

    assert calls == [("dml:0", restorer)]


class FakeIoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeApolloSession:
    """Eco identidad: devuelve el mismo audio que recibe, reshape incluido."""

    def get_inputs(self) -> list[FakeIoInfo]:
        return [FakeIoInfo("audio")]

    def get_outputs(self) -> list[FakeIoInfo]:
        return [FakeIoInfo("restored")]

    def run(self, _output_names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [feeds["audio"].copy()]


def fake_session(_device: str) -> FakeApolloSession:
    return FakeApolloSession()


def write_mono_input_wav(path: Path, seconds: float = 0.2, rate: int = 44100) -> None:
    t = np.arange(int(rate * seconds)) / rate
    sf.write(str(path), (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)


def write_stereo_input_wav(path: Path, seconds: float = 0.2, rate: int = 44100) -> None:
    t = np.arange(int(rate * seconds)) / rate
    left = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(path), np.stack([left, right], axis=1), rate)


def test_available_follows_settings_gate(tmp_path: Path) -> None:
    coordinator = GpuSessionCoordinator()
    assert ApolloRestorer(make_settings_with_model(tmp_path, enabled=True), coordinator).available() is True
    assert ApolloRestorer(make_settings_with_model(tmp_path / "off", enabled=False), coordinator).available() is False


def test_run_when_unavailable_raises_actionable_error(tmp_path: Path) -> None:
    restorer = ApolloRestorer(make_settings_with_model(tmp_path, enabled=False), GpuSessionCoordinator())
    input_wav = tmp_path / "in.wav"
    write_mono_input_wav(input_wav)

    with pytest.raises(RuntimeError, match="ENABLE_AUDIO_RESTORE"):
        asyncio.run(restorer.run(input_wav, tmp_path / "out.wav", device="cpu"))


def test_run_and_save_keeps_mono_input_mono(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restorer = ApolloRestorer(make_settings_with_model(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(restorer, "_create_session", fake_session)
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"
    write_mono_input_wav(input_wav)

    asyncio.run(restorer.run(input_wav, output_wav, device="cpu"))

    result, rate = sf.read(str(output_wav), always_2d=True)
    assert rate == 44100
    assert result.shape[1] == 1


def test_run_and_save_preserves_stereo_via_multichannel_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restorer = ApolloRestorer(make_settings_with_model(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(restorer, "_create_session", fake_session)
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"
    write_stereo_input_wav(input_wav)

    asyncio.run(restorer.run(input_wav, output_wav, device="cpu"))

    result, rate = sf.read(str(output_wav), always_2d=True)
    assert result.shape[1] == 2  # sigue estereo, no colapsa a mono
