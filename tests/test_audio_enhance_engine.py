from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pytest

from app.config import Settings
from app.services.engines.audio_enhance import AudioEnhancer

# ---------------------------------------------------------------------------
# Task 19 (6.1b) - Audio enhance engine (deepfilter/rnnoise): argv
# construction, shared guarded runner reuse, output validation, availability.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), **overrides)  # type: ignore[arg-type]


def make_fake_deepfilter_binary(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "deep-filter.exe"
    binary.write_bytes(b"fake")
    return binary


def make_fake_rnnoise_model(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "sh.rnnn"
    model.write_bytes(b"fake")
    return model


def make_deepfilter_available_settings(tmp_path: Path, **overrides: object) -> Settings:
    binary = make_fake_deepfilter_binary(tmp_path / "install")
    return make_settings(tmp_path, DEEPFILTER_BINARY=str(binary), **overrides)


def make_rnnoise_available_settings(tmp_path: Path, **overrides: object) -> Settings:
    model = make_fake_rnnoise_model(tmp_path / "install")
    return make_settings(tmp_path, RNNOISE_MODEL=str(model), **overrides)


def write_wav(path: Path, content: bytes = b"fake-wav") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ---------------------------------------------------------------------------
# mode validation
# ---------------------------------------------------------------------------


def test_audio_enhancer_raises_value_error_for_unknown_mode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(ValueError, match="Unknown audio enhance mode"):
        AudioEnhancer(settings, mode="not-a-real-mode")


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------


def test_audio_enhancer_available_true_when_deepfilter_binary_exists(tmp_path: Path) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")

    assert engine.available() is True


def test_audio_enhancer_available_false_when_deepfilter_binary_missing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, DEEPFILTER_BINARY=str(tmp_path / "missing-deep-filter.exe"))
    engine = AudioEnhancer(settings, mode="deepfilter")

    assert engine.available() is False


def test_audio_enhancer_available_true_when_rnnoise_model_exists(tmp_path: Path) -> None:
    settings = make_rnnoise_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="rnnoise")

    assert engine.available() is True


def test_audio_enhancer_available_false_when_rnnoise_model_missing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, RNNOISE_MODEL=str(tmp_path / "missing.rnnn"))
    engine = AudioEnhancer(settings, mode="rnnoise")

    assert engine.available() is False


# ---------------------------------------------------------------------------
# deepfilter argv construction + output relocation
# ---------------------------------------------------------------------------


async def test_deepfilter_engine_run_builds_expected_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    calls: list[list[str]] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(command)
        write_wav(output_wav.parent / input_wav.name, b"enhanced-audio")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert calls == [
        [
            str(settings.deepfilter_binary_path),
            "-o",
            str(output_wav.parent),
            str(input_wav),
        ]
    ]


async def test_deepfilter_engine_run_relocates_output_to_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_wav(output_wav.parent / input_wav.name, b"enhanced-audio")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert output_wav.read_bytes() == b"enhanced-audio"
    assert not (output_wav.parent / input_wav.name).exists()


async def test_deepfilter_engine_run_skips_relocation_when_names_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "clip.wav"
    output_wav = tmp_path / "out" / "clip.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_wav(output_wav.parent / input_wav.name, b"same-name-output")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert output_wav.read_bytes() == b"same-name-output"


async def test_deepfilter_engine_run_creates_output_dir_before_invoking_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    dir_existed_at_call_time = False

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        nonlocal dir_existed_at_call_time
        dir_existed_at_call_time = output_wav.parent.exists()
        write_wav(output_wav.parent / input_wav.name, b"enhanced-audio")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert dir_existed_at_call_time is True


# ---------------------------------------------------------------------------
# rnnoise argv construction (ffmpeg arnndn filter)
# ---------------------------------------------------------------------------


async def test_rnnoise_engine_run_builds_expected_argv_with_escaped_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_rnnoise_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="rnnoise")
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out" / "out.wav"
    write_wav(input_wav)

    calls: list[list[str]] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(command)
        write_wav(output_wav, b"denoised-audio")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    expected_model_arg = str(settings.rnnoise_model_path).replace("\\", "/").replace(":", "\\:")
    assert calls == [
        [
            str(settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(input_wav),
            "-af",
            f"arnndn=m={expected_model_arg}",
            str(output_wav),
        ]
    ]


async def test_rnnoise_engine_run_creates_output_dir_before_invoking_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_rnnoise_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="rnnoise")
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out" / "out.wav"
    write_wav(input_wav)

    dir_existed_at_call_time = False

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        nonlocal dir_existed_at_call_time
        dir_existed_at_call_time = output_wav.parent.exists()
        write_wav(output_wav, b"denoised-audio")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert dir_existed_at_call_time is True


def test_escape_filter_path_escapes_windows_drive_colon_and_backslashes() -> None:
    path = PureWindowsPath(r"C:\vendor\deepfilternet\models\sh.rnnn")

    escaped = AudioEnhancer._escape_filter_path(path)

    assert escaped == r"C\:/vendor/deepfilternet/models/sh.rnnn"


# ---------------------------------------------------------------------------
# output validation (0-byte / missing output)
# ---------------------------------------------------------------------------


async def test_deepfilter_engine_run_raises_when_produced_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="no output file was produced"):
        await engine.run(input_wav, output_wav)


async def test_deepfilter_engine_run_raises_when_output_is_zero_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_wav(output_wav.parent / input_wav.name, b"")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="no output file"):
        await engine.run(input_wav, output_wav)


async def test_rnnoise_engine_run_raises_when_output_is_zero_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_rnnoise_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="rnnoise")
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out" / "out.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_wav(output_wav, b"")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="no output file"):
        await engine.run(input_wav, output_wav)


async def test_rnnoise_engine_run_raises_when_output_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_rnnoise_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="rnnoise")
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out" / "out.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="no output file"):
        await engine.run(input_wav, output_wav)


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------


async def test_audio_enhancer_run_raises_when_not_available_for_deepfilter(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, DEEPFILTER_BINARY=str(tmp_path / "missing.exe"))
    engine = AudioEnhancer(settings, mode="deepfilter")

    with pytest.raises(RuntimeError, match="not available"):
        await engine.run(tmp_path / "in.wav", tmp_path / "out.wav")


async def test_audio_enhancer_run_raises_when_not_available_for_rnnoise(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, RNNOISE_MODEL=str(tmp_path / "missing.rnnn"))
    engine = AudioEnhancer(settings, mode="rnnoise")

    with pytest.raises(RuntimeError, match="not available"):
        await engine.run(tmp_path / "in.wav", tmp_path / "out.wav")


async def test_deepfilter_engine_run_raises_clear_error_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom: simulated deep-filter failure\n", 1

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="boom: simulated deep-filter failure"):
        await engine.run(input_wav, output_wav)


async def test_rnnoise_engine_run_raises_clear_error_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_rnnoise_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="rnnoise")
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out" / "out.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom: simulated ffmpeg arnndn failure\n", 1

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="boom: simulated ffmpeg arnndn failure"):
        await engine.run(input_wav, output_wav)


# ---------------------------------------------------------------------------
# shared runner reuse + timeout pass-through
# ---------------------------------------------------------------------------


async def test_deepfilter_engine_run_uses_shared_runner_with_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path, SUBPROCESS_TIMEOUT=789)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    calls: list[float] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(timeout)
        write_wav(output_wav.parent / input_wav.name, b"enhanced-audio")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert calls == [789]


async def test_rnnoise_engine_run_uses_shared_runner_with_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_rnnoise_available_settings(tmp_path, SUBPROCESS_TIMEOUT=321)
    engine = AudioEnhancer(settings, mode="rnnoise")
    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out" / "out.wav"
    write_wav(input_wav)

    calls: list[float] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(timeout)
        write_wav(output_wav, b"denoised-audio")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert calls == [321]
