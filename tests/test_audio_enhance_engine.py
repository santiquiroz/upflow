from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pytest

from app.config import AUDIO_ENHANCE_MODES, Settings
from app.services.engines.audio_enhance import AudioEnhancer
from app.services.process_runner import SubprocessTimeoutError

# ---------------------------------------------------------------------------
# Task 19 (6.1b) - Audio enhance engine (deepfilter/rnnoise): argv
# construction, shared guarded runner reuse, output validation, availability,
# temp-dir isolation for deep-filter's dir-only output contract.
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


def deepfilter_out_dir(command: list[str]) -> Path:
    return Path(command[command.index("-o") + 1])


def write_deepfilter_output(command: list[str], input_wav: Path, content: bytes = b"enhanced-audio") -> None:
    # Mimics the real binary: writes the enhanced file under the input's own
    # basename inside whatever -o directory it was given.
    write_wav(deepfilter_out_dir(command) / input_wav.name, content)


def leftover_temp_dirs(directory: Path) -> list[Path]:
    return list(directory.glob(".dfn-*"))


# ---------------------------------------------------------------------------
# mode validation
# ---------------------------------------------------------------------------


def test_audio_enhancer_raises_value_error_for_unknown_mode(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(ValueError, match="Unknown audio enhance mode"):
        AudioEnhancer(settings, mode="not-a-real-mode")


def test_audio_enhancer_accepts_every_mode_known_to_settings(tmp_path: Path) -> None:
    # Guards against the engine's accepted modes drifting from the config
    # helper's: both must consume the same AUDIO_ENHANCE_MODES source.
    settings = make_settings(tmp_path)

    for mode in AUDIO_ENHANCE_MODES:
        engine = AudioEnhancer(settings, mode=mode)
        assert engine.mode == mode


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
# deepfilter argv construction + temp-dir output promotion
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
        write_deepfilter_output(command, input_wav)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert len(calls) == 1
    command = calls[0]
    assert len(command) == 4
    assert command[0] == str(settings.deepfilter_binary_path)
    assert command[1] == "-o"
    assert command[3] == str(input_wav)
    out_dir = Path(command[2])
    assert out_dir.parent == output_wav.parent
    assert out_dir.name.startswith(".dfn-")


async def test_deepfilter_engine_run_promotes_output_to_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_deepfilter_output(command, input_wav)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert output_wav.read_bytes() == b"enhanced-audio"


async def test_deepfilter_engine_run_supports_same_name_input_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "clip.wav"
    output_wav = tmp_path / "out" / "clip.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_deepfilter_output(command, input_wav, b"same-name-output")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert output_wav.read_bytes() == b"same-name-output"


async def test_deepfilter_engine_run_same_directory_leaves_input_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression guard for the same-directory hazard: input and output share
    # a parent dir, so pointing -o at output_wav.parent would make the binary
    # overwrite the SOURCE wav in place. The temp-dir strategy must keep the
    # source bytes intact and still honor the requested output path.
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    work_dir = tmp_path / "job-work"
    input_wav = work_dir / "raw.wav"
    output_wav = work_dir / "enhanced.wav"
    write_wav(input_wav, b"original-source-audio")

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        assert deepfilter_out_dir(command) != input_wav.parent
        write_deepfilter_output(command, input_wav)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert input_wav.read_bytes() == b"original-source-audio"
    assert output_wav.read_bytes() == b"enhanced-audio"


async def test_deepfilter_engine_run_creates_output_dir_before_invoking_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    dirs_existed_at_call_time = False

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        nonlocal dirs_existed_at_call_time
        dirs_existed_at_call_time = output_wav.parent.exists() and deepfilter_out_dir(command).exists()
        write_deepfilter_output(command, input_wav)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert dirs_existed_at_call_time is True


# ---------------------------------------------------------------------------
# deepfilter temp-dir cleanup
# ---------------------------------------------------------------------------


async def test_deepfilter_engine_run_cleans_temp_dir_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_deepfilter_output(command, input_wav)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    await engine.run(input_wav, output_wav)

    assert leftover_temp_dirs(output_wav.parent) == []


async def test_deepfilter_engine_run_cleans_temp_dir_when_runner_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        raise SubprocessTimeoutError("Process 'deep-filter.exe' timed out after 1s")

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(SubprocessTimeoutError):
        await engine.run(input_wav, output_wav)

    assert leftover_temp_dirs(output_wav.parent) == []


async def test_deepfilter_engine_run_cleans_temp_dir_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom\n", 1

    monkeypatch.setattr("app.services.engines.audio_enhance.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="boom"):
        await engine.run(input_wav, output_wav)

    assert leftover_temp_dirs(output_wav.parent) == []


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

    expected_model_arg = str(settings.rnnoise_model_path).replace("\\", "/").replace(":", "\\\\:")
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

    # A single backslash before the colon parses as "no option name" in the
    # real ffmpeg arnndn filter graph when invoked via argv (no shell)
    # because asyncio.create_subprocess_exec passes the value through with
    # no shell-level backslash consumption first, unlike the double-escaped
    # invocations documented in ffmpeg's own Windows-path examples. Verified
    # against the vendored ffmpeg binary in Task 21's real-binary smoke test:
    # only a double backslash reaches the filtergraph parser as an escaped
    # colon.
    assert escaped == r"C\\:/vendor/deepfilternet/models/sh.rnnn"


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

    assert leftover_temp_dirs(output_wav.parent) == []


async def test_deepfilter_engine_run_raises_when_output_is_zero_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_deepfilter_available_settings(tmp_path)
    engine = AudioEnhancer(settings, mode="deepfilter")
    input_wav = tmp_path / "in" / "raw.wav"
    output_wav = tmp_path / "out" / "enhanced.wav"
    write_wav(input_wav)

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_deepfilter_output(command, input_wav, b"")
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
        write_deepfilter_output(command, input_wav)
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
