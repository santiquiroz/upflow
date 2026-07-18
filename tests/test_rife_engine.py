from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.engines.rife_ncnn import RifeNcnnEngine

# ---------------------------------------------------------------------------
# Task 11 (4.3) - RIFE interpolation engine wrapper: argv construction,
# availability, shared guarded runner reuse, output frame-count validation.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), **overrides)  # type: ignore[arg-type]


def make_fake_rife_install(tmp_path: Path, model_name: str = "rife-v4.6") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "rife-ncnn-vulkan.exe"
    binary.write_bytes(b"fake")
    models_dir = tmp_path / "models"
    (models_dir / model_name).mkdir(parents=True)
    return binary, models_dir


def make_available_settings(tmp_path: Path, model_name: str = "rife-v4.6", **overrides: object) -> Settings:
    binary, models_dir = make_fake_rife_install(tmp_path / "install", model_name)
    return make_settings(
        tmp_path,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(models_dir),
        RIFE_MODEL=model_name,
        **overrides,
    )


def write_fake_frames(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:08d}.png").write_bytes(b"fake-frame")


def write_png_with_dimensions(path: Path, width: int, height: int) -> None:
    import struct

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(signature + ihdr + b"\x08\x06\x00\x00\x00")


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------


def test_rife_engine_available_true_when_binary_and_model_exist(tmp_path: Path) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)

    assert engine.available() is True


def test_rife_engine_available_false_when_binary_missing(tmp_path: Path) -> None:
    _, models_dir = make_fake_rife_install(tmp_path / "install")
    settings = make_settings(
        tmp_path,
        RIFE_BINARY=str(tmp_path / "missing-rife.exe"),
        RIFE_MODELS_DIR=str(models_dir),
    )
    engine = RifeNcnnEngine(settings)

    assert engine.available() is False


def test_rife_engine_available_false_when_configured_model_folder_missing(tmp_path: Path) -> None:
    binary, models_dir = make_fake_rife_install(tmp_path / "install", model_name="rife-v4.25")
    settings = make_settings(
        tmp_path,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(models_dir),
        RIFE_MODEL="rife-v4.6",
    )
    engine = RifeNcnnEngine(settings)

    assert engine.available() is False


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multiplier,source_frame_count", [(2, 10), (3, 10), (4, 10)])
async def test_rife_engine_run_builds_expected_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, multiplier: int, source_frame_count: int
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    calls: list[list[str]] = []
    target_frame_count = source_frame_count * multiplier

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(command)
        write_fake_frames(frames_out, target_frame_count)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    await engine.run(frames_in, frames_out, source_frame_count, multiplier)

    assert len(calls) == 1
    assert calls[0] == [
        str(settings.rife_binary_path),
        "-i",
        str(frames_in),
        "-o",
        str(frames_out),
        "-m",
        str(settings.rife_models_path / settings.rife_model),
        "-n",
        str(target_frame_count),
        "-g",
        "0",
        "-j",
        engine._thread_spec(),
        "-f",
        "%08d.png",
    ]


@pytest.mark.parametrize(
    "device, expected_g",
    [("dml:0", "0"), ("dml:1", "1"), ("cpu", "0"), (None, "0")],
)
async def test_rife_engine_run_targets_job_device_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str | None, expected_g: str
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    calls: list[list[str]] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(command)
        write_fake_frames(frames_out, 4)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    await engine.run(frames_in, frames_out, 2, 2, device=device)

    gpu_arg = calls[0][calls[0].index("-g") + 1]
    assert gpu_arg == expected_g


# ---------------------------------------------------------------------------
# output dir lifecycle + shared runner reuse
# ---------------------------------------------------------------------------


async def test_rife_engine_run_creates_output_dir_before_invoking_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    dir_existed_at_call_time = False

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        nonlocal dir_existed_at_call_time
        dir_existed_at_call_time = frames_out.exists()
        write_fake_frames(frames_out, 20)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    await engine.run(frames_in, frames_out, source_frame_count=10, multiplier=2)

    assert dir_existed_at_call_time is True


async def test_rife_engine_run_uses_shared_runner_with_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path, SUBPROCESS_TIMEOUT=789)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    calls: list[float] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(timeout)
        write_fake_frames(frames_out, 20)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    await engine.run(frames_in, frames_out, source_frame_count=10, multiplier=2)

    assert calls == [789]


async def test_rife_engine_run_returns_frames_out_path_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_fake_frames(frames_out, 20)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    result = await engine.run(frames_in, frames_out, source_frame_count=10, multiplier=2)

    assert result == frames_out


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------


async def test_rife_engine_run_raises_when_not_available(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        RIFE_BINARY=str(tmp_path / "missing.exe"),
        RIFE_MODELS_DIR=str(tmp_path / "missing-models"),
    )
    engine = RifeNcnnEngine(settings)

    with pytest.raises(RuntimeError, match="not available"):
        await engine.run(tmp_path / "in", tmp_path / "out", source_frame_count=10, multiplier=2)


async def test_rife_engine_run_raises_clear_error_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom: simulated rife failure\n", 1

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="boom: simulated rife failure"):
        await engine.run(frames_in, frames_out, source_frame_count=10, multiplier=2)


async def test_rife_engine_run_raises_when_output_frame_count_is_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="no output frames"):
        await engine.run(frames_in, frames_out, source_frame_count=10, multiplier=2)


async def test_rife_engine_run_raises_when_output_frame_count_does_not_match_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_fake_frames(frames_out, 15)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="15"):
        await engine.run(frames_in, frames_out, source_frame_count=10, multiplier=2)


# ---------------------------------------------------------------------------
# Task 15 (6.6) - explicit target_frame_count (TARGET_FPS mode), keeping
# multiplier-based callers working unchanged.
# ---------------------------------------------------------------------------


async def test_rife_engine_run_accepts_explicit_target_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    calls: list[list[str]] = []
    target_frame_count = 250

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(command)
        write_fake_frames(frames_out, target_frame_count)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    await engine.run(frames_in, frames_out, source_frame_count=100, target_frame_count=target_frame_count)

    assert calls[0][calls[0].index("-n") + 1] == str(target_frame_count)


async def test_rife_engine_run_target_frame_count_overrides_multiplier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    calls: list[list[str]] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(command)
        write_fake_frames(frames_out, 250)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    await engine.run(frames_in, frames_out, source_frame_count=100, multiplier=2, target_frame_count=250)

    assert calls[0][calls[0].index("-n") + 1] == "250"


async def test_rife_engine_run_validates_output_against_target_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_available_settings(tmp_path)
    engine = RifeNcnnEngine(settings)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_in.mkdir()

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        write_fake_frames(frames_out, 100)
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.rife_ncnn.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="100"):
        await engine.run(frames_in, frames_out, source_frame_count=100, target_frame_count=250)


# ---------------------------------------------------------------------------
# Perf flags (fix del interpolado eterno): -j threads escalados + -u UHD auto
# ---------------------------------------------------------------------------


def test_thread_spec_scales_with_cpu_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = RifeNcnnEngine(make_available_settings(tmp_path))

    monkeypatch.setattr("app.services.engines.rife_ncnn.os.cpu_count", lambda: 16)
    assert engine._thread_spec() == "8:4:12"

    monkeypatch.setattr("app.services.engines.rife_ncnn.os.cpu_count", lambda: 4)
    assert engine._thread_spec() == "2:2:4"

    monkeypatch.setattr("app.services.engines.rife_ncnn.os.cpu_count", lambda: None)
    assert engine._thread_spec() == "2:2:4"


def test_thread_spec_honors_override_setting(tmp_path: Path) -> None:
    engine = RifeNcnnEngine(make_available_settings(tmp_path, RIFE_THREADS="3:3:3"))

    assert engine._thread_spec() == "3:3:3"


def test_rife_threads_setting_rejects_malformed_spec() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, RIFE_THREADS="muchos")
    with pytest.raises(ValueError):
        Settings(_env_file=None, RIFE_THREADS="0:2:2")


def test_uhd_flag_added_for_4k_frames(tmp_path: Path) -> None:
    engine = RifeNcnnEngine(make_available_settings(tmp_path))
    frames_in = tmp_path / "frames-4k"
    write_png_with_dimensions(frames_in / "00000001.png", 3840, 2160)

    command = engine._build_command(frames_in, tmp_path / "out", 10, "0")

    assert "-u" in command


def test_uhd_flag_absent_for_1080p_frames(tmp_path: Path) -> None:
    engine = RifeNcnnEngine(make_available_settings(tmp_path))
    frames_in = tmp_path / "frames-hd"
    write_png_with_dimensions(frames_in / "00000001.png", 1920, 1080)

    command = engine._build_command(frames_in, tmp_path / "out", 10, "0")

    assert "-u" not in command


def test_uhd_mode_setting_forces_on_and_off(tmp_path: Path) -> None:
    frames_small = tmp_path / "frames-small"
    write_png_with_dimensions(frames_small / "00000001.png", 640, 480)
    forced_on = RifeNcnnEngine(make_available_settings(tmp_path, RIFE_UHD_MODE="on"))
    assert "-u" in forced_on._build_command(frames_small, tmp_path / "o1", 10, "0")

    frames_big = tmp_path / "frames-big"
    write_png_with_dimensions(frames_big / "00000001.png", 5120, 2880)
    forced_off = RifeNcnnEngine(make_available_settings(tmp_path / "b", RIFE_UHD_MODE="off"))
    assert "-u" not in forced_off._build_command(frames_big, tmp_path / "o2", 10, "0")


def test_uhd_auto_tolerates_unparseable_frames(tmp_path: Path) -> None:
    engine = RifeNcnnEngine(make_available_settings(tmp_path))
    frames_in = tmp_path / "frames-bad"
    frames_in.mkdir()
    (frames_in / "00000001.png").write_bytes(b"not-a-png")

    command = engine._build_command(frames_in, tmp_path / "out", 10, "0")

    assert "-u" not in command
