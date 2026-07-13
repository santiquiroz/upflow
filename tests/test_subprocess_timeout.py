from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from app.config import Settings
from app.models import UpscaleJob
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.process_runner import SubprocessTimeoutError, run_guarded_process
from app.services.video_upscaler import VideoUpscaler


def make_settings(tmp_path: Path, subprocess_timeout: float = 3600) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), SUBPROCESS_TIMEOUT=subprocess_timeout)


def make_sleep_command(sleep_seconds: float, exit_code: int = 0) -> list[str]:
    script = f"import time, sys; time.sleep({sleep_seconds}); sys.exit({exit_code})"
    return [sys.executable, "-c", script]


async def record_spawned_processes(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.subprocess.Process]:
    """Wraps asyncio.create_subprocess_exec so tests can inspect the real child after the fact."""
    spawned: list[asyncio.subprocess.Process] = []
    real_spawn = asyncio.create_subprocess_exec

    async def recording_spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await real_spawn(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording_spawn)
    return spawned


async def test_run_guarded_process_returns_output_on_normal_completion() -> None:
    command = [sys.executable, "-c", "print('hello')"]

    stdout, stderr, returncode = await run_guarded_process(command, timeout=5)

    assert returncode == 0
    assert b"hello" in stdout


async def test_run_guarded_process_raises_on_nonzero_exit_without_treating_it_as_timeout() -> None:
    command = make_sleep_command(sleep_seconds=0, exit_code=7)

    stdout, stderr, returncode = await run_guarded_process(command, timeout=5)

    assert returncode == 7


async def test_run_guarded_process_kills_and_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = await record_spawned_processes(monkeypatch)
    command = make_sleep_command(sleep_seconds=30)

    with pytest.raises(SubprocessTimeoutError, match="timed out after 0.2s"):
        await run_guarded_process(command, timeout=0.2)

    assert len(spawned) == 1
    assert spawned[0].returncode is not None, "hanging process was left running after the timeout"


async def test_run_guarded_process_kills_process_on_task_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned = await record_spawned_processes(monkeypatch)
    command = make_sleep_command(sleep_seconds=30)

    task = asyncio.create_task(run_guarded_process(command, timeout=30))
    while not spawned:
        await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert spawned[0].returncode is not None, "process was left running after task cancellation"


async def test_video_upscaler_run_process_uses_shared_runner_with_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path, subprocess_timeout=123)
    upscaler = VideoUpscaler(settings, engine=object(), media_tools=object())  # type: ignore[arg-type]

    calls: list[tuple[list[str], float]] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append((command, timeout))
        return b"", b"", 0

    monkeypatch.setattr("app.services.video_upscaler.run_guarded_process", fake_run_guarded_process)

    await upscaler._run_process(["ffmpeg", "-y"])

    assert calls == [(["ffmpeg", "-y"], 123)]


async def test_video_upscaler_run_process_raises_clear_error_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    upscaler = VideoUpscaler(settings, engine=object(), media_tools=object())  # type: ignore[arg-type]

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom: simulated ffmpeg failure\n", 1

    monkeypatch.setattr("app.services.video_upscaler.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="boom: simulated ffmpeg failure"):
        await upscaler._run_process(["ffmpeg"])


async def test_video_upscaler_run_process_propagates_timeout_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    upscaler = VideoUpscaler(settings, engine=object(), media_tools=object())  # type: ignore[arg-type]

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        raise SubprocessTimeoutError(f"Process 'ffmpeg' timed out after {timeout}s")

    monkeypatch.setattr("app.services.video_upscaler.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(SubprocessTimeoutError):
        await upscaler._run_process(["ffmpeg"])


def make_image_job(source_path: Path) -> UpscaleJob:
    return UpscaleJob(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )


async def test_realesrgan_engine_run_uses_shared_runner_with_configured_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path, subprocess_timeout=456)
    engine = RealEsrganNcnnEngine(settings)
    monkeypatch.setattr(engine, "available", lambda: True)

    job = make_image_job(tmp_path / "input.png")
    calls: list[float] = []

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        calls.append(timeout)
        output_path = settings.outputs_path / f"{job.id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output")
        return b"", b"", 0

    monkeypatch.setattr("app.services.engines.realesrgan_ncnn.run_guarded_process", fake_run_guarded_process)

    output_path = await engine.run(job)

    assert calls == [456]
    assert output_path.exists()


async def test_realesrgan_engine_run_raises_clear_error_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    engine = RealEsrganNcnnEngine(settings)
    monkeypatch.setattr(engine, "available", lambda: True)

    job = make_image_job(tmp_path / "input.png")

    async def fake_run_guarded_process(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom: simulated engine failure\n", 1

    monkeypatch.setattr("app.services.engines.realesrgan_ncnn.run_guarded_process", fake_run_guarded_process)

    with pytest.raises(RuntimeError, match="boom: simulated engine failure"):
        await engine.run(job)
