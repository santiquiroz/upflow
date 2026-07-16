from __future__ import annotations

import asyncio
import io
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from app.config import Settings
from app.services import media_tools as media_tools_module
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.media_tools import MediaTools
from app.services.storage import StorageService

BLOCKING_DELAY_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 0.02
HEARTBEAT_TICKS = 15


class FakeEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job) -> Path:  # type: ignore[no-untyped-def]
        return job.source_path


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def count_heartbeats_during(coro: "asyncio.Future") -> int:
    """Runs coro concurrently with a fast-ticking heartbeat and returns how many ticks fired."""
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        for _ in range(HEARTBEAT_TICKS):
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            ticks += 1

    heartbeat_task = asyncio.create_task(heartbeat())
    await coro
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    return ticks


def make_fake_ffprobe_command(sleep_seconds: float, exit_code: int = 0) -> list[str]:
    script = (
        "import sys, time; "
        f"time.sleep({sleep_seconds}); "
        "print('{\"streams\": []}'); "
        f"sys.exit({exit_code})"
    )
    return [sys.executable, "-c", script]


def make_media_tools_with_fake_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> MediaTools:
    media_tools = MediaTools(make_settings(tmp_path))
    monkeypatch.setattr(media_tools, "available", lambda: True)
    monkeypatch.setattr(media_tools, "_build_ffprobe_command", lambda source_path: command)
    return media_tools


async def test_ffprobe_json_does_not_block_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_tools = make_media_tools_with_fake_command(
        tmp_path, monkeypatch, make_fake_ffprobe_command(BLOCKING_DELAY_SECONDS)
    )

    ticks = await count_heartbeats_during(media_tools.ffprobe_json(Path("fake-source.mp4")))

    assert ticks >= HEARTBEAT_TICKS // 2, (
        "the event loop was blocked while ffprobe ran: heartbeat coroutine barely progressed"
    )


async def test_ffprobe_json_times_out_and_kills_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_tools = make_media_tools_with_fake_command(
        tmp_path, monkeypatch, make_fake_ffprobe_command(sleep_seconds=30)
    )
    monkeypatch.setattr(media_tools_module, "FFPROBE_TIMEOUT_SECONDS", 0.2)

    spawned: list[asyncio.subprocess.Process] = []
    real_spawn = asyncio.create_subprocess_exec

    async def recording_spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        process = await real_spawn(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", recording_spawn)

    with pytest.raises(RuntimeError, match="ffprobe timed out after 0.2s"):
        await media_tools.ffprobe_json(Path("fake-source.mp4"))

    assert len(spawned) == 1
    assert spawned[0].returncode is not None, "the ffprobe child process was left running after the timeout"


async def test_ffprobe_json_keeps_calledprocesserror_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_tools = make_media_tools_with_fake_command(
        tmp_path, monkeypatch, make_fake_ffprobe_command(sleep_seconds=0, exit_code=3)
    )

    with pytest.raises(subprocess.CalledProcessError):
        await media_tools.ffprobe_json(Path("fake-source.mp4"))


async def test_validate_input_image_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    jobs = JobManager(settings, FakeEngine(), DeviceSemaphores(settings))

    source_path = tmp_path / "input.png"
    source_path.write_bytes(make_png_bytes())

    real_image_open = Image.open

    def slow_image_open(path: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        time.sleep(BLOCKING_DELAY_SECONDS)
        return real_image_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", slow_image_open)

    ticks = await count_heartbeats_during(
        jobs.create_job(
            source_path=source_path,
            original_filename="input.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
        )
    )

    assert ticks >= HEARTBEAT_TICKS // 2, (
        "the event loop was blocked while validating the image: heartbeat coroutine barely progressed"
    )


async def test_save_upload_does_not_use_blocking_path_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    destination = tmp_path / "upload.bin"
    content = b"x" * (1024 * 1024 + 10)

    def guard_sync_path_open(self: Path, *args: object, **kwargs: object) -> object:
        raise AssertionError("StorageService.save_upload must not open the destination with sync Path.open")

    monkeypatch.setattr(Path, "open", guard_sync_path_open)

    upload = make_upload("upload.bin", content)
    await storage.save_upload(upload, destination)

    with open(destination, "rb") as handle:
        written = handle.read()

    assert written == content
