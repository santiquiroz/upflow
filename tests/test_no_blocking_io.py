from __future__ import annotations

import asyncio
import io
import subprocess
import time
from pathlib import Path

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from app.config import Settings
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


async def test_ffprobe_json_does_not_block_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    media_tools = MediaTools(settings)
    monkeypatch.setattr(media_tools, "available", lambda: True)

    def slow_ffprobe(source_path: Path) -> subprocess.CompletedProcess[str]:
        time.sleep(BLOCKING_DELAY_SECONDS)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout='{"streams": []}', stderr="")

    monkeypatch.setattr(media_tools, "_run_ffprobe", slow_ffprobe)

    ticks = await count_heartbeats_during(media_tools.ffprobe_json(Path("fake-source.mp4")))

    assert ticks >= HEARTBEAT_TICKS // 2, (
        "the event loop was blocked while ffprobe ran: heartbeat coroutine barely progressed"
    )


async def test_validate_input_image_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    jobs = JobManager(settings, FakeEngine())

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
