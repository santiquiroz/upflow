from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import pytest

from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource


class FakeDecodeProc:
    """Popen fake: stdout con frames rgb24 crudos pre-armados, stderr fake."""

    def __init__(self, stdout_bytes: bytes, returncode: int = 0) -> None:
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(b"fake ffmpeg stderr line")
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def make_source(tmp_path: Path, width: int = 4, height: int = 2) -> FfmpegFrameSource:
    return FfmpegFrameSource(Path("ffmpeg.exe"), tmp_path / "clip.mp4", width, height, decode_threads=2)


def raw_frames(count: int, width: int, height: int) -> bytes:
    # Frame i = todos los bytes en (i % 256): orden verificable por el primer byte.
    return b"".join(bytes([i % 256]) * (width * height * 3) for i in range(count))


def test_build_command_uses_fps_mode_passthrough_never_vsync(tmp_path: Path) -> None:
    command = make_source(tmp_path).build_command()
    assert "-vsync" not in command  # flag deprecado, prohibido por el spec
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"
    assert command[command.index("-f") + 1] == "rawvideo"
    assert command[-1] == "pipe:1"


def test_frames_yields_nhwc_uint8_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(3, 4, 2))
    monkeypatch.setattr(source, "_spawn", lambda command: fake)

    frames = list(source.frames(threading.Event()))

    assert len(frames) == 3
    assert all(f.shape == (1, 2, 4, 3) and f.dtype == np.uint8 for f in frames)
    assert [int(f[0, 0, 0, 0]) for f in frames] == [0, 1, 2]


def test_frames_raises_on_truncated_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(1, 4, 2) + b"\x00" * 5)  # 5 bytes sueltos al final
    monkeypatch.setattr(source, "_spawn", lambda command: fake)

    with pytest.raises(RuntimeError, match="truncado"):
        list(source.frames(threading.Event()))
    assert fake.killed is True  # el proceso no queda huérfano tras el error


def test_frames_raises_when_ffmpeg_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(2, 4, 2), returncode=1)
    monkeypatch.setattr(source, "_spawn", lambda command: fake)

    with pytest.raises(RuntimeError) as exc_info:
        list(source.frames(threading.Event()))
    assert str(exc_info.value) == (
        "ffmpeg falló al decodificar (código de salida 1): fake ffmpeg stderr line"
    )


def test_frames_kills_process_when_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(10, 4, 2))
    monkeypatch.setattr(source, "_spawn", lambda command: fake)
    cancel = threading.Event()

    iterator = source.frames(cancel)
    next(iterator)
    cancel.set()
    remaining = list(iterator)

    assert remaining == []
    assert fake.killed is True
