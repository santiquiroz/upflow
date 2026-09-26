from __future__ import annotations

import io
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.cctv_chain import ai_lane_plan, steps_from_request
from app.services.cctv_enhance_plan import prefilter_args
from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource
from app.services.ffmpeg_filters import output_dims_after
from ffmpeg_support import needs_ffmpeg


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


def make_source(
    tmp_path: Path, width: int = 4, height: int = 2, fps: str = "24/1"
) -> FfmpegFrameSource:
    return FfmpegFrameSource(
        Path("ffmpeg.exe"), tmp_path / "clip.mp4", width, height, decode_threads=2, fps=fps
    )


def raw_frames(count: int, width: int, height: int) -> bytes:
    # Frame i = todos los bytes en (i % 256): orden verificable por el primer byte.
    return b"".join(bytes([i % 256]) * (width * height * 3) for i in range(count))


def test_build_command_normalises_to_cfr_never_vsync(tmp_path: Path) -> None:
    # Antes era passthrough, que conservaba la cadencia VFR de la fuente
    # mientras el encode asumia fps fijo: el video salia mas corto que el audio
    # y los subtitulos (177 s de deriva en 25 min sobre material real).
    command = make_source(tmp_path, fps="30000/1001").build_command()
    assert "-vsync" not in command  # flag deprecado, prohibido por el spec
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert command[command.index("-r") + 1] == "30000/1001"
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


# --- Modo CCTV: pasos pre-IA en el -vf del decode (spec §4.7) ---


def test_cctv_prefilter_goes_after_the_input_and_before_the_cfr_rate(tmp_path: Path) -> None:
    source = FfmpegFrameSource(
        Path("ffmpeg.exe"), tmp_path / "work.mkv", 320, 240, decode_threads=2, fps="25/2",
        prefilter_args=("-vf", "crop=w=320:h=240:x=0:y=0:exact=1"),
    )  # fmt: skip

    command = source.build_command()

    vf = command.index("-vf")
    assert command.index("-i") < vf < command.index("-fps_mode")
    assert command[vf + 1] == "crop=w=320:h=240:x=0:y=0:exact=1"
    assert command[command.index("-r") + 1] == "25/2"


def test_cctv_without_prefilter_the_decode_command_is_unchanged(tmp_path: Path) -> None:
    command = make_source(tmp_path, fps="24/1").build_command()

    assert command == [
        "ffmpeg.exe", "-v", "error", "-threads", "2", "-i", str(tmp_path / "clip.mp4"),
        "-fps_mode", "cfr", "-r", "24/1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]  # fmt: skip


def make_cctv_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "camera.mkv"
    command = [
        str(Settings().ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=25", "-frames:v", "50",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return clip


def cctv_decode(tmp_path: Path, raw_steps: list[dict]) -> list[np.ndarray]:
    lane = ai_lane_plan(steps_from_request(raw_steps, "ai"))
    decoded = output_dims_after(lane.decode, 128, 96)
    source = FfmpegFrameSource(
        Settings().ffmpeg_binary_path, make_cctv_clip(tmp_path), decoded.width, decoded.height,
        decode_threads=1, fps="25/1", prefilter_args=prefilter_args(lane.decode),
    )  # fmt: skip
    return list(source.frames(threading.Event()))


@needs_ffmpeg
def test_cctv_real_trim_yields_only_the_kept_frames(tmp_path: Path) -> None:
    trim = {"id": "trim", "params": {"start_frame": 25, "end_frame": 34}}

    frames = cctv_decode(tmp_path, [trim])

    assert len(frames) == 10


@needs_ffmpeg
def test_cctv_real_crop_decodes_whole_frames_at_the_cropped_size(tmp_path: Path) -> None:
    crop = {"id": "crop", "params": {"x": 16, "y": 8, "w": 64, "h": 48}}

    frames = cctv_decode(tmp_path, [crop])

    assert len(frames) == 50
    assert all(frame.shape == (1, 48, 64, 3) for frame in frames)
