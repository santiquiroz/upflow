"""Decodificacion del rango de cuadros de la fusion de ROI (spec §4.9 paso 2).

Saca `[a, b]` de `work.mkv` por pipe `rawvideo` en luma de 8 bits y lo pasa a
float32. Los prefiltros (solo `deinterlace` y `deblock` clasicos, en ese
orden) van antes del `select`: el desentrelazado mira los cuadros vecinos y el
numero de cuadro tiene que ser el de salida. El `pict_type` de cada cuadro sale
del indice de la sesion (§4.2), no de este paso.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.services.cctv_frame_index import FrameEntry
from app.services.process_runner import run_guarded_process
from app.services.roi_registration import RoiFrame
from app.services.video_analysis import ProcessRunner, run_pass

ROI_DECODE_PASS = "roi-decode"
ROI_DECODE_FAILED = "cctv.error.roiDecode"
DECODE_TIMEOUT_SECONDS = 600.0


class RoiDecodeError(RuntimeError):
    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key


def frame_count(first: int, last: int) -> int:
    if first < 0 or last < first:
        raise RoiDecodeError(ROI_DECODE_FAILED, f"invalid frame range {first}..{last}")
    return last - first + 1


def select_range_filter(first: int, last: int) -> str:
    return f"select=between(n\\,{first}\\,{last})"


def roi_decode_vf(first: int, last: int, prefilters: Sequence[str]) -> str:
    return ",".join((*prefilters, select_range_filter(first, last), "format=gray"))


def build_roi_decode_command(
    ffmpeg: Path, video: Path, first: int, last: int, prefilters: Sequence[str] = ()
) -> list[str]:
    count = frame_count(first, last)
    return [
        str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-v", "error",
        "-i", str(video), "-map", "0:v:0",
        "-vf", roi_decode_vf(first, last, prefilters),
        "-fps_mode", "passthrough", "-frames:v", str(count),
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]  # fmt: skip


def split_gray_frames(buffer: bytes, width: int, height: int, count: int) -> np.ndarray:
    expected = width * height * count
    if len(buffer) != expected:
        raise RoiDecodeError(ROI_DECODE_FAILED, f"decoded {len(buffer)} bytes, expected {expected}")
    return np.frombuffer(buffer, dtype=np.uint8).reshape(count, height, width).astype(np.float32)


def roi_frames_from(stack: np.ndarray, first: int, index: Sequence[FrameEntry]) -> tuple[RoiFrame, ...]:
    last = first + len(stack) - 1
    if last >= len(index):
        raise RoiDecodeError(ROI_DECODE_FAILED, f"frame {last} is past the frame index ({len(index)} frames)")
    return tuple(RoiFrame(first + i, index[first + i].pict_type, pixels) for i, pixels in enumerate(stack))


async def decode_roi_frames(
    ffmpeg: Path,
    video: Path,
    first: int,
    last: int,
    width: int,
    height: int,
    prefilters: Sequence[str] = (),
    run: ProcessRunner = run_guarded_process,
    timeout: float = DECODE_TIMEOUT_SECONDS,
) -> np.ndarray:
    command = build_roi_decode_command(ffmpeg, video, first, last, prefilters)
    stdout, _ = await run_pass(ROI_DECODE_PASS, command, run, timeout)
    return split_gray_frames(stdout, width, height, frame_count(first, last))
