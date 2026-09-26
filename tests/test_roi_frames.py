from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services import roi_frames
from app.services.cctv_frame_index import FrameEntry
from app.services.video_analysis import VideoAnalysisError

FFMPEG = Path("ffmpeg.exe")
VIDEO = Path("work.mkv")


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_the_range_is_selected_by_output_frame_number_after_the_prefilters() -> None:
    argv = roi_frames.build_roi_decode_command(FFMPEG, VIDEO, 10, 19, ("bwdif=mode=send_frame", "pp7"))
    assert value_after(argv, "-vf") == "bwdif=mode=send_frame,pp7,select=between(n\\,10\\,19),format=gray"


def test_the_decode_stops_after_the_last_frame_of_the_range_and_keeps_every_frame() -> None:
    argv = roi_frames.build_roi_decode_command(FFMPEG, VIDEO, 10, 19)
    assert value_after(argv, "-frames:v") == "10"
    assert value_after(argv, "-fps_mode") == "passthrough"
    assert value_after(argv, "-pix_fmt") == "gray"
    assert argv[-3:] == ["-f", "rawvideo", "-"]


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(roi_frames.RoiDecodeError):
        roi_frames.build_roi_decode_command(FFMPEG, VIDEO, 5, 4)


def test_raw_gray_bytes_become_float32_frames() -> None:
    buffer = bytes(range(12)) * 2
    stack = roi_frames.split_gray_frames(buffer, width=4, height=3, count=2)
    assert stack.shape == (2, 3, 4)
    assert stack.dtype == np.float32
    assert stack[1, 2, 3] == 11.0


def test_a_short_decode_is_an_error() -> None:
    with pytest.raises(roi_frames.RoiDecodeError) as error:
        roi_frames.split_gray_frames(bytes(12), width=4, height=3, count=2)
    assert error.value.key == roi_frames.ROI_DECODE_FAILED


def test_decoded_frames_carry_their_number_and_pict_type_from_the_index() -> None:
    index = tuple(FrameEntry(n, n * 0.04, n == 0, "I" if n == 0 else "P", 100) for n in range(6))
    stack = np.zeros((3, 2, 2), dtype=np.float32)
    frames = roi_frames.roi_frames_from(stack, 3, index)
    assert [(frame.n, frame.pict_type) for frame in frames] == [(3, "P"), (4, "P"), (5, "P")]


def test_a_range_past_the_index_is_an_error() -> None:
    index = (FrameEntry(0, 0.0, True, "I", 100),)
    with pytest.raises(roi_frames.RoiDecodeError):
        roi_frames.roi_frames_from(np.zeros((2, 2, 2), dtype=np.float32), 0, index)


async def test_a_failed_ffmpeg_run_raises_with_the_pass_name() -> None:
    async def failing(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"boom", 1

    with pytest.raises(VideoAnalysisError) as error:
        await roi_frames.decode_roi_frames(FFMPEG, VIDEO, 0, 1, 4, 3, run=failing)
    assert roi_frames.ROI_DECODE_PASS in str(error.value)
