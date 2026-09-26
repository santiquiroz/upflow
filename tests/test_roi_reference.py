from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.cctv_chain import CctvChainError
from app.services.cctv_frame_index import FrameEntry
from app.services.cctv_preview import PreviewSource
from app.services.ffmpeg_filters import FrameGeometry
from app.services.roi_reference import ReferenceRequest, check_reference_request, suggest_session_reference

WIDTH, HEIGHT, FRAMES = 32, 24, 12
BOX = (8, 6, 16, 12)


def source() -> PreviewSource:
    frames = tuple(FrameEntry(n, n * 0.04, n == 0, "P", 100) for n in range(FRAMES))
    return PreviewSource(Path("work.mkv"), frames, FrameGeometry(WIDTH, HEIGHT))


def flat_frame(value: int = 90) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), value, dtype=np.uint8)


def textured_frame() -> np.ndarray:
    frame = flat_frame()
    frame[6:18, 8:24] = np.indices((12, 16)).sum(axis=0) % 2 * 120 + 60
    return frame


def fake_decoder(frames: list[np.ndarray], seen: list[list[str]]):
    async def run(command, timeout, **_):
        seen.append(list(command))
        return b"".join(frame.tobytes() for frame in frames), b"", 0

    return run


async def test_the_sharpest_unsaturated_frame_in_the_box_is_suggested() -> None:
    frames = [flat_frame() for _ in range(5)]
    frames[3] = textured_frame()
    seen: list[list[str]] = []

    frame = await suggest_session_reference(
        Path("ffmpeg.exe"), source(), ReferenceRequest(4, 8, BOX), (), run=fake_decoder(frames, seen)
    )

    assert frame == 7
    assert r"select=between(n\,4\,8)" in seen[0][seen[0].index("-vf") + 1]


async def test_a_saturated_frame_loses_to_a_sharp_clean_one() -> None:
    frames = [flat_frame() for _ in range(3)]
    frames[0] = textured_frame()
    frames[0][6:18, 8:24] = np.where(frames[0][6:18, 8:24] > 100, 255, 0)
    frames[2] = textured_frame()

    frame = await suggest_session_reference(
        Path("ffmpeg.exe"), source(), ReferenceRequest(0, 2, BOX), (), run=fake_decoder(frames, [])
    )

    assert frame == 2


@pytest.mark.parametrize(
    ("request_", "key"),
    [
        (ReferenceRequest(5, 4, BOX), "cctv.error.roiFrames"),
        (ReferenceRequest(0, 11, BOX), "cctv.error.roiFrames"),
        (ReferenceRequest(0, 3, (20, 6, 16, 12)), "cctv.error.roiOutsideFrame"),
        (ReferenceRequest(0, 3, (8, 6, 15, 12)), "cctv.error.roiOdd"),
    ],
)
def test_the_range_and_the_box_are_checked_like_the_job(request_: ReferenceRequest, key: str) -> None:
    with pytest.raises(CctvChainError) as caught:
        check_reference_request(request_, source(), max_frames=10)

    assert caught.value.code == key
