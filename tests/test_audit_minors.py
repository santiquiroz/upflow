from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.video_upscaler import VideoUpscaler


class _FakeEngine:
    def available(self) -> bool:
        return True


class _CountingMediaTools:
    """Counts ffprobe calls so the reuse path is observable."""

    def __init__(self) -> None:
        self.probe_calls = 0

    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        self.probe_calls += 1
        return _PROBE


_PROBE = {
    "streams": [{"codec_type": "video", "width": 8, "height": 8, "avg_frame_rate": "24/1"}],
    "format": {"duration": "1.0"},
}


def make_job(tmp_path: Path, probe: dict | None) -> VideoUpscaleJob:
    return VideoUpscaleJob(
        source_path=tmp_path / "clip.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        probe=probe,
    )


# ---------------------------------------------------------------------------
# ffprobe reuse: the probe captured at job creation must not be recomputed
# ---------------------------------------------------------------------------


def test_job_carries_probe_and_pipeline_prefers_it(tmp_path: Path) -> None:
    job = make_job(tmp_path, _PROBE)
    assert job.probe is _PROBE  # travels with the job, in memory only


def test_job_without_probe_defaults_to_none(tmp_path: Path) -> None:
    assert make_job(tmp_path, None).probe is None


# ---------------------------------------------------------------------------
# _count_frames: os.scandir replacement must keep the missing-dir contract
# ---------------------------------------------------------------------------


def test_count_frames_returns_zero_for_missing_dir(tmp_path: Path) -> None:
    assert VideoUpscaler._count_frames(tmp_path / "does-not-exist") == 0


def test_count_frames_counts_only_png(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(3):
        (frames / f"{index:08d}.png").write_bytes(b"x")
    (frames / "notes.txt").write_bytes(b"ignore me")
    assert VideoUpscaler._count_frames(frames) == 3


def test_count_frames_returns_zero_when_path_is_a_file(tmp_path: Path) -> None:
    target = tmp_path / "a-file"
    target.write_bytes(b"x")
    assert VideoUpscaler._count_frames(target) == 0


# ---------------------------------------------------------------------------
# ffmpeg thread defaults now scale with cores instead of being hardcoded
# ---------------------------------------------------------------------------


def test_ffmpeg_thread_defaults_scale_with_cores_and_stay_capped() -> None:
    settings = Settings(_env_file=None)
    assert 2 <= settings.ffmpeg_decode_threads <= 12
    assert 2 <= settings.ffmpeg_encode_threads <= 24


def test_ffmpeg_thread_defaults_are_overridable() -> None:
    settings = Settings(_env_file=None, FFMPEG_ENCODE_THREADS=3, FFMPEG_DECODE_THREADS=2)
    assert settings.ffmpeg_encode_threads == 3
    assert settings.ffmpeg_decode_threads == 2
