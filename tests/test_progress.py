from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.api.routes import job_to_response, video_job_to_response
from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.progress import (
    Stage,
    advance_image_stage,
    advance_video_stage,
    apply_image_tile_progress,
    apply_stage_transition,
    build_image_stages,
    build_video_stages,
    compute_progress,
    complete_image_stages,
    complete_video_stages,
    mark_all_done,
    resolve_frames_total,
)
from app.services.storage import StorageService
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# SP5 Task 1 - weighted stage model + framesTotal + progress in job responses.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_video_job(source_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )
    fields.update(overrides)
    return VideoUpscaleJob(**fields)


def make_image_job(source_path: Path, **overrides: object) -> UpscaleJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )
    fields.update(overrides)
    return UpscaleJob(**fields)


# ---------------------------------------------------------------------------
# build_video_stages: active-stage filtering + weight normalization
# ---------------------------------------------------------------------------


def test_build_video_stages_excludes_audio_and_interpolation_by_default(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    stages = build_video_stages(job)

    keys = [stage.key for stage in stages]
    assert keys == ["probing", "extracting_frames", "upscaling_frames", "encoding_video"]


def test_build_video_stages_includes_extracting_audio_when_keep_audio(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", keep_audio=True)

    stages = build_video_stages(job)

    keys = [stage.key for stage in stages]
    assert "extracting_audio" in keys
    assert "enhancing_audio" not in keys


def test_build_video_stages_includes_enhancing_audio_when_audio_enhance_set(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", keep_audio=True, audio_enhance="deepfilternet")

    stages = build_video_stages(job)

    keys = [stage.key for stage in stages]
    assert keys.index("extracting_audio") < keys.index("enhancing_audio")


def test_build_video_stages_excludes_enhancing_audio_without_keep_audio(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", keep_audio=False, audio_enhance="deepfilternet")

    stages = build_video_stages(job)

    assert "enhancing_audio" not in [stage.key for stage in stages]


def test_build_video_stages_includes_interpolation_when_fps_multiplier_over_one(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", fps_multiplier=2)

    stages = build_video_stages(job)

    assert "interpolating_frames" in [stage.key for stage in stages]


def test_build_video_stages_includes_interpolation_when_target_fps_set(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", target_fps="60")

    stages = build_video_stages(job)

    assert "interpolating_frames" in [stage.key for stage in stages]


def test_build_video_stages_excludes_interpolation_when_multiplier_is_one(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", fps_multiplier=1)

    stages = build_video_stages(job)

    assert "interpolating_frames" not in [stage.key for stage in stages]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"keep_audio": True},
        {"keep_audio": True, "audio_enhance": "deepfilternet"},
        {"fps_multiplier": 2},
        {"target_fps": "60"},
        {"keep_audio": True, "audio_enhance": "deepfilternet", "fps_multiplier": 2},
    ],
)
def test_build_video_stages_weights_normalize_to_one(tmp_path: Path, overrides: dict) -> None:
    job = make_video_job(tmp_path / "clip.mp4", **overrides)

    stages = build_video_stages(job)

    assert sum(stage.weight for stage in stages) == pytest.approx(1.0)


def test_build_video_stages_start_all_pending(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    stages = build_video_stages(job)

    assert all(stage.status == "pending" for stage in stages)


# ---------------------------------------------------------------------------
# build_image_stages: coarse two-stage model
# ---------------------------------------------------------------------------


def test_build_image_stages_has_validating_then_upscaling() -> None:
    stages = build_image_stages()

    assert [stage.key for stage in stages] == ["validating", "upscaling"]
    assert sum(stage.weight for stage in stages) == pytest.approx(1.0)
    assert stages[0].weight == pytest.approx(0.1)
    assert stages[1].weight == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# apply_stage_transition / mark_all_done
# ---------------------------------------------------------------------------


def test_apply_stage_transition_marks_previous_done_current_active_rest_pending() -> None:
    stages = [
        Stage(key="a", label="A", weight=0.2),
        Stage(key="b", label="B", weight=0.3),
        Stage(key="c", label="C", weight=0.5),
    ]

    transitioned = apply_stage_transition(stages, "b")

    statuses = {stage.key: stage.status for stage in transitioned}
    assert statuses == {"a": "done", "b": "active", "c": "pending"}


def test_apply_stage_transition_unknown_key_returns_stages_unchanged() -> None:
    stages = [Stage(key="a", label="A", weight=1.0)]

    transitioned = apply_stage_transition(stages, "completed")

    assert transitioned == stages


def test_mark_all_done_sets_every_stage_to_done() -> None:
    stages = [Stage(key="a", label="A", weight=0.5, status="active"), Stage(key="b", label="B", weight=0.5)]

    done_stages = mark_all_done(stages)

    assert all(stage.status == "done" for stage in done_stages)


# ---------------------------------------------------------------------------
# compute_progress: pure weight math
# ---------------------------------------------------------------------------


def test_compute_progress_sums_done_stage_weights() -> None:
    stages = [
        Stage(key="a", label="A", weight=0.2, status="done"),
        Stage(key="b", label="B", weight=0.3, status="active"),
        Stage(key="c", label="C", weight=0.5, status="pending"),
    ]

    assert compute_progress(stages) == pytest.approx(0.2)


def test_compute_progress_includes_current_fraction_of_active_stage() -> None:
    stages = [
        Stage(key="a", label="A", weight=0.2, status="done"),
        Stage(key="b", label="B", weight=0.3, status="active"),
        Stage(key="c", label="C", weight=0.5, status="pending"),
    ]

    assert compute_progress(stages, current_fraction=0.5) == pytest.approx(0.35)


def test_compute_progress_all_done_is_one() -> None:
    stages = mark_all_done(build_image_stages())

    assert compute_progress(stages) == pytest.approx(1.0)


def test_compute_progress_all_pending_is_zero() -> None:
    stages = build_image_stages()

    assert compute_progress(stages) == pytest.approx(0.0)


def test_compute_progress_clamps_current_fraction_above_one() -> None:
    stages = [Stage(key="a", label="A", weight=1.0, status="active")]

    assert compute_progress(stages, current_fraction=5.0) == pytest.approx(1.0)


def test_compute_progress_is_monotonic_across_video_stage_transitions(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", keep_audio=True, fps_multiplier=2)
    stages = build_video_stages(job)
    keys = [stage.key for stage in stages]

    progress_values = []
    for key in keys:
        stages = apply_stage_transition(stages, key)
        progress_values.append(compute_progress(stages))
    progress_values.append(compute_progress(mark_all_done(stages)))

    assert progress_values == sorted(progress_values)
    assert progress_values[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# resolve_frames_total: nb_frames primary, duration*fps fallback, honest None
# ---------------------------------------------------------------------------


def test_resolve_frames_total_uses_nb_frames_when_present() -> None:
    probe = {"format": {"duration": "999"}}
    video_stream = {"nb_frames": "150"}

    assert resolve_frames_total(probe, video_stream, "30/1") == 150


def test_resolve_frames_total_falls_back_to_duration_times_fps() -> None:
    probe = {"format": {"duration": "5.0"}}
    video_stream = {}

    assert resolve_frames_total(probe, video_stream, "30/1") == 150


def test_resolve_frames_total_falls_back_when_nb_frames_is_na() -> None:
    probe = {"format": {"duration": "2.0"}}
    video_stream = {"nb_frames": "N/A"}

    assert resolve_frames_total(probe, video_stream, "30/1") == 60


def test_resolve_frames_total_falls_back_when_nb_frames_is_zero() -> None:
    probe = {"format": {"duration": "1.0"}}
    video_stream = {"nb_frames": "0"}

    assert resolve_frames_total(probe, video_stream, "30/1") == 30


def test_resolve_frames_total_none_when_duration_and_nb_frames_missing() -> None:
    probe: dict = {"format": {}}
    video_stream: dict = {}

    assert resolve_frames_total(probe, video_stream, "30/1") is None


def test_resolve_frames_total_none_when_fps_is_invalid() -> None:
    probe = {"format": {"duration": "5.0"}}
    video_stream: dict = {}

    assert resolve_frames_total(probe, video_stream, "0/0") is None


# ---------------------------------------------------------------------------
# advance_video_stage / complete_video_stages: job.metadata side effects
# ---------------------------------------------------------------------------


def test_advance_video_stage_populates_metadata(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    advance_video_stage(job, "probing")

    assert job.metadata["stage"] == "probing"
    assert job.metadata["progress"] == pytest.approx(0.0)
    assert [stage["status"] for stage in job.metadata["stages"]][0] == "active"
    assert job.metadata["framesDone"] == 0
    assert job.metadata["framesTotal"] is None
    assert job.metadata["stageStartedAt"]


def test_advance_video_stage_marks_earlier_stages_done(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    advance_video_stage(job, "probing")
    advance_video_stage(job, "extracting_frames")

    statuses = {stage["key"]: stage["status"] for stage in job.metadata["stages"]}
    assert statuses["probing"] == "done"
    assert statuses["extracting_frames"] == "active"
    assert job.metadata["progress"] > 0.0


def test_advance_video_stage_does_not_overwrite_frames_total_once_set(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    advance_video_stage(job, "probing")
    job.metadata["framesTotal"] = 150
    advance_video_stage(job, "extracting_frames")

    assert job.metadata["framesTotal"] == 150


def test_complete_video_stages_sets_progress_to_one_and_all_done(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    advance_video_stage(job, "probing")
    complete_video_stages(job)

    assert job.metadata["progress"] == pytest.approx(1.0)
    assert all(stage["status"] == "done" for stage in job.metadata["stages"])
    assert job.metadata["stage"] == "completed"


# ---------------------------------------------------------------------------
# advance_image_stage / complete_image_stages
# ---------------------------------------------------------------------------


def test_advance_image_stage_upscaling_marks_validating_done(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")

    advance_image_stage(job, "upscaling")

    statuses = {stage["key"]: stage["status"] for stage in job.metadata["stages"]}
    assert statuses == {"validating": "done", "upscaling": "active"}
    assert job.metadata["progress"] == pytest.approx(0.1)


def test_complete_image_stages_sets_progress_to_one(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")

    advance_image_stage(job, "upscaling")
    complete_image_stages(job)

    assert job.metadata["progress"] == pytest.approx(1.0)
    assert job.metadata["stage"] == "completed"


# ---------------------------------------------------------------------------
# apply_image_tile_progress (SP5 Task 4 - ONNX tile-based image progress)
# ---------------------------------------------------------------------------


def test_apply_image_tile_progress_sets_frame_counts(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")
    advance_image_stage(job, "upscaling")

    apply_image_tile_progress(job, tiles_done=1, tiles_total=4)

    assert job.metadata["framesDone"] == 1
    assert job.metadata["framesTotal"] == 4
    assert job.metadata["stage"] == "upscaling"


def test_apply_image_tile_progress_scales_within_upscaling_weight(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")
    advance_image_stage(job, "upscaling")

    apply_image_tile_progress(job, tiles_done=2, tiles_total=4)

    # validating(10%) done + upscaling(90%) at 50% fraction = 0.10 + 0.45
    assert job.metadata["progress"] == pytest.approx(0.55)


def test_apply_image_tile_progress_reaches_full_upscaling_weight_on_last_tile(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")
    advance_image_stage(job, "upscaling")

    apply_image_tile_progress(job, tiles_done=4, tiles_total=4)

    assert job.metadata["progress"] == pytest.approx(1.0)


def test_apply_image_tile_progress_is_monotonically_increasing_across_calls(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")
    advance_image_stage(job, "upscaling")

    progress_values = []
    for tiles_done in (1, 2, 3, 4):
        apply_image_tile_progress(job, tiles_done=tiles_done, tiles_total=4)
        progress_values.append(job.metadata["progress"])

    assert progress_values == sorted(progress_values)
    assert progress_values[0] < progress_values[-1]


# ---------------------------------------------------------------------------
# End-to-end: JobManager worker drives image stage transitions
# ---------------------------------------------------------------------------


class FakeImageEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        return job.source_path


async def test_job_manager_worker_advances_and_completes_image_stages(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeImageEngine(), DeviceSemaphores(settings))
    job = make_image_job(tmp_path / "in.png")
    manager.jobs[job.id] = job
    manager.queue.put_nowait(job)

    await manager.start()
    try:
        await manager.queue.join()
    finally:
        await manager.stop()

    assert job.status == JobStatus.completed
    assert job.metadata["stage"] == "completed"
    assert job.metadata["progress"] == pytest.approx(1.0)
    assert all(stage["status"] == "done" for stage in job.metadata["stages"])


# ---------------------------------------------------------------------------
# Video pipeline integration: fakes drive real advance_video_stage wiring
# ---------------------------------------------------------------------------


class FakeVideoEngine:
    def available(self) -> bool:
        return True


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 4,
                    "height": 4,
                    "avg_frame_rate": "30/1",
                    "nb_frames": "30",
                }
            ],
            "format": {"duration": "1.0"},
        }


class FakeMediaToolsWithAudio:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 4,
                    "height": 4,
                    "avg_frame_rate": "30/1",
                    "nb_frames": "30",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "1.0"},
        }


class ProgressTrackingVideoUpscaler(VideoUpscaler):
    """Fakes _run_process so no real ffmpeg/engine binary runs; records progress snapshots."""

    def __init__(self, *args: object, snapshots: list[tuple[str, float]], **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.snapshots = snapshots
        self._current_job: VideoUpscaleJob | None = None

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        self._current_job = job
        return await super().run(job, fps_multiplier=fps_multiplier)

    async def _run_process(self, command: list[str]) -> None:
        assert self._current_job is not None
        self.snapshots.append((self._current_job.metadata["stage"], self._current_job.metadata["progress"]))
        if "-fps_mode" in command:
            self._write_dummy_frame(command)
        elif "-vn" in command:
            self._write_dummy_audio(command)
        elif command[0] == str(self.settings.engine_binary_path):
            self._write_dummy_upscaled_frame(command)
        elif "-framerate" in command:
            self._write_dummy_output(command)

    @staticmethod
    def _write_dummy_frame(command: list[str]) -> None:
        frames_in_dir = Path(command[-1]).parent
        frames_in_dir.mkdir(parents=True, exist_ok=True)
        (frames_in_dir / "00000001.png").write_bytes(b"fake-frame-in")

    @staticmethod
    def _write_dummy_audio(command: list[str]) -> None:
        audio_path = Path(command[-1])
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake-audio")

    @staticmethod
    def _write_dummy_upscaled_frame(command: list[str]) -> None:
        frames_out_dir = Path(command[command.index("-o") + 1])
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        (frames_out_dir / "00000001.png").write_bytes(b"fake-frame-out")

    @staticmethod
    def _write_dummy_output(command: list[str]) -> None:
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output-video")


def make_progress_upscaler(
    tmp_path: Path,
    snapshots: list[tuple[str, float]],
    media_tools: object | None = None,
) -> ProgressTrackingVideoUpscaler:
    settings = make_settings(tmp_path)
    StorageService(settings)
    return ProgressTrackingVideoUpscaler(
        settings, FakeVideoEngine(), media_tools or FakeMediaTools(), snapshots=snapshots
    )


async def test_video_pipeline_progress_increases_monotonically_and_reaches_completion(tmp_path: Path) -> None:
    snapshots: list[tuple[str, float]] = []
    upscaler = make_progress_upscaler(tmp_path, snapshots)
    source = upscaler.settings.uploads_path / "clip.mp4"
    source.write_bytes(b"fake-video-bytes")
    job = make_video_job(source)

    await upscaler.run(job)

    progress_values = [value for _, value in snapshots]
    assert progress_values == sorted(progress_values)
    assert job.metadata["stage"] == "completed"
    assert job.metadata["progress"] == pytest.approx(1.0)
    assert all(stage["status"] == "done" for stage in job.metadata["stages"])


async def test_video_pipeline_sets_frames_total_from_nb_frames_after_probe(tmp_path: Path) -> None:
    snapshots: list[tuple[str, float]] = []
    upscaler = make_progress_upscaler(tmp_path, snapshots)
    source = upscaler.settings.uploads_path / "clip.mp4"
    source.write_bytes(b"fake-video-bytes")
    job = make_video_job(source)

    await upscaler.run(job)

    assert job.metadata["framesTotal"] == 30


# ---------------------------------------------------------------------------
# Reviewer fix: audio stages are phantom when keep_audio=True but the source
# has no audio track. build_video_stages must read the probed hasAudio flag so
# the stepper never shows Extract/Enhance audio steps that instantly complete.
# ---------------------------------------------------------------------------


def test_build_video_stages_excludes_audio_when_has_audio_false(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", keep_audio=True, audio_enhance="deepfilternet")
    job.metadata["hasAudio"] = False

    stages = build_video_stages(job)

    keys = [stage.key for stage in stages]
    assert "extracting_audio" not in keys
    assert "enhancing_audio" not in keys
    assert sum(stage.weight for stage in stages) == pytest.approx(1.0)


def test_build_video_stages_includes_audio_when_has_audio_true(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4", keep_audio=True)
    job.metadata["hasAudio"] = True

    stages = build_video_stages(job)

    assert "extracting_audio" in [stage.key for stage in stages]


async def test_video_pipeline_keep_audio_but_no_audio_track_excludes_audio_stages(tmp_path: Path) -> None:
    snapshots: list[tuple[str, float]] = []
    upscaler = make_progress_upscaler(tmp_path, snapshots, media_tools=FakeMediaTools())
    source = upscaler.settings.uploads_path / "clip.mp4"
    source.write_bytes(b"fake-video-bytes")
    job = make_video_job(source, keep_audio=True)

    await upscaler.run(job)

    keys = [stage["key"] for stage in job.metadata["stages"]]
    assert "extracting_audio" not in keys
    assert "enhancing_audio" not in keys
    progress_values = [value for _, value in snapshots]
    assert progress_values == sorted(progress_values)
    assert job.metadata["progress"] == pytest.approx(1.0)
    assert all(stage["status"] == "done" for stage in job.metadata["stages"])


async def test_video_pipeline_keep_audio_with_audio_track_activates_audio_stage(tmp_path: Path) -> None:
    snapshots: list[tuple[str, float]] = []
    upscaler = make_progress_upscaler(tmp_path, snapshots, media_tools=FakeMediaToolsWithAudio())
    source = upscaler.settings.uploads_path / "clip.mp4"
    source.write_bytes(b"fake-video-bytes")
    job = make_video_job(source, keep_audio=True)

    await upscaler.run(job)

    stage_snapshots = [stage for stage, _ in snapshots]
    keys = [stage["key"] for stage in job.metadata["stages"]]
    assert "extracting_audio" in keys
    assert "extracting_audio" in stage_snapshots  # the audio stage actually became active mid-run
    assert all(stage["status"] == "done" for stage in job.metadata["stages"])
    assert job.metadata["progress"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Response mapping: progress exposed as progressPct + metadata.stages
# ---------------------------------------------------------------------------


def test_job_to_response_exposes_progress_pct_from_metadata(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")
    advance_image_stage(job, "upscaling")

    response = job_to_response(job)

    assert response.progress_pct == pytest.approx(10.0)
    assert response.metadata["stages"][0]["key"] == "validating"


def test_job_to_response_progress_pct_is_none_without_metadata(tmp_path: Path) -> None:
    job = make_image_job(tmp_path / "in.png")

    response = job_to_response(job)

    assert response.progress_pct is None


def test_video_job_to_response_exposes_progress_pct_and_stages(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")
    advance_video_stage(job, "probing")
    job.metadata["framesTotal"] = 90

    response = video_job_to_response(job)
    serialized = response.model_dump(by_alias=True)

    assert response.progress_pct == pytest.approx(0.0)
    assert serialized["metadata"]["framesTotal"] == 90
    assert serialized["metadata"]["framesDone"] == 0
    assert len(serialized["metadata"]["stages"]) == 4


def test_video_job_to_response_reaches_full_progress_on_completion(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")
    advance_video_stage(job, "probing")
    complete_video_stages(job)

    response = video_job_to_response(job)

    assert response.progress_pct == pytest.approx(100.0)
    assert response.status != JobStatus.completed  # status is independent of stage bookkeeping


# ---------------------------------------------------------------------------
# SP5 Task 2 - live frame poller: _track_frame_progress advances framesDone
# and progress while a frame stage (extract/upscale/interpolate) is running.
# ---------------------------------------------------------------------------


def make_tracking_upscaler(tmp_path: Path, poll_interval: float = 0.01) -> VideoUpscaler:
    settings = make_settings(tmp_path)
    StorageService(settings)
    return VideoUpscaler(
        settings, FakeVideoEngine(), FakeMediaTools(), frame_poll_interval_seconds=poll_interval
    )


async def write_frames_incrementally(directory: Path, count: int, delay: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:08d}.png").write_bytes(b"frame")
        await asyncio.sleep(delay)


async def test_track_frame_progress_reports_increasing_frames_done_and_monotonic_progress(
    tmp_path: Path,
) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = 8
    advance_video_stage(job, "extracting_frames")
    output_dir = tmp_path / "frames-in"

    snapshots: list[tuple[int, float]] = []
    original_apply = upscaler._apply_frame_progress

    def spying_apply(
        job_arg: VideoUpscaleJob, stage_key: str, frames_done: int, frames_total: int | None = None
    ) -> None:
        original_apply(job_arg, stage_key, frames_done, frames_total)
        snapshots.append((job_arg.metadata["framesDone"], job_arg.metadata["progress"]))

    upscaler._apply_frame_progress = spying_apply  # type: ignore[method-assign]

    async with upscaler._track_frame_progress(job, output_dir, "extracting_frames"):
        await write_frames_incrementally(output_dir, count=8, delay=0.03)

    frames_done_values = [value for value, _ in snapshots]
    progress_values = [value for _, value in snapshots]

    assert len(snapshots) >= 2, "poller must tick at least once while the stage is still running"
    assert frames_done_values == sorted(frames_done_values)
    assert progress_values == sorted(progress_values)
    assert job.metadata["framesDone"] == 8


async def test_track_frame_progress_stage_progress_stays_within_active_stage_band(tmp_path: Path) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = 4
    advance_video_stage(job, "extracting_frames")
    stage_floor = job.metadata["progress"]
    stages = build_video_stages(job)
    extracting_weight = next(stage.weight for stage in stages if stage.key == "extracting_frames")
    output_dir = tmp_path / "frames-in"

    async with upscaler._track_frame_progress(job, output_dir, "extracting_frames"):
        await write_frames_incrementally(output_dir, count=4, delay=0.03)

    assert job.metadata["progress"] == pytest.approx(stage_floor + extracting_weight)


async def test_track_frame_progress_honest_floor_when_frames_total_unknown(tmp_path: Path) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = None
    advance_video_stage(job, "extracting_frames")
    stage_floor = job.metadata["progress"]
    output_dir = tmp_path / "frames-in"

    async with upscaler._track_frame_progress(job, output_dir, "extracting_frames"):
        await write_frames_incrementally(output_dir, count=3, delay=0.03)

    assert job.metadata["framesDone"] == 3
    assert job.metadata["progress"] == pytest.approx(stage_floor)


async def test_track_frame_progress_cleans_up_poller_and_does_not_mask_stage_error(tmp_path: Path) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = 4
    advance_video_stage(job, "extracting_frames")
    output_dir = tmp_path / "frames-in"

    tasks_before = asyncio.all_tasks()

    with pytest.raises(RuntimeError, match="stage exploded"):
        async with upscaler._track_frame_progress(job, output_dir, "extracting_frames"):
            await asyncio.sleep(0.03)
            raise RuntimeError("stage exploded")

    assert asyncio.all_tasks() == tasks_before


async def test_track_frame_progress_swallows_count_errors_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = 4
    advance_video_stage(job, "extracting_frames")
    output_dir = tmp_path / "frames-in"

    def broken_count(_directory: Path) -> int:
        raise OSError("disk hiccup")

    monkeypatch.setattr(upscaler, "_count_frames", broken_count)

    async with upscaler._track_frame_progress(job, output_dir, "extracting_frames"):
        await asyncio.sleep(0.05)

    assert job.metadata["framesDone"] == 0


class LiveFrameVideoUpscaler(VideoUpscaler):
    """Fakes _run_process with real async delays so the live poller has time
    to tick mid-stage; proves the pipeline actually wraps the three frame
    stages with the tracker instead of just being covered by direct unit
    tests against _track_frame_progress."""

    async def _run_process(self, command: list[str]) -> None:
        if "-fps_mode" in command:
            await write_frames_incrementally(Path(command[-1]).parent, count=4, delay=0.03)
        elif command[0] == str(self.settings.engine_binary_path):
            frames_out_dir = Path(command[command.index("-o") + 1])
            await write_frames_incrementally(frames_out_dir, count=4, delay=0.03)
        elif "-framerate" in command:
            output_path = Path(command[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-output-video")


class LiveFrameRifeEngine:
    async def run(
        self,
        frames_in: Path,
        frames_out: Path,
        source_frame_count: int,
        multiplier: int = 1,
        *,
        target_frame_count: int | None = None,
        device: str | None = None,
    ) -> Path:
        count = target_frame_count if target_frame_count is not None else source_frame_count * multiplier
        await write_frames_incrementally(frames_out, count=count, delay=0.03)
        return frames_out


async def test_pipeline_wraps_all_three_frame_stages_with_live_poller(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = LiveFrameVideoUpscaler(
        settings,
        FakeVideoEngine(),
        FakeMediaTools(),
        LiveFrameRifeEngine(),
        frame_poll_interval_seconds=0.01,
    )
    source = upscaler.settings.uploads_path / "clip.mp4"
    source.write_bytes(b"fake-video-bytes")
    job = make_video_job(source, fps_multiplier=2)

    stage_snapshots: list[tuple[str, int]] = []
    original_apply = upscaler._apply_frame_progress

    def spying_apply(
        job_arg: VideoUpscaleJob, stage_key: str, frames_done: int, frames_total: int | None = None
    ) -> None:
        original_apply(job_arg, stage_key, frames_done, frames_total)
        stage_snapshots.append((stage_key, job_arg.metadata["framesDone"]))

    upscaler._apply_frame_progress = spying_apply  # type: ignore[method-assign]

    await upscaler.run(job, fps_multiplier=2)

    observed_stages = {stage_key for stage_key, frames_done in stage_snapshots if frames_done > 0}
    assert observed_stages == {"extracting_frames", "upscaling_frames", "interpolating_frames"}
    assert job.metadata["progress"] == pytest.approx(1.0)
    assert job.metadata["stage"] == "completed"


# ---------------------------------------------------------------------------
# SP5 Task 2 review fix 1: a failure INSIDE _apply_frame_progress (metadata
# mutation) must be swallowed by the poller's guard, never re-raised through
# `await poller`, so it cannot mask the stage's own exception.
# ---------------------------------------------------------------------------


async def test_track_frame_progress_apply_error_does_not_mask_stage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = 4
    advance_video_stage(job, "extracting_frames")
    output_dir = tmp_path / "frames-in"

    def boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("apply mutation blew up")

    monkeypatch.setattr(upscaler, "_apply_frame_progress", boom)

    tasks_before = asyncio.all_tasks()

    with pytest.raises(RuntimeError, match="stage exploded"):
        async with upscaler._track_frame_progress(job, output_dir, "extracting_frames"):
            await asyncio.sleep(0.05)  # let the poller tick and hit boom
            raise RuntimeError("stage exploded")

    assert asyncio.all_tasks() == tasks_before


# ---------------------------------------------------------------------------
# SP5 Task 2 review fix 2: interpolating_frames uses the RIFE target count as
# denominator (source * multiplier / target_fps count), NOT the source
# framesTotal, so the stage does not clamp to 100% halfway through.
# ---------------------------------------------------------------------------


async def test_apply_frame_progress_interpolation_uses_explicit_target_denominator(tmp_path: Path) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4", fps_multiplier=2)
    job.metadata["framesTotal"] = 4  # source count
    advance_video_stage(job, "interpolating_frames")
    floor = job.metadata["progress"]
    stages = build_video_stages(job)
    interp_weight = next(stage.weight for stage in stages if stage.key == "interpolating_frames")

    # 4 produced frames against a target of 8 is HALF the stage, not full: with
    # the source framesTotal (4) as denominator this would already read 100%.
    upscaler._apply_frame_progress(job, "interpolating_frames", frames_done=4, frames_total=8)
    assert job.metadata["progress"] == pytest.approx(floor + 0.5 * interp_weight)

    upscaler._apply_frame_progress(job, "interpolating_frames", frames_done=8, frames_total=8)
    assert job.metadata["progress"] == pytest.approx(floor + interp_weight)


async def test_pipeline_passes_target_count_as_interpolation_denominator(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = LiveFrameVideoUpscaler(
        settings,
        FakeVideoEngine(),
        FakeMediaTools(),
        LiveFrameRifeEngine(),
        frame_poll_interval_seconds=0.01,
    )
    source = upscaler.settings.uploads_path / "clip.mp4"
    source.write_bytes(b"fake-video-bytes")
    job = make_video_job(source, fps_multiplier=2)

    captured: dict[str, int | None] = {}
    original_track = upscaler._track_frame_progress

    def spying_track(
        job_arg: VideoUpscaleJob, output_dir: Path, stage_key: str, frames_total: int | None = None
    ):
        captured[stage_key] = frames_total
        return original_track(job_arg, output_dir, stage_key, frames_total=frames_total)

    upscaler._track_frame_progress = spying_track  # type: ignore[method-assign]

    await upscaler.run(job, fps_multiplier=2)

    # extract is 1:1 with the source -> no explicit denominator (falls back to
    # framesTotal). interpolation gets the doubled target (4 source * 2), and
    # since the reorder the upscale consumes the INTERPOLATED frames, so its
    # honest denominator is that same doubled count.
    assert captured["extracting_frames"] is None
    assert captured["upscaling_frames"] == 8
    assert captured["interpolating_frames"] == 8
    assert job.metadata["interpFramesTotal"] == 8


# ---------------------------------------------------------------------------
# SP5 branch-review fix: framesDone is job-wide and was never reset between
# frame stages. A stage that finishes with framesDone == framesTotal left the
# NEXT stage starting with fraction=framesDone/framesTotal=1.0, freezing the
# bar at that stage's ceiling (upscale, the heaviest stage, sat at ~83% the
# whole time and frames X/Y showed "10/10"). Each frame stage must reset
# framesDone to 0 on entry so it counts its own output honestly.
# ---------------------------------------------------------------------------


def _stage_floor_and_weight(job: VideoUpscaleJob, stage_key: str) -> tuple[float, float]:
    stages = build_video_stages(job)
    floor = compute_progress(apply_stage_transition(stages, stage_key))
    weight = next(stage.weight for stage in stages if stage.key == stage_key)
    return floor, weight


async def test_frames_done_resets_on_entry_to_next_frame_stage_so_bar_does_not_freeze(
    tmp_path: Path,
) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = 10

    # Stage 1 (extract) runs to completion: framesDone reaches framesTotal.
    advance_video_stage(job, "extracting_frames")
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir()
    for index in range(10):
        (frames_in / f"{index:08d}.png").write_bytes(b"frame")
    async with upscaler._track_frame_progress(job, frames_in, "extracting_frames"):
        await asyncio.sleep(0.03)
    assert job.metadata["framesDone"] == 10
    extract_ceiling = job.metadata["progress"]

    # Stage 2 (upscale) begins. Its floor is continuous with extract's ceiling.
    advance_video_stage(job, "upscaling_frames")
    upscale_floor, upscale_weight = _stage_floor_and_weight(job, "upscaling_frames")
    assert upscale_floor == pytest.approx(extract_ceiling)

    frames_out = tmp_path / "frames-out"
    snapshots: list[tuple[int, float]] = []
    original_apply = upscaler._apply_frame_progress

    def spying_apply(
        job_arg: VideoUpscaleJob, stage_key: str, frames_done: int, frames_total: int | None = None
    ) -> None:
        original_apply(job_arg, stage_key, frames_done, frames_total)
        snapshots.append((job_arg.metadata["framesDone"], job_arg.metadata["progress"]))

    upscaler._apply_frame_progress = spying_apply  # type: ignore[method-assign]

    async with upscaler._track_frame_progress(job, frames_out, "upscaling_frames"):
        # Reset happens on entry, before any upscale frame exists.
        assert job.metadata["framesDone"] == 0
        await write_frames_incrementally(frames_out, count=10, delay=0.02)

    # The first poll tick starts near the floor with a SMALL frame count, not
    # pinned at 10/10 and the stage ceiling (the freeze this fix removes).
    first_done, first_progress = snapshots[0]
    assert first_done < 10
    assert first_progress < upscale_floor + upscale_weight

    # It then climbs honestly within the upscale band, monotonically, to the ceiling.
    progress_values = [progress for _, progress in snapshots]
    assert progress_values == sorted(progress_values)
    assert job.metadata["framesDone"] == 10
    assert job.metadata["progress"] == pytest.approx(upscale_floor + upscale_weight)


async def test_interpolation_stage_starts_at_its_floor_not_frozen_midway(tmp_path: Path) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4", fps_multiplier=2)
    job.metadata["framesTotal"] = 10

    # Upscale completes with framesDone == framesTotal before interpolation.
    advance_video_stage(job, "upscaling_frames")
    frames_out = tmp_path / "frames-out"
    frames_out.mkdir()
    for index in range(10):
        (frames_out / f"{index:08d}.png").write_bytes(b"frame")
    async with upscaler._track_frame_progress(job, frames_out, "upscaling_frames"):
        await asyncio.sleep(0.03)
    assert job.metadata["framesDone"] == 10

    advance_video_stage(job, "interpolating_frames")
    interp_floor, interp_weight = _stage_floor_and_weight(job, "interpolating_frames")
    frames_interp = tmp_path / "frames-interp"

    first_progress: list[float] = []
    async with upscaler._track_frame_progress(
        job, frames_interp, "interpolating_frames", frames_total=20
    ):
        assert job.metadata["framesDone"] == 0
        first_progress.append(job.metadata["progress"])
        await write_frames_incrementally(frames_interp, count=20, delay=0.01)

    # Interpolation begins at its floor, NOT frozen half-way through the run.
    assert first_progress[0] == pytest.approx(interp_floor)
    assert first_progress[0] < interp_floor + interp_weight
    assert job.metadata["progress"] == pytest.approx(interp_floor + interp_weight)


async def test_frames_counter_reflects_current_stage_not_previous(tmp_path: Path) -> None:
    upscaler = make_tracking_upscaler(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4")
    job.metadata["framesTotal"] = 10

    advance_video_stage(job, "extracting_frames")
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir()
    for index in range(10):
        (frames_in / f"{index:08d}.png").write_bytes(b"frame")
    async with upscaler._track_frame_progress(job, frames_in, "extracting_frames"):
        await asyncio.sleep(0.03)
    assert job.metadata["framesDone"] == 10

    advance_video_stage(job, "upscaling_frames")
    frames_out = tmp_path / "frames-out"
    async with upscaler._track_frame_progress(job, frames_out, "upscaling_frames"):
        # On entry the counter is scoped to this stage, never the stale 10.
        assert job.metadata["framesDone"] == 0
        await write_frames_incrementally(frames_out, count=3, delay=0.02)
        # Mid-stage it tracks the upscaled frames produced so far, still not 10.
        assert job.metadata["framesDone"] < 10

    # The finally-block authoritative count settles on this stage's 3 frames.
    assert job.metadata["framesDone"] == 3
