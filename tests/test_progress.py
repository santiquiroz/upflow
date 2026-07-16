from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.api.routes import job_to_response, video_job_to_response
from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.progress import (
    Stage,
    advance_image_stage,
    advance_video_stage,
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
# End-to-end: JobManager worker drives image stage transitions
# ---------------------------------------------------------------------------


class FakeImageEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        return job.source_path


async def test_job_manager_worker_advances_and_completes_image_stages(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeImageEngine(), asyncio.Semaphore(1))
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
    def _write_dummy_upscaled_frame(command: list[str]) -> None:
        frames_out_dir = Path(command[command.index("-o") + 1])
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        (frames_out_dir / "00000001.png").write_bytes(b"fake-frame-out")

    @staticmethod
    def _write_dummy_output(command: list[str]) -> None:
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output-video")


def make_progress_upscaler(tmp_path: Path, snapshots: list[tuple[str, float]]) -> ProgressTrackingVideoUpscaler:
    settings = make_settings(tmp_path)
    StorageService(settings)
    return ProgressTrackingVideoUpscaler(settings, FakeVideoEngine(), FakeMediaTools(), snapshots=snapshots)


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
