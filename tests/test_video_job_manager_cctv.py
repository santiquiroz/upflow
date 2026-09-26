from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.models import CctvOptions, CctvStep, JobStatus, RoiFusionRequest
from app.services import cctv_session
from app.services.cctv_chain import CCTV_CHAIN, CctvChainError
from app.services.cctv_frame_index import FrameEntry, frame_index_csv
from app.services.cctv_ingest import (
    FRAME_INDEX_NAME,
    WORK_COPY_NAME,
    IngestTools,
    MediaTools as IngestMediaTools,
    ReceivedAt,
    SourceRecord,
    ingest_source,
    ingest_working_copy,
    write_source_record,
)
from app.services.cctv_clarify_runner import ClarifyThreads, ClarifyTools
from app.services.cctv_ingest import VerifiedCopyMismatch
from app.services.cctv_job_runner import CctvClarifyRunner, CctvRunnerConfig, build_cctv_runners
from app.services.cctv_report import load_report
from app.services.device_semaphores import DeviceSemaphores
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.frame_export import StillFrameError
from app.services.handover_package import check_files_unchanged
from app.services.media_signature import MATROSKA
from app.services.media_tools import MediaTools
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler
from ffmpeg_support import needs_ffmpeg

TOKEN = "session0token1"
WIDTH, HEIGHT, FRAMES = 320, 240, 100
UPLOAD_BYTES = b"not really a video, the fakes never decode it"
ALL_FILTERS = frozenset(name for step in CCTV_CHAIN for spec in step.filters for name in spec.ffmpeg_filters)
CAPS = FfmpegCapabilities(
    binary_sha256="ab" * 32,
    version="ffmpeg version test",
    configuration=("--enable-gpl",),
    filters=ALL_FILTERS,
    encoders=frozenset({"ffv1", "libx264"}),
    cpu_extensions=(),
)
DENOISE = CctvStep("denoise", {"filter": "hqdn3d"})
OSD_BOX = (0, 0, 100, 20)


class FakeUpscaler:
    def __init__(self, tasks: tuple[str, ...]) -> None:
        self.tasks = tasks
        self.ran: list[str] = []

    def cctv_task_available(self, task: str) -> bool:
        return task in self.tasks

    async def run(self, job, fps_multiplier: int = 1) -> Path:
        self.ran.append(job.id)
        assert job.source_path.is_file()
        return job.source_path


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        assert source_path.name == WORK_COPY_NAME
        stream = {"codec_type": "video", "width": WIDTH, "height": HEIGHT, "pix_fmt": "yuv420p"}
        return {"streams": [{**stream, "sample_aspect_ratio": "1:1"}]}


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_record() -> SourceRecord:
    return SourceRecord(
        original_name="clip.mp4",
        size_bytes=len(UPLOAD_BYTES),
        modified_at="2026-09-25T10:00:00.000Z",
        sha256=sha256(UPLOAD_BYTES),
        received_at=ReceivedAt("2026-09-25T10:00:00.000Z", "2026-09-25T05:00:00.000-05:00"),
        container=MATROSKA,
    )


def write_fake_session(settings: Settings, token: str = TOKEN, frames: int = FRAMES) -> Path:
    directory = cctv_session.session_dir(settings.video_work_path, token)
    upload = directory / cctv_session.UPLOAD_DIRNAME / "clip.mp4"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(UPLOAD_BYTES)
    write_source_record(directory, fake_record())
    (directory / WORK_COPY_NAME).write_bytes(b"mkv")
    entries = [FrameEntry(n, n / 25, n % 25 == 0, "P", 100) for n in range(frames)]
    (directory / FRAME_INDEX_NAME).write_text(frame_index_csv(entries), encoding="utf-8")
    return directory


def make_manager(
    tmp_path: Path, tasks: tuple[str, ...] = ("clarify", "roi_fusion"), caps: FfmpegCapabilities = CAPS
) -> VideoJobManager:
    settings = make_settings(tmp_path)
    return VideoJobManager(
        settings, FakeUpscaler(tasks), FakeMediaTools(), DeviceSemaphores(settings), cctv_capabilities=lambda: caps
    )


def clarify(**overrides) -> CctvOptions:
    base = {
        "task": "clarify",
        "session_token": TOKEN,
        "steps": (DENOISE,),
        "osd_boxes": (OSD_BOX,),
        "osd_boxes_confirmed": True,
    }
    return CctvOptions(**{**base, **overrides})


def roi_request(**overrides) -> RoiFusionRequest:
    base = {"first_frame": 10, "last_frame": 20, "reference_frame": 15, "box": (40, 40, 64, 32), "kind": "plate"}
    return RoiFusionRequest(**{**base, **overrides})


def roi_task(**overrides) -> CctvOptions:
    return CctvOptions(task="roi_fusion", session_token=TOKEN, roi=overrides.pop("roi", roi_request()), **overrides)


async def rejected(manager: VideoJobManager, options: CctvOptions, device: str | None = None) -> CctvChainError:
    with pytest.raises(CctvChainError) as caught:
        await manager.create_cctv_job(cctv=options, device=device)
    return caught.value


@pytest.fixture
def manager(tmp_path: Path) -> VideoJobManager:
    built = make_manager(tmp_path)
    write_fake_session(built.settings)
    return built


# --- Validacion (400) ---


async def test_osd_neither_confirmed_nor_no_osd_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(osd_boxes_confirmed=False))

    assert error.code == "cctv.error.osdUnconfirmed"


async def test_no_osd_without_boxes_is_accepted(manager: VideoJobManager) -> None:
    job = await manager.create_cctv_job(cctv=clarify(osd_boxes=(), osd_boxes_confirmed=False, no_osd=True))

    assert job.cctv.no_osd and job.status == JobStatus.queued


async def test_an_osd_box_outside_the_frame_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(osd_boxes=((WIDTH - 50, 0, 100, 20),)))

    assert error.code == "cctv.error.osdBoxOutsideFrame"


async def test_a_crop_outside_the_frame_is_rejected(manager: VideoJobManager) -> None:
    crop = CctvStep("crop", {"x": 300, "y": 0, "w": 100, "h": 100})

    error = await rejected(manager, clarify(steps=(crop,)))

    assert error.code == "cctv.error.cropOutsideFrame"


async def test_a_trim_past_the_frame_index_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(trim=(10, FRAMES)))

    assert error.code == "cctv.error.trimOutOfRange"


async def test_the_last_indexed_frame_is_a_valid_trim_end(manager: VideoJobManager) -> None:
    job = await manager.create_cctv_job(cctv=clarify(trim=(10, FRAMES - 1)))

    assert job.cctv.trim == (10, FRAMES - 1)


async def test_an_ai_step_in_the_classic_lane_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(steps=(CctvStep("ai_deblock", {"strength": 40}),)))

    assert error.code == "cctv.error.stepNotInLane"


async def test_interpolation_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(steps=(CctvStep("interpolate", {"filter": "rife"}),)))

    assert error.code == "cctv.error.stepDeferred"


async def test_the_ai_lane_on_the_cpu_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(task="enhance", steps=()), device="cpu")

    assert error.code == "cctv.ai.cpuBlocked"


async def test_the_ai_lane_without_the_restore_pack_is_rejected(manager: VideoJobManager) -> None:
    ai_deblock = CctvStep("ai_deblock", {"strength": 40})

    error = await rejected(manager, clarify(task="enhance", steps=(ai_deblock,)), device="dml:0")

    assert error.code == "cctv.error.aiPackMissing" and "restauración" in str(error)


async def test_a_task_without_a_runner_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(task="enhance", steps=()), device="dml:0")

    assert error.code == "cctv.error.taskUnavailable"


async def test_a_step_missing_from_the_ffmpeg_build_is_rejected(tmp_path: Path) -> None:
    lgpl = FfmpegCapabilities("cd" * 32, "ffmpeg lgpl", (), ALL_FILTERS - {"hqdn3d"}, CAPS.encoders, ())
    built = make_manager(tmp_path, caps=lgpl)
    write_fake_session(built.settings)

    error = await rejected(built, clarify())

    assert error.code == "cctv.filterUnavailable"


async def test_a_build_without_ffv1_cannot_run_cctv(tmp_path: Path) -> None:
    no_ffv1 = FfmpegCapabilities("cd" * 32, "ffmpeg", (), ALL_FILTERS, frozenset({"libx264"}), ())
    built = make_manager(tmp_path, caps=no_ffv1)
    write_fake_session(built.settings)

    error = await rejected(built, clarify())

    assert error.code == "cctv.error.modeUnavailable"


@pytest.mark.parametrize(("task", "preset"), [("sharpen_all", None), ("clarify", "sunset")])
async def test_unknown_task_or_preset_is_rejected(manager: VideoJobManager, task: str, preset: str | None) -> None:
    error = await rejected(manager, clarify(task=task, preset=preset))

    assert error.code in {"cctv.error.unknownTask", "cctv.error.unknownPreset"}


async def test_still_frames_outside_the_trim_are_rejected(manager: VideoJobManager) -> None:
    with pytest.raises(StillFrameError, match="outside"):
        await manager.create_cctv_job(cctv=clarify(trim=(10, 20), still_frames=(5,)))


async def test_more_still_frames_than_the_setting_are_rejected(tmp_path: Path) -> None:
    built = make_manager(tmp_path)
    built.settings.cctv_max_still_frames = 2
    write_fake_session(built.settings)

    with pytest.raises(StillFrameError, match="At most 2"):
        await built.create_cctv_job(cctv=clarify(still_frames=(1, 2, 3)))


async def test_malformed_case_details_are_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(acquisition={"clockOffsetSeconds": "about ten"}))

    assert error.code == "cctv.error.invalidCaseDetails"


@pytest.mark.parametrize(
    ("roi", "code"),
    [
        (roi_request(box=(300, 200, 64, 64)), "cctv.error.roiOutsideFrame"),
        (roi_request(box=(40, 40, 63, 32)), "cctv.error.roiOdd"),
        (roi_request(last_frame=FRAMES), "cctv.error.roiFrames"),
        (roi_request(reference_frame=5), "cctv.error.roiFrames"),
        (roi_request(first_frame=0, last_frame=60, reference_frame=5), "cctv.error.roiFrames"),
        (roi_request(scale=5), "cctv.error.roiInvalid"),
    ],
)
async def test_a_bad_region_of_interest_is_rejected(manager: VideoJobManager, roi: RoiFusionRequest, code: str) -> None:
    error = await rejected(manager, roi_task(roi=roi))

    assert error.code == code


async def test_the_multi_frame_still_needs_a_region(manager: VideoJobManager) -> None:
    error = await rejected(manager, CctvOptions(task="roi_fusion", session_token=TOKEN))

    assert error.code == "cctv.error.roiRequired"


async def test_a_region_on_a_video_task_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(roi=roi_request()))

    assert error.code == "cctv.error.roiUnexpected"


async def test_a_valid_region_is_accepted_without_osd(manager: VideoJobManager) -> None:
    job = await manager.create_cctv_job(cctv=roi_task())

    assert job.cctv.task == "roi_fusion" and job.device == "cpu"


@pytest.mark.parametrize("token", ["../escape", "short", "x" * 65])
async def test_a_bad_token_is_rejected(manager: VideoJobManager, token: str) -> None:
    error = await rejected(manager, clarify(session_token=token))

    assert error.code == "cctv.error.invalidSessionToken"


async def test_an_expired_session_is_rejected(manager: VideoJobManager) -> None:
    error = await rejected(manager, clarify(session_token="someothertoken"))

    assert error.code == "cctv.error.sessionNotFound"


async def test_a_session_without_its_frame_index_is_rejected(manager: VideoJobManager) -> None:
    (cctv_session.session_dir(manager.settings.video_work_path, TOKEN) / FRAME_INDEX_NAME).unlink()

    error = await rejected(manager, clarify())

    assert error.code == "cctv.error.sessionNotAnalyzed"


async def test_a_rejected_job_leaves_nothing_behind(manager: VideoJobManager) -> None:
    await rejected(manager, clarify(osd_boxes_confirmed=False))

    assert not manager.jobs and not list(manager.settings.outputs_path.glob("*.cctv"))


# --- Job aceptado ---


@pytest.mark.parametrize("requested", [None, "auto", "dml:0"])
async def test_clarify_always_runs_on_the_cpu(manager: VideoJobManager, requested: str | None) -> None:
    job = await manager.create_cctv_job(cctv=clarify(), device=requested)

    assert job.device == "cpu"


async def test_the_verified_copy_is_made_when_the_job_is_created(manager: VideoJobManager) -> None:
    job = await manager.create_cctv_job(cctv=clarify())

    job_dir = cctv_session.cctv_job_dir(manager.settings.outputs_path, job.id)
    copy = job_dir / "01_original" / "clip.mp4"
    assert sha256(copy.read_bytes()) == sha256(UPLOAD_BYTES)
    assert json.loads((job_dir / "source.json").read_text(encoding="utf-8"))["sourceSha256"] == sha256(UPLOAD_BYTES)
    assert job.metadata["cctv"] == {"task": "clarify", "lane": "classic", "sourceSha256": sha256(UPLOAD_BYTES)}


async def test_a_session_file_changed_after_analysis_is_rejected(manager: VideoJobManager) -> None:
    upload = cctv_session.session_upload(cctv_session.session_dir(manager.settings.video_work_path, TOKEN))
    upload.write_bytes(b"tampered")

    error = await rejected(manager, clarify())

    assert error.code == "cctv.error.sessionChanged"
    assert not list(manager.settings.outputs_path.glob("*.cctv"))


async def test_creating_a_job_marks_the_session_as_active(manager: VideoJobManager) -> None:
    directory = cctv_session.session_dir(manager.settings.video_work_path, TOKEN)
    old = time.time() - 3 * 3600
    os.utime(directory, (old, old))

    await manager.create_cctv_job(cctv=clarify())

    assert directory.stat().st_mtime > old + 3600


async def test_the_upload_survives_two_jobs_on_the_same_token(manager: VideoJobManager) -> None:
    first = await manager.create_cctv_job(cctv=clarify())
    second = await manager.create_cctv_job(cctv=clarify(osd_boxes=(), osd_boxes_confirmed=False, no_osd=True))

    await manager._process_next()
    await manager._process_next()

    upload = cctv_session.session_upload(cctv_session.session_dir(manager.settings.video_work_path, TOKEN))
    assert first.status == second.status == JobStatus.completed
    assert upload.read_bytes() == UPLOAD_BYTES
    assert manager.upscaler.ran == [first.id, second.id]


async def test_a_cancelled_cctv_job_keeps_the_session_upload(manager: VideoJobManager) -> None:
    job = await manager.create_cctv_job(cctv=clarify())
    manager.cancel_job(job.id)

    await manager._process_next()

    assert job.source_path.is_file()


async def test_a_verified_copy_changed_before_the_run_stops_the_job(manager: VideoJobManager, tmp_path: Path) -> None:
    job = await manager.create_cctv_job(cctv=clarify())
    job_dir = cctv_session.cctv_job_dir(manager.settings.outputs_path, job.id)
    (job_dir / "01_original" / "clip.mp4").write_bytes(b"edited after the job was queued")
    unused = Path("ffmpeg-not-called.exe")
    config = CctvRunnerConfig(
        outputs_root=manager.settings.outputs_path,
        media=IngestMediaTools(unused, unused),
        clarify=ClarifyTools(unused, unused),
        threads=ClarifyThreads(1, 4, 1),
        app_version="0.0.0",
        capabilities=lambda: CAPS,
    )

    with pytest.raises(VerifiedCopyMismatch):
        await CctvClarifyRunner(config).run(job, tmp_path / "work", lambda stage, fraction: None)


# --- Modo headless (run_cctv_inline) ---


class FailingUpscaler(FakeUpscaler):
    async def run(self, job, fps_multiplier: int = 1) -> Path:
        raise RuntimeError("ffmpeg died")


async def test_an_inline_job_runs_without_the_queue_and_keeps_its_outputs(manager: VideoJobManager) -> None:
    job = await manager.run_cctv_inline(cctv=clarify())

    assert job.status == JobStatus.completed and job.device == "cpu"
    assert manager.upscaler.ran == [job.id]
    assert manager.queue_depth() == 0 and job.id not in manager.jobs
    assert (cctv_session.cctv_job_dir(manager.settings.outputs_path, job.id) / "01_original" / "clip.mp4").is_file()


async def test_an_inline_job_is_validated_like_a_queued_one(manager: VideoJobManager) -> None:
    with pytest.raises(CctvChainError) as caught:
        await manager.run_cctv_inline(cctv=clarify(osd_boxes_confirmed=False))

    assert caught.value.code == "cctv.error.osdUnconfirmed"
    assert manager.upscaler.ran == []


async def test_a_failed_inline_job_leaves_no_output_directory(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    write_fake_session(settings)
    manager = VideoJobManager(
        settings, FailingUpscaler(("clarify",)), FakeMediaTools(), DeviceSemaphores(settings), cctv_capabilities=lambda: CAPS
    )

    with pytest.raises(RuntimeError, match="ffmpeg died"):
        await manager.run_cctv_inline(cctv=clarify())

    assert not list(settings.outputs_path.glob("*" + cctv_session.JOB_DIR_SUFFIX))


# --- De punta a punta con ffmpeg real ---


def make_clip(path: Path, settings: Settings) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(settings.ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25,noise=alls=12:allf=t",
        "-f", "lavfi", "-i", "sine=sample_rate=8000",
        "-t", "2", "-threads", "1", "-c:v", "libx264", "-x264-params", "threads=1", "-b:v", "200k", "-g", "25",
        "-c:a", "pcm_alaw", "-shortest", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path


async def analyzed_session(settings: Settings, token: str) -> Path:
    directory = cctv_session.session_dir(settings.video_work_path, token)
    upload = make_clip(directory / cctv_session.UPLOAD_DIRNAME / "cámara 1 & (50%).mkv", settings)
    record = ingest_source(upload, upload.name, IngestTools())
    write_source_record(directory, record)
    tools = IngestMediaTools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path)
    await ingest_working_copy(tools, upload, directory, record.container)
    return directory


class NoNcnn:
    def available(self) -> bool:
        return False


def real_manager(settings: Settings) -> VideoJobManager:
    media_tools = MediaTools(settings)
    upscaler = VideoUpscaler(settings, NoNcnn(), media_tools, cctv_runners=build_cctv_runners(settings))
    return VideoJobManager(settings, upscaler, media_tools, DeviceSemaphores(settings))


def assert_finished_cleanly(settings: Settings, job) -> Path:
    assert job.status == JobStatus.completed, job.error
    job_dir = cctv_session.cctv_job_dir(settings.outputs_path, job.id)
    assert job.output_path.parent == job_dir / "02_processed" and job.output_path.suffix == ".mp4"
    assert not (settings.video_work_path / job.id).exists()
    assert check_files_unchanged(job_dir).ok
    assert (job_dir / job.metadata["cctv"]["outputs"]["package"]).is_file()
    assert job.metadata["stage"] == "completed" and job.metadata["progress"] == 1.0
    return job_dir


@needs_ffmpeg
async def test_two_real_jobs_on_the_same_token_each_verify_the_original_and_build_a_package(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    session = await analyzed_session(settings, TOKEN)
    manager = real_manager(settings)
    osd = CctvOptions(
        task="clarify",
        session_token=TOKEN,
        steps=(DENOISE,),
        osd_boxes=((0, 0, 96, 24),),
        osd_boxes_confirmed=True,
        trim=(3, 40),
        still_frames=(5, 30),
        case_label="Caso 7",
        acquisition={"recorderMake": "HiLook", "clockOffsetSeconds": "12.5"},
    )
    first = await manager.create_cctv_job(cctv=osd)
    second = await manager.create_cctv_job(cctv=CctvOptions(task="clarify", session_token=TOKEN, no_osd=True))

    await manager._process_next()
    await manager._process_next()

    upload = cctv_session.session_upload(session)
    assert upload.is_file()
    source_sha = cctv_session.session_record(session).sha256
    for job in (first, second):
        job_dir = assert_finished_cleanly(settings, job)
        original = job_dir / "01_original" / upload.name
        assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    report = load_report((cctv_session.cctv_job_dir(settings.outputs_path, first.id) / "report.json").read_text("utf-8"))
    assert report.inputs[0].sha256 == source_sha and report.acquisition.clock_offset_seconds == 12.5
    assert [step.id for step in report.steps] == ["trim", "denoise", "osd_protect"]
    assert report.trim is not None and len(report.stills) == 2 and report.osd.osd_boxes_confirmed
    assert first.metadata["cctv"]["framesOut"] == 38
    second_report = cctv_session.cctv_job_dir(settings.outputs_path, second.id) / "report.json"
    assert load_report(second_report.read_text("utf-8")).osd.no_osd
