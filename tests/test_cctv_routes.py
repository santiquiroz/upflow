from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from app.api import cctv_routes
from app.api.cctv_routes import CctvApiState
from app.api.routes import video_job_to_response
from app.config import Settings
from app.models import CctvOptions, JobStatus, VideoUpscaleJob
from app.schemas_cctv import CctvJobRequest, cctv_options, cctv_summary
from app.services import cctv_session
from app.services.cctv_analysis import CctvAnalysisError
from app.services.cctv_analysis_jobs import CctvAnalysisJobs
from app.services.cctv_artifacts import ArtifactNotFound, listed_artifacts, resolve_artifact
from app.services.cctv_chain import CCTV_CHAIN, CctvChainError
from app.services.cctv_frame_index import FrameEntry, frame_index_csv
from app.services.cctv_ingest import FRAME_INDEX_NAME, WORK_COPY_NAME, ReceivedAt, SourceRecord, write_source_record
from app.services.cctv_job_runner import build_cctv_runners
from app.services.cctv_presets_view import presets_payload
from app.services.cctv_preview import FrameWindow, frame_window, preview_steps, window_filter
from app.services.device_semaphores import DeviceSemaphores
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.ffmpeg_filters import FrameGeometry
from app.services.media_signature import MATROSKA
from app.services.media_tools import MediaTools
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler
from ffmpeg_support import needs_ffmpeg

TOKEN = "session0token1"
WIDTH, HEIGHT, FRAMES = 320, 240, 50
ALL_FILTERS = frozenset(name for step in CCTV_CHAIN for spec in step.filters for name in spec.ffmpeg_filters)
CAPS = FfmpegCapabilities("ab" * 32, "ffmpeg test", ("--enable-gpl",), ALL_FILTERS, frozenset({"ffv1", "libx264"}), ())
GEOMETRY = FrameGeometry(WIDTH, HEIGHT)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def job_body(**overrides) -> dict:
    return {"token": TOKEN, "task": "clarify", "noOsd": True, **overrides}


# --- Fakes para las rutas sin ffmpeg ---


class FakeUpscaler:
    def cctv_task_available(self, task: str) -> bool:
        return task == "clarify"


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video", "width": WIDTH, "height": HEIGHT, "sample_aspect_ratio": "1:1"}]}


class NoDevices:
    def list_devices(self) -> list[dict]:
        raise AssertionError("an explicit device must not enumerate hardware")


def write_fake_session(settings: Settings) -> Path:
    directory = cctv_session.session_dir(settings.video_work_path, TOKEN)
    upload = directory / cctv_session.UPLOAD_DIRNAME / "clip.mp4"
    upload.parent.mkdir(parents=True)
    upload.write_bytes(b"clip")
    record = SourceRecord(
        "clip.mp4", 4, "2026-09-25T10:00:00.000Z", hashlib.sha256(b"clip").hexdigest(),
        ReceivedAt("2026-09-25T10:00:00.000Z", "2026-09-25T05:00:00.000-05:00"), MATROSKA,
    )  # fmt: skip
    write_source_record(directory, record)
    (directory / WORK_COPY_NAME).write_bytes(b"mkv")
    entries = [FrameEntry(n, n / 25, n % 25 == 0, "P", 100) for n in range(FRAMES)]
    (directory / FRAME_INDEX_NAME).write_text(frame_index_csv(entries), encoding="utf-8")
    return directory


def fake_manager(tmp_path: Path) -> VideoJobManager:
    settings = make_settings(tmp_path)
    write_fake_session(settings)
    return VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings), cctv_capabilities=lambda: CAPS
    )


async def post_job(manager: VideoJobManager, body: dict):
    return await cctv_routes.create_cctv_job(
        request=None,
        body=CctvJobRequest.model_validate(body),
        video_jobs=manager,
        settings=manager.settings,
        devices=NoDevices(),
    )


async def rejected(call) -> HTTPException:
    with pytest.raises(HTTPException) as caught:
        await call
    return caught.value


# --- Esquemas ---


def test_the_job_request_is_camel_case_and_maps_to_cctv_options() -> None:
    body = CctvJobRequest.model_validate(
        {
            "token": TOKEN,
            "task": "roi_fusion",
            "steps": [{"id": "denoise", "params": {"filter": "hqdn3d"}}],
            "osdBoxes": [[0, 0, 96, 24]],
            "osdBoxesConfirmed": True,
            "trim": [3, 40],
            "stillFrames": [5],
            "roi": {"firstFrame": 1, "lastFrame": 9, "referenceFrame": 4, "box": [2, 2, 16, 8], "kind": "plate"},
            "acquisition": {"recorderMake": "HiLook", "clockOffsetSeconds": 12.5},
            "caseLabel": "Caso 7",
        }
    )

    options = cctv_options(body)

    assert options.session_token == TOKEN and options.osd_boxes == ((0, 0, 96, 24),) and options.trim == (3, 40)
    assert options.steps[0].id == "denoise" and dict(options.steps[0].params) == {"filter": "hqdn3d"}
    assert options.roi.box == (2, 2, 16, 8) and options.roi.method == "median" and options.roi.scale == 2
    assert dict(options.acquisition) == {"recorderMake": "HiLook", "clockOffsetSeconds": 12.5}


@pytest.mark.parametrize(
    "bad",
    [{"filterGraph": "movie=/etc/passwd"}, {"osdBoxes": [[0, 0, True, 24]]}, {"trim": ["3", 40]}, {"noOsd": "yes"}],
)
def test_the_job_request_rejects_unknown_fields_and_loose_types(bad: dict) -> None:
    with pytest.raises(ValidationError):
        CctvJobRequest.model_validate(job_body(**bad))


# --- Jobs (sin ffmpeg) ---


async def test_a_valid_job_answers_with_its_cctv_summary(tmp_path: Path) -> None:
    manager = fake_manager(tmp_path)

    response = await post_job(manager, job_body(device="cpu"))

    assert response.status == JobStatus.queued and response.device == "cpu"
    assert response.cctv.task == "clarify" and response.cctv.no_osd and response.cctv.artifacts == []
    assert response.cctv.source_sha256 == hashlib.sha256(b"clip").hexdigest() and response.cctv.verify_url is None
    assert manager.get_job(response.job_id).cctv.session_token == TOKEN


async def test_a_validation_error_is_a_400_with_its_key(tmp_path: Path) -> None:
    error = await rejected(post_job(fake_manager(tmp_path), job_body(noOsd=False, device="cpu")))

    assert error.status_code == 400 and error.detail["key"] == "cctv.error.osdUnconfirmed"


async def test_an_expired_session_is_a_404(tmp_path: Path) -> None:
    error = await rejected(post_job(fake_manager(tmp_path), job_body(token="someothertoken", device="cpu")))

    assert error.status_code == 404 and error.detail["key"] == "cctv.error.sessionNotFound"


async def test_still_frame_errors_keep_their_key(tmp_path: Path) -> None:
    error = await rejected(post_job(fake_manager(tmp_path), job_body(trim=[10, 20], stillFrames=[5], device="cpu")))

    assert error.status_code == 400 and error.detail["key"] == "cctv.error.stillFrameOutOfRange"


async def test_ai_lane_options_on_a_classic_task_are_rejected(tmp_path: Path) -> None:
    error = await rejected(post_job(fake_manager(tmp_path), job_body(modelId="realesrgan-x2", device="cpu")))

    assert error.status_code == 400 and error.detail["key"] == "cctv.error.enhanceOnly"


def test_a_plain_video_job_has_no_cctv_summary(tmp_path: Path) -> None:
    job = VideoUpscaleJob(tmp_path / "a.mp4", "a.mp4", "x", 2, "mp4", "libx264", "medium", 18, True)

    assert video_job_to_response(job).cctv is None


def completed_cctv_job(job_dir: Path, outputs: dict) -> VideoUpscaleJob:
    job = VideoUpscaleJob(
        job_dir / "v.mp4", "clip.mp4", "cctv-clarify", 1, "mp4", "libx264", "medium", 16, True,
        cctv=CctvOptions(task="clarify", session_token=TOKEN, no_osd=True),
        metadata={"cctv": {"lane": "classic", "sourceSha256": "ab" * 32, "outputs": outputs, "warnings": ["cctv.lite"]}},
    )  # fmt: skip
    job.status = JobStatus.completed
    return job


STILL_PAIR = {"frame": 5, "original": {"file": "04_stills/original_f5.png"}, "processed": {"file": "04_stills/processed_f5.png"}}


def test_a_completed_job_lists_its_artifacts_and_the_verify_link(tmp_path: Path) -> None:
    outputs = {"analysis": "02_processed/a.mkv", "viewing": "02_processed/a.mp4", "comparison": None, "stills": [STILL_PAIR]}
    job = completed_cctv_job(tmp_path, outputs)

    summary = cctv_summary(job)

    names = [link.name for link in summary.artifacts]
    assert names[:2] == ["analysis", "viewing"] and "comparison" not in names
    assert {"report_html", "sha256sums", "still:5:original", "still:5:processed"} <= set(names)
    assert summary.artifacts[0].url == f"/api/v1/video/jobs/{job.id}/artifacts/analysis"
    assert summary.verify_url == f"/api/v1/video/jobs/{job.id}/verify" and summary.warnings == ["cctv.lite"]
    assert summary.roi is None


ROI_FACTS = {
    "kind": "plate", "scale": 2, "method": "median", "motion": "homography", "referenceFrame": 12,
    "framesTotal": 30, "framesUsed": 23, "effectiveSamples": 6, "rejectedFrames": [3], "nearCopies": False,
    "density": {"kind": "plate", "axis": "height", "storedPx": 14, "displayPx": 14}, "clippedFramesPct": 0.0,
    "notices": [{"key": "cctv.roi.densityPlate", "params": {"px": 14}}],
}  # fmt: skip


def test_a_multi_frame_still_summary_carries_the_fusion_facts(tmp_path: Path) -> None:
    job = completed_cctv_job(tmp_path, {"roi": {"fused": "04_stills/roi_fused_x2.png"}})
    job.metadata["cctv"]["roi"] = ROI_FACTS

    summary = cctv_summary(job)

    assert summary.roi == ROI_FACTS and "roi:fused" in [link.name for link in summary.artifacts]
    assert summary.model_dump(by_alias=True)["roi"]["effectiveSamples"] == 6


# --- Artefactos: lista blanca ---


def test_artifacts_resolve_only_whitelisted_names_inside_the_job_dir(tmp_path: Path) -> None:
    job_dir = tmp_path / "job.cctv"
    (job_dir / "04_stills").mkdir(parents=True)
    (job_dir / "04_stills" / "original_f5.png").write_bytes(b"png")
    (job_dir / "report.html").write_text("<html>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    outputs = {"stills": [STILL_PAIR], "viewing": "../secret.txt"}

    assert resolve_artifact(job_dir, "still:5:original", outputs).name == "original_f5.png"
    assert resolve_artifact(job_dir, "report_html", outputs).name == "report.html"
    for name in ("viewing", "still:5:processed", "../secret.txt", "report.json", "still:6:original", "roi:../x"):
        with pytest.raises(ArtifactNotFound):
            resolve_artifact(job_dir, name, outputs)


def test_a_queued_job_lists_no_artifacts(tmp_path: Path) -> None:
    job = completed_cctv_job(tmp_path, {"viewing": "x.mp4"})
    job.status = JobStatus.running

    assert listed_artifacts(job) == []


async def test_artifacts_and_verify_of_an_unfinished_job_are_409(tmp_path: Path) -> None:
    manager = fake_manager(tmp_path)
    created = await post_job(manager, job_body(device="cpu"))
    settings = manager.settings

    artifact = await rejected(cctv_routes.get_cctv_artifact(created.job_id, "viewing", None, manager, settings))
    verify = await rejected(cctv_routes.verify_cctv_job(created.job_id, None, manager, settings))

    assert artifact.status_code == verify.status_code == 409


async def test_artifacts_of_an_unknown_job_are_404(tmp_path: Path) -> None:
    manager = fake_manager(tmp_path)

    error = await rejected(cctv_routes.get_cctv_artifact("nope", "viewing", None, manager, manager.settings))

    assert error.status_code == 404


# --- Registro de analisis (200 o 202) ---


async def test_a_fast_analysis_answers_synchronously() -> None:
    async def work() -> dict:
        return {"token": "t"}

    snapshot = await CctvAnalysisJobs(sync_seconds=5).submit(work, "u1")

    assert snapshot.status == "completed" and snapshot.result == {"token": "t"}


async def test_a_slow_analysis_is_pending_and_can_be_polled() -> None:
    release = asyncio.Event()

    async def work() -> dict:
        await release.wait()
        return {"token": "t"}

    jobs = CctvAnalysisJobs(sync_seconds=0)
    pending = await jobs.submit(work, "u1")
    release.set()
    await asyncio.sleep(0.05)

    assert pending.status == "running" and jobs.get(pending.id).status == "completed"


async def test_a_failed_analysis_keeps_its_key() -> None:
    async def work() -> dict:
        raise CctvAnalysisError("cctv.error.noVideoStream", "no video")

    snapshot = await CctvAnalysisJobs(sync_seconds=5).submit(work)

    assert snapshot.status == "failed" and snapshot.error.key == "cctv.error.noVideoStream"


async def test_finished_analyses_are_pruned_oldest_first() -> None:
    async def work() -> dict:
        return {}

    jobs = CctvAnalysisJobs(sync_seconds=5, max_finished=2)
    first = await jobs.submit(work)
    for _ in range(3):
        last = await jobs.submit(work)

    assert jobs.get(first.id) is None and jobs.get(last.id) is not None


async def test_another_users_analysis_is_not_visible() -> None:
    async def work() -> dict:
        return {}

    state = CctvApiState(analyses=CctvAnalysisJobs(sync_seconds=5))
    snapshot = await state.analyses.submit(work, "owner")
    request = type("R", (), {"state": type("S", (), {"current_user": _user("intruder")})()})()

    error = await rejected(cctv_routes.get_cctv_analysis(snapshot.id, request, state))

    assert error.status_code == 404


def _user(user_id: str):
    from app.services.auth.identity import AuthenticatedUser

    return AuthenticatedUser(user_id, user_id, "user", frozenset(), False, {})


async def test_analyze_needs_exactly_one_of_file_or_upload_token(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    upload = UploadFile(file=io.BytesIO(b"x"), filename="clip.mkv")

    both = await rejected(analyze(settings, file=upload, upload_token="a" * 32))
    neither = await rejected(analyze(settings))

    assert both.detail["key"] == neither.detail["key"] == "cctv.error.uploadChoice"


async def test_an_unknown_upload_token_leaves_no_session_behind(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    error = await rejected(analyze(settings, upload_token="f" * 32))

    assert error.detail["key"] == "cctv.error.uploadNotFound"
    assert not any(settings.video_work_path.glob("cctv-*"))


# --- Vista previa (piezas puras) ---


def test_the_preview_window_is_clamped_and_bounded() -> None:
    assert frame_window(2, 4, FRAMES) == FrameWindow(0, 2, 6)
    assert frame_window(48, 4, FRAMES) == FrameWindow(44, 48, 49)
    for frame, radius in ((FRAMES, 4), (-1, 4), (10, 13)):
        with pytest.raises(CctvChainError):
            frame_window(frame, radius, FRAMES)


def test_the_window_filter_selects_by_time_then_takes_the_target_by_position() -> None:
    frames = tuple(FrameEntry(n, 1.0 + n / 25, False, "P", 1) for n in range(FRAMES))

    vf = window_filter(frames, FrameWindow(6, 10, 14), "hqdn3d=4:3:6:4.5")

    assert vf == "select=between(t\\,1.220000\\,1.580000),hqdn3d=4:3:6:4.5,select=eq(n\\,4)"


def test_preview_steps_are_classic_only_and_skip_trim_and_osd() -> None:
    raw = [{"id": "trim", "params": {"start_frame": 0, "end_frame": 9}}, {"id": "denoise", "params": {"filter": "hqdn3d"}}]

    assert [step.id for step in preview_steps([*raw, {"id": "osd_protect"}], CAPS, GEOMETRY)] == ["denoise"]
    with pytest.raises(CctvChainError) as caught:
        preview_steps([{"id": "ai_deblock", "params": {"strength": 40}}], CAPS, GEOMETRY)
    assert caught.value.code == "cctv.error.stepNotInLane"


def test_the_single_frame_preview_leaves_out_stabilization() -> None:
    raw = [{"id": "stabilize"}, {"id": "denoise", "params": {"filter": "hqdn3d"}}]

    assert [step.id for step in preview_steps(raw, CAPS, GEOMETRY)] == ["denoise"]


def test_the_presets_payload_marks_what_the_build_lacks() -> None:
    lgpl = FfmpegCapabilities("cd" * 32, "ffmpeg lgpl", (), ALL_FILTERS - {"hqdn3d"}, frozenset({"libx264"}), ())

    payload = presets_payload(lgpl)

    denoise = next(step for step in payload["steps"]["classic"] if step["id"] == "denoise")
    hqdn3d = next(spec for spec in denoise["filters"] if spec["name"] == "hqdn3d")
    assert not payload["modeAvailable"] and "ffv1" in payload["modeUnavailableReason"]
    assert denoise["available"] and not hqdn3d["available"] and hqdn3d["unavailableReasonKey"] == "cctv.filterUnavailable"
    assert {"day", "night_ir", "analog", "low_res"} <= {preset["id"] for preset in payload["presets"]}


# --- Con ffmpeg real ---


def make_clip(path: Path, settings: Settings, seconds: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(settings.ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25,noise=alls=12:allf=t",
        "-f", "lavfi", "-i", "sine=sample_rate=8000",
        "-t", str(seconds), "-threads", "1", "-c:v", "libx264", "-x264-params", "threads=1", "-bf", "2",
        "-b:v", "200k", "-g", "25", "-c:a", "pcm_alaw", "-shortest", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path


def analyze(settings: Settings, state: CctvApiState | None = None, **kwargs):
    return cctv_routes.analyze_cctv(
        request=None,
        file=kwargs.get("file"),
        upload_token=kwargs.get("upload_token"),
        frame_rate=None,
        storage=StorageService(settings),
        settings=settings,
        state=state or CctvApiState(),
    )


async def settled(jobs: CctvAnalysisJobs, analysis_id: str, attempts: int = 600) -> None:
    for _ in range(attempts):
        if jobs.get(analysis_id).status != "running":
            return
        await asyncio.sleep(0.1)
    raise AssertionError("the analysis did not finish")


def clip_upload(clip: Path) -> UploadFile:
    return UploadFile(file=io.BytesIO(clip.read_bytes()), filename="cámara 1.mkv")


def decode_exact(settings: Settings, video: Path, frame: int, vf: str = "") -> np.ndarray:
    graph = ",".join(part for part in (vf, f"select=eq(n\\,{frame})") if part)
    command = [
        str(settings.ffmpeg_binary_path), "-hide_banner", "-v", "error", "-i", str(video), "-vf", graph,
        "-frames:v", "1", "-fps_mode", "passthrough", "-f", "image2pipe", "-c:v", "png", "pipe:1",
    ]  # fmt: skip
    return png_pixels(subprocess.run(command, check=True, capture_output=True).stdout)


def png_pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


async def preview(settings: Settings, token: str, frame: int, steps: str | None = None, window: int = 4) -> bytes:
    response = await cctv_routes.preview_cctv_frame(token, frame, steps, window, settings, CctvApiState())
    assert response.media_type == "image/png"
    return response.body


@pytest.fixture
async def analyzed(tmp_path: Path):
    settings = make_settings(tmp_path)
    clip = make_clip(tmp_path / "src" / "clip.mkv", settings)
    result = await analyze(settings, file=clip_upload(clip))
    return settings, clip, result


@needs_ffmpeg
async def test_analyze_hashes_the_upload_and_returns_the_diagnosis(analyzed) -> None:
    settings, clip, result = analyzed

    body = result.model_dump(mode="json", by_alias=True)
    assert body["sourceSha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()
    assert body["container"]["kind"] == "matroska" and body["originalName"] == "cámara 1.mkv"
    assert body["frameIndex"]["frameCount"] == FRAMES and body["gop"]["keyframes"] >= 2
    assert body["video"]["width"] == WIDTH and body["audio"][0]["family"] == "g711"
    assert not body["decodeFailed"] and body["suggestedPreset"] in {"day", "analog"} and body["quality"]["frameStats"]
    assert body["proposedSteps"]["classic"][-1]["id"] == "osd_protect" and body["modeAvailable"]
    session = cctv_session.load_session(settings.video_work_path, body["token"])
    assert session.frame_count == FRAMES and session.record.sha256 == body["sourceSha256"]


@needs_ffmpeg
async def test_a_long_analysis_answers_202_and_the_poll_returns_the_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    clip = make_clip(tmp_path / "src" / "clip.mkv", settings)
    state = CctvApiState(analyses=CctvAnalysisJobs(sync_seconds=0))

    reply = await analyze(settings, state, file=clip_upload(clip))

    assert isinstance(reply, JSONResponse) and reply.status_code == 202
    pending = json.loads(reply.body)
    assert pending["status"] == "running" and pending["statusUrl"].endswith(pending["analysisJobId"])
    await settled(state.analyses, pending["analysisJobId"])
    polled = await cctv_routes.get_cctv_analysis(pending["analysisJobId"], None, state)
    assert polled.status == "completed" and polled.result.frame_index["frameCount"] == FRAMES


@needs_ffmpeg
async def test_a_staged_upload_is_moved_into_the_session_and_hashed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    upload_token = "0123456789abcdef0123456789abcdef"
    staged = make_clip(settings.uploads_path / f"{upload_token}-clip.mkv", settings)
    expected = hashlib.sha256(staged.read_bytes()).hexdigest()

    result = await analyze(settings, upload_token=upload_token)

    assert result.source_sha256 == expected and result.original_name == "clip.mkv" and not staged.exists()


@needs_ffmpeg
async def test_a_file_without_video_is_rejected_and_its_session_removed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    upload = UploadFile(file=io.BytesIO(b"\x00" * 4096), filename="empty.dav")

    error = await rejected(analyze(settings, file=upload))

    assert error.status_code == 400 and error.detail["key"].startswith("cctv.error.")
    assert not any(settings.video_work_path.glob("cctv-*"))


@needs_ffmpeg
@pytest.mark.parametrize("frame", [0, 1, 26, FRAMES - 1])
async def test_the_frame_preview_is_the_exact_decoded_frame(analyzed, frame: int) -> None:
    settings, _, result = analyzed
    work = cctv_session.session_dir(settings.video_work_path, result.token) / WORK_COPY_NAME

    shown = png_pixels(await preview(settings, result.token, frame))

    assert np.array_equal(shown, decode_exact(settings, work, frame))


@needs_ffmpeg
async def test_the_processed_preview_applies_the_steps_on_the_frame(analyzed) -> None:
    settings, _, result = analyzed
    work = cctv_session.session_dir(settings.video_work_path, result.token) / WORK_COPY_NAME
    deblock = json.dumps([{"id": "deblock", "params": {"filter": "deblock", "filter_type": "strong", "block": 8}}])
    denoise = json.dumps([{"id": "denoise", "params": {"filter": "hqdn3d"}}])

    spatial = png_pixels(await preview(settings, result.token, 30, deblock))
    temporal = png_pixels(await preview(settings, result.token, 30, denoise, window=2))

    assert np.array_equal(spatial, decode_exact(settings, work, 30, "deblock=filter=strong:block=8"))
    assert not np.array_equal(temporal, decode_exact(settings, work, 30))


@needs_ffmpeg
async def test_the_processed_preview_flattens_a_fisheye_at_the_frame_size(analyzed) -> None:
    settings, _, result = analyzed
    work = cctv_session.session_dir(settings.video_work_path, result.token) / WORK_COPY_NAME
    original = decode_exact(settings, work, 30)
    height, width = original.shape[:2]
    lens = json.dumps([{"id": "lens", "params": {"filter": "v360", "d_fov": 90.0}}])

    shown = png_pixels(await preview(settings, result.token, 30, lens))

    fisheye = "v360=input=fisheye:output=flat:ih_fov=180.0:iv_fov=180.0:d_fov=90.0:yaw=0.0:pitch=0.0"
    assert np.array_equal(shown, decode_exact(settings, work, 30, f"{fisheye}:w={width}:h={height}:interp=near"))
    assert not np.array_equal(shown, original)


@needs_ffmpeg
@pytest.mark.parametrize(
    ("frame", "steps", "window", "key"),
    [
        (FRAMES, None, 4, "cctv.error.frameOutOfRange"),
        (10, "not json", 4, "cctv.error.invalidStep"),
        (10, json.dumps([{"id": "ai_deblock", "params": {"strength": 40}}]), 4, "cctv.error.stepNotInLane"),
        (10, json.dumps([{"id": "denoise", "params": {"filter": "hqdn3d"}}]), 13, "cctv.error.invalidPreviewWindow"),
    ],
)
async def test_bad_previews_are_400_with_their_key(analyzed, frame: int, steps: str | None, window: int, key: str) -> None:
    settings, _, result = analyzed

    error = await rejected(cctv_routes.preview_cctv_frame(result.token, frame, steps, window, settings, CctvApiState()))

    assert error.status_code == 400 and error.detail["key"] == key


@needs_ffmpeg
async def test_the_osd_check_reports_each_box(analyzed) -> None:
    settings, _, result = analyzed
    body = cctv_routes.OsdCheckRequest.model_validate({"boxes": [[0, 0, 96, 24], [200, 180, 64, 32]], "frame": 5})

    response = await cctv_routes.check_cctv_osd(result.token, body, settings)

    assert [check.box for check in response.checks] == [[0, 0, 96, 24], [200, 180, 64, 32]]
    assert all(check.frames >= 1 for check in response.checks)
    for check in response.checks:
        assert check.warning_key == (None if check.looks_like_text else "cctv.osd.notText")


@needs_ffmpeg
async def test_an_osd_box_outside_the_frame_is_400(analyzed) -> None:
    settings, _, result = analyzed
    body = cctv_routes.OsdCheckRequest.model_validate({"boxes": [[300, 0, 64, 24]]})

    error = await rejected(cctv_routes.check_cctv_osd(result.token, body, settings))

    assert error.status_code == 400 and error.detail["key"] == "cctv.error.osdBoxOutsideFrame"


def reference_body(**overrides) -> cctv_routes.RoiReferenceRequest:
    raw = {"firstFrame": 10, "lastFrame": 20, "box": [100, 80, 64, 32], "steps": [], **overrides}
    return cctv_routes.RoiReferenceRequest.model_validate(raw)


@needs_ffmpeg
async def test_the_suggested_reference_frame_is_inside_the_range(analyzed) -> None:
    settings, _, result = analyzed
    state = cctv_routes.CctvApiState()
    deblock = [{"id": "deblock", "params": {"filter": "deblock"}}]

    response = await cctv_routes.suggest_roi_reference(result.token, reference_body(steps=deblock), settings, state)

    assert 10 <= response.reference_frame <= 20


@needs_ffmpeg
@pytest.mark.parametrize(
    ("overrides", "key"),
    [
        ({"lastFrame": 90}, "cctv.error.roiFrames"),
        ({"box": [300, 0, 64, 32]}, "cctv.error.roiOutsideFrame"),
        ({"steps": [{"id": "denoise", "params": {"filter": "hqdn3d"}}]}, "cctv.error.roiSteps"),
    ],
)
async def test_a_bad_reference_request_is_400_with_its_key(analyzed, overrides, key) -> None:
    settings, _, result = analyzed
    body = reference_body(**overrides)

    error = await rejected(cctv_routes.suggest_roi_reference(result.token, body, settings, cctv_routes.CctvApiState()))

    assert error.status_code == 400 and error.detail["key"] == key


def real_manager(settings: Settings) -> VideoJobManager:
    media_tools = MediaTools(settings)
    upscaler = VideoUpscaler(settings, NoNcnnEngine(), media_tools, cctv_runners=build_cctv_runners(settings))
    return VideoJobManager(settings, upscaler, media_tools, DeviceSemaphores(settings))


class NoNcnnEngine:
    def available(self) -> bool:
        return False


@needs_ffmpeg
async def test_a_job_from_the_route_serves_its_artifacts_and_detects_changes(analyzed) -> None:
    settings, _, result = analyzed
    manager = real_manager(settings)
    body = {
        "token": result.token,
        "task": "clarify",
        "steps": [{"id": "denoise", "params": {"filter": "hqdn3d", "luma_spatial": 4}}],
        "noOsd": True,
        "stillFrames": [7],
        "acquisition": {"recorderMake": "HiLook", "clockOffsetSeconds": 12.5},
        "caseLabel": "Caso 7",
        "device": "cpu",
    }

    created = await cctv_routes.create_cctv_job(
        request=None, body=CctvJobRequest.model_validate(body), video_jobs=manager, settings=settings, devices=NoDevices()
    )
    await manager._process_next()

    job = manager.get_job(created.job_id)
    assert job.status == JobStatus.completed, job.error
    names = [link.name for link in video_job_to_response(job).cctv.artifacts]
    assert {"analysis", "viewing", "comparison", "package", "still:7:original", "report_html"} <= set(names)
    report = await cctv_routes.get_cctv_artifact(job.id, "report_html", None, manager, settings)
    viewing = await cctv_routes.get_cctv_artifact(job.id, "viewing", None, manager, settings)
    assert isinstance(report, FileResponse) and report.media_type.startswith("text/html")
    assert report.headers["content-disposition"].startswith("inline") and "default-src 'none'" in report.headers["content-security-policy"]
    assert viewing.headers["content-disposition"].startswith("attachment") and Path(viewing.path).suffix == ".mp4"

    unchanged = await cctv_routes.verify_cctv_job(job.id, None, manager, settings)
    Path(viewing.path).write_bytes(b"edited")
    changed = await cctv_routes.verify_cctv_job(job.id, None, manager, settings)

    assert unchanged.ok and unchanged.checked >= 5 and not unchanged.missing
    assert not changed.ok and changed.mismatches == [Path(viewing.path).relative_to(Path(report.path).parent).as_posix()]


@needs_ffmpeg
def test_the_app_serves_the_presets_and_keyed_errors() -> None:
    from app.main import app

    with TestClient(app) as client:
        presets = client.get("/api/v1/video/cctv/presets")
        missing = client.get("/api/v1/video/cctv/abcdefgh1234/preview", params={"frame": 0})
        artifact = client.get("/api/v1/video/jobs/nope/artifacts/viewing")

    assert presets.status_code == 200 and presets.json()["modeAvailable"] is True
    assert "hqdn3d" in {spec["name"] for step in presets.json()["steps"]["classic"] for spec in step["filters"]}
    assert missing.status_code == 404 and missing.json()["detail"]["key"] == "cctv.error.sessionNotFound"
    assert artifact.status_code == 404


@needs_ffmpeg
def test_the_app_analyzes_a_clip_and_queues_a_job_with_camel_case_json(tmp_path: Path) -> None:
    from app.config import get_settings
    from app.main import app

    clip = make_clip(tmp_path / "clip.mkv", get_settings())
    with TestClient(app) as client:
        with clip.open("rb") as handle:
            analysis = client.post("/api/v1/video/cctv/analyze", files={"file": ("clip.mkv", handle, "video/x-matroska")})
        token = analysis.json()["token"]
        job = client.post("/api/v1/video/cctv/jobs", json={"token": token, "task": "clarify", "noOsd": True, "device": "cpu"})
        loose = client.post("/api/v1/video/cctv/jobs", json={"token": token, "task": "clarify", "filterGraph": "x"})
        status = client.get(f"/api/v1/video/jobs/{job.json()['jobId']}")

    assert analysis.status_code == 200 and analysis.json()["frameIndex"]["frameCount"] == FRAMES
    assert job.status_code == 202 and job.json()["cctv"]["task"] == "clarify" and job.json()["cctv"]["noOsd"] is True
    assert status.json()["cctv"]["sourceSha256"] == analysis.json()["sourceSha256"]
    assert loose.status_code == 422


# --- Carril IA: modelo, escala y alto de salida ---


async def test_a_target_height_in_the_ai_lane_is_rejected_in_favor_of_the_resize_step(tmp_path: Path) -> None:
    body = job_body(task="enhance", targetHeight=720, device="dml:0")

    error = await rejected(post_job(fake_manager(tmp_path), body))

    assert error.status_code == 400 and error.detail["key"] == "cctv.error.targetHeightUnsupported"


async def test_the_ai_upscale_model_and_scale_reach_the_job_validation(tmp_path: Path) -> None:
    steps = [{"id": "ai_upscale", "params": {}}]
    body = job_body(task="enhance", steps=steps, modelId="someone/hf-upscaler", scale=2, device="dml:0")

    error = await rejected(post_job(fake_manager(tmp_path), body))

    assert error.status_code == 400 and error.detail["key"] == "cctv.error.aiUpscaleModel"


def test_the_presets_list_only_installed_stream_upscalers_with_their_generative_label(tmp_path: Path) -> None:
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "realesrgan-x4plus-x2-uint8.onnx").write_bytes(b"x")
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), BUILTIN_ONNX_DIR=str(onnx_dir))

    models = cctv_routes.ai_upscale_models_payload(settings)

    assert [(model["id"], model["scales"], model["generativeLabel"]) for model in models] == [
        ("realesrgan-x4plus", [2], "Generative (invents texture)")
    ]


# --- Reproduce (P4-REPRODUCE) ---

CLIP_SHA = hashlib.sha256(b"clip").hexdigest()


def fake_report(tmp_path: Path, sha256: str = CLIP_SHA, **fields) -> dict:
    from test_cctv_report import report_for

    payload = report_for(tmp_path / "report").model_dump(mode="json", by_alias=True)
    payload["inputs"][0].update(sha256=sha256, verifiedCopySha256=sha256)
    return {**payload, **fields}


async def post_reproduce(manager: VideoJobManager, report: dict, token: str = TOKEN, **fields):
    body = cctv_routes.CctvReproduceRequest.model_validate({"token": token, "report": report, **fields})
    return await cctv_routes.reproduce_cctv_job(request=None, body=body, video_jobs=manager, settings=manager.settings)


async def test_reproduce_queues_a_clarify_job_from_the_report_and_warns_about_the_build(tmp_path: Path) -> None:
    manager = fake_manager(tmp_path)

    started = await post_reproduce(manager, fake_report(tmp_path))

    job = manager.get_job(started.job_id)
    assert job.cctv.task == "clarify" and job.cctv.trim == (2, 5) and job.cctv.still_frames == (3,)
    assert [step.id for step in job.cctv.steps] == ["deblock", "denoise"] and job.device == "cpu"
    assert job.metadata["reproduce"]["sourceSha256"] == CLIP_SHA
    assert started.result_url == f"/api/v1/video/jobs/{job.id}/reproduce"
    assert {"cctv.reproduce.otherBuild", "cctv.reproduce.otherCpu"} <= set(started.warnings)


async def test_reproduce_runs_under_the_caller_operator_and_records_the_report_it_reproduces(tmp_path: Path) -> None:
    from app.services.cctv_report import load_report, report_json_text

    manager = fake_manager(tmp_path)
    report = fake_report(tmp_path)

    started = await post_reproduce(manager, report, operatorName="Ana")

    options = manager.get_job(started.job_id).cctv
    written = report_json_text(load_report(json.dumps(report)))
    assert options.operator_name == "Ana" and options.case_label == "Case 7"
    assert options.reproduction_of.report_sha256 == hashlib.sha256(written.encode("utf-8")).hexdigest()


async def test_reproduce_without_an_operator_leaves_it_blank_instead_of_copying_the_report(tmp_path: Path) -> None:
    manager = fake_manager(tmp_path)

    started = await post_reproduce(manager, fake_report(tmp_path))

    assert manager.get_job(started.job_id).cctv.operator_name is None


@pytest.mark.parametrize(
    ("report", "key"),
    [
        (lambda tmp: {"schemaVersion": 1, "steps": []}, "cctv.error.reproduceReportInvalid"),
        (lambda tmp: fake_report(tmp, sha256="0" * 64), "cctv.error.reproduceSourceMismatch"),
        (lambda tmp: fake_report(tmp, mode="ai-visual", aiUsed=True), "cctv.error.reproduceNotClassic"),
    ],
)
async def test_reproduce_rejects_untrusted_reports_with_their_key(tmp_path: Path, report, key) -> None:
    manager = fake_manager(tmp_path)

    error = await rejected(post_reproduce(manager, report(tmp_path)))

    assert error.status_code == 400 and error.detail["key"] == key and manager.jobs == {}


async def test_reproduce_of_an_expired_session_is_a_404(tmp_path: Path) -> None:
    error = await rejected(post_reproduce(fake_manager(tmp_path), fake_report(tmp_path), token="someothertoken"))

    assert error.status_code == 404 and error.detail["key"] == "cctv.error.sessionNotFound"


async def test_the_reproduce_result_of_an_ordinary_job_is_a_404(tmp_path: Path) -> None:
    manager = fake_manager(tmp_path)
    created = await post_job(manager, job_body(device="cpu"))
    manager.get_job(created.job_id).status = JobStatus.completed

    error = await rejected(cctv_routes.get_reproduce_result(created.job_id, None, manager, manager.settings))

    assert error.status_code == 404 and error.detail["key"] == "cctv.error.notAReproduction"


async def test_the_reproduce_result_of_an_unfinished_job_is_a_409(tmp_path: Path) -> None:
    manager = fake_manager(tmp_path)
    started = await post_reproduce(manager, fake_report(tmp_path))

    error = await rejected(cctv_routes.get_reproduce_result(started.job_id, None, manager, manager.settings))

    assert error.status_code == 409


async def run_route_job(manager: VideoJobManager, body: dict) -> VideoUpscaleJob:
    created = await post_job(manager, body)
    await manager._process_next()
    job = manager.get_job(created.job_id)
    assert job.status == JobStatus.completed, job.error
    return job


def job_report(settings: Settings, job: VideoUpscaleJob) -> dict:
    path = cctv_session.cctv_job_dir(settings.outputs_path, job.id) / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))


@needs_ffmpeg
async def test_reproduce_repeats_a_real_job_frame_for_frame_and_file_for_file(analyzed) -> None:
    settings, clip, result = analyzed
    manager = real_manager(settings)
    steps = [{"id": "denoise", "params": {"filter": "hqdn3d", "luma_spatial": 4}}]
    first = await run_route_job(manager, {**job_body(token=result.token, steps=steps, trim=[3, 40], stillFrames=[7]),
                                          "caseLabel": "Caso 7", "operatorName": "Santiago", "device": "cpu"})  # fmt: skip
    again = await analyze(settings, file=clip_upload(clip))

    started = await post_reproduce(manager, job_report(settings, first), token=again.token)
    await manager._process_next()
    outcome = await cctv_routes.get_reproduce_result(started.job_id, None, manager, settings)

    assert started.warnings == [] and manager.get_job(started.job_id).status == JobStatus.completed
    assert outcome.identical and outcome.frames_identical, [c for c in outcome.checks if not c.match]
    kinds = [check.kind for check in outcome.checks]
    assert kinds.count("framehash") == 2 and kinds.count("output") == 2 and "file" in kinds
    first_json = cctv_session.cctv_job_dir(settings.outputs_path, first.id) / "report.json"
    case = job_report(settings, manager.get_job(started.job_id))["case"]
    assert case["reproductionOf"]["reportSha256"] == hashlib.sha256(first_json.read_bytes()).hexdigest()
    assert case["operatorName"] is None and case["caseLabel"] == "Caso 7"


@needs_ffmpeg
async def test_reproduce_of_an_edited_report_does_not_match(analyzed) -> None:
    settings, _, result = analyzed
    manager = real_manager(settings)
    first = await run_route_job(manager, {**job_body(token=result.token, stillFrames=[7]), "device": "cpu"})
    report = job_report(settings, first)
    report["stills"][0]["processed"]["framehashSha256"] = "0" * 64

    started = await post_reproduce(manager, report, token=result.token)
    await manager._process_next()
    outcome = await cctv_routes.get_reproduce_result(started.job_id, None, manager, settings)

    failed = [check.name for check in outcome.checks if not check.match]
    assert failed == ["frame 7 processed"] and not outcome.identical and outcome.frames_identical is False
