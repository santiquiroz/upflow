from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.config import AUDIO_ENHANCE_MODES, AUDIO_RESTORE_MODES, GMFSS_ENGINE, INTERP_ENGINES, RIFE_ENGINE, Settings
from app.exceptions import QueueFullError
from app.models import JobStatus, TERMINAL_JOB_STATUSES, VideoUpscaleJob, utc_now
from app.services.backend_registry import validate_backend_choice
from app.services.video_encoders import VIDEO_ENCODERS
from app.services.device_router import DeviceRouter, has_compatible_device
from app.services.restorer_registry import validate_restore_mode_ready
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.media_tools import MediaTools, parse_fps_fraction, resolve_video_fps
from app.services.model_registry import ModelKind, ModelRegistry, ModelStatus
from app.services.video_upscaler import VideoUpscaler

logger = logging.getLogger(__name__)

MAX_TARGET_FPS = 240


@dataclass(frozen=True, slots=True)
class VideoModelResolution:
    model_id: str
    engine_model_name: str
    kind: ModelKind
    scale: int


class VideoJobManager:
    def __init__(
        self,
        settings: Settings,
        upscaler: VideoUpscaler,
        media_tools: MediaTools,
        device_semaphores: DeviceSemaphores,
        *,
        registry: ModelRegistry | None = None,
        devices: DevicesService | None = None,
        device_router: DeviceRouter | None = None,
    ) -> None:
        self.settings = settings
        self.upscaler = upscaler
        self.media_tools = media_tools
        self.registry = registry
        self.devices = devices
        self.jobs: dict[str, VideoUpscaleJob] = {}
        self.queue: asyncio.Queue[VideoUpscaleJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.device_semaphores = device_semaphores
        self.device_router = device_router or DeviceRouter(device_semaphores)
        self.worker_tasks: list[asyncio.Task] = []
        self._active: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        if self.worker_tasks:
            return
        self.worker_tasks = [
            asyncio.create_task(self._worker(), name=f"video-upscale-worker-{i}")
            for i in range(self.settings.max_concurrent_jobs)
        ]

    async def stop(self) -> None:
        for task in self.worker_tasks:
            task.cancel()
        for task in self.worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.worker_tasks = []

    def queue_depth(self) -> int:
        return self.queue.qsize()

    async def create_job(
        self,
        *,
        source_path: Path | None = None,
        upload_token: str | None = None,
        original_filename: str,
        model_name: str,
        scale: int,
        output_container: str,
        video_codec: str,
        video_preset: str,
        crf: int,
        keep_audio: bool,
        fps_multiplier: int = 1,
        target_fps: str | None = None,
        audio_enhance: str | None = None,
        audio_restore: str | None = None,
        audio_track_indices: list[int] | None = None,
        keep_subtitles: bool = False,
        audio_output_format: str = "auto",
        interp_engine: str = RIFE_ENGINE,
        model_id: str | None = None,
        device: str | None = None,
        backend: str | None = None,
        video_encoder: str = "auto",
        job_id: str | None = None,
    ) -> VideoUpscaleJob:
        resolved_source_path = self._resolve_source_path(source_path, upload_token)
        source_fps, probe = await self._validate_video(resolved_source_path)
        validate_backend_choice(backend)
        self._validate_video_encoder(video_encoder)
        resolved_model_id = model_id if model_id is not None else model_name
        if device is not None and device != AUTO_DEVICE_ID and self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)
        resolution = self._resolve_model(resolved_model_id, scale, device)
        if device == AUTO_DEVICE_ID:
            await self._validate_auto_device(resolution.kind)
        self._validate_request(
            output_container,
            video_codec,
            video_preset,
            crf,
            fps_multiplier,
            target_fps,
            source_fps,
            keep_audio,
            audio_enhance,
            interp_engine,
        )
        self._validate_audio_restore_mode(audio_restore, keep_audio)
        resolved_container, container_upgrade_reason = self._resolve_output_container(
            output_container, keep_subtitles, audio_restore, audio_output_format
        )

        job = VideoUpscaleJob(
            source_path=resolved_source_path,
            original_filename=original_filename,
            model_name=resolution.engine_model_name,
            scale=resolution.scale,
            output_container=resolved_container,
            video_codec=video_codec,
            video_preset=video_preset,
            crf=crf,
            keep_audio=keep_audio,
            fps_multiplier=fps_multiplier,
            target_fps=target_fps,
            audio_enhance=audio_enhance,
            audio_restore=audio_restore,
            audio_track_indices=audio_track_indices,
            keep_subtitles=keep_subtitles,
            audio_output_format=audio_output_format,
            interp_engine=interp_engine,
            model_id=resolution.model_id,
            device=device,
            backend=backend,
            video_encoder=video_encoder,
            probe=probe,
        )
        if container_upgrade_reason is not None:
            job.metadata["containerUpgradedReason"] = container_upgrade_reason
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> VideoUpscaleJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status in TERMINAL_JOB_STATUSES:
            return False
        if job.status == JobStatus.queued:
            # Still in the queue: mark it so the worker skips it on dequeue.
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
            return True
        task = self._active.get(job_id)
        if task is not None:
            task.cancel()
        return True

    def _enqueue(self, job: VideoUpscaleJob) -> None:
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError("Video job queue is full; try again later") from exc

    def _resolve_source_path(self, source_path: Path | None, upload_token: str | None) -> Path:
        if upload_token is not None:
            matches = sorted(self.settings.uploads_path.glob(f"{upload_token}-*"))
            if not matches:
                raise ValueError(f"No staged upload found for upload_token={upload_token!r}")
            return matches[0]
        if source_path is None:
            raise ValueError("Either source_path or upload_token must be provided")
        return source_path

    @staticmethod
    def _resolve_output_container(
        output_container: str, keep_subtitles: bool, audio_restore: str | None, audio_output_format: str
    ) -> tuple[str, str | None]:
        wants_flac = audio_output_format == "flac" or (
            audio_output_format == "auto" and audio_restore is not None
        )
        reasons = []
        if keep_subtitles and output_container != "mkv":
            reasons.append("preserve subtitles without quality loss")
        if wants_flac and output_container != "mkv":
            reasons.append("keep restored audio lossless (FLAC)")
        if not reasons:
            return output_container, None
        return "mkv", f"Output container upgraded to mkv to {' and '.join(reasons)}"

    async def _validate_video(self, source_path: Path) -> tuple[Fraction, dict]:
        """Returns (source_fps, probe). The probe travels with the job so the
        pipeline doesn't ffprobe the same file a second time."""
        try:
            probe = await self.media_tools.ffprobe_json(source_path)
        except subprocess.CalledProcessError as exc:
            raise ValueError("Uploaded file is not a valid video") from exc
        streams = probe.get("streams", [])
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        if video_stream is None:
            raise ValueError("Uploaded file is not a valid video")
        fps = resolve_video_fps(video_stream.get("avg_frame_rate"), video_stream.get("r_frame_rate"))
        return fps, probe

    def _resolve_model(self, model_id: str, scale: int, device: str | None) -> VideoModelResolution:
        if model_id in self.settings.model_keys:
            return self._resolve_builtin_model(model_id, scale, device)
        return self._resolve_onnx_model(model_id)

    def _resolve_builtin_model(self, model_id: str, scale: int, device: str | None) -> VideoModelResolution:
        option = self.settings.get_model_option(model_id)
        if option and scale not in option["scales"]:
            raise ValueError(f"Model {model_id} supports only scales {option['scales']}")
        if device == "cpu":
            raise ValueError(
                f"Device 'cpu' is not supported for builtin model {model_id!r} (requires a Vulkan GPU device)"
            )
        engine_model_name = self.settings.resolve_engine_model_name(model_id, scale)
        return VideoModelResolution(
            model_id=model_id, engine_model_name=engine_model_name, kind=ModelKind.builtin_ncnn, scale=scale
        )

    def _resolve_onnx_model(self, model_id: str) -> VideoModelResolution:
        if self.registry is None:
            raise ValueError(f"Model must be one of {sorted(self.settings.model_keys)}")
        entry = self.registry.get(model_id)
        if entry is None or entry.kind != ModelKind.onnx:
            raise ValueError(f"Unknown model id: {model_id!r}")
        if entry.status != ModelStatus.installed:
            raise ValueError(f"Model {model_id!r} is not ready for inference (status={entry.status.value})")
        # See JobManager._resolve_onnx_model: the model's own registered
        # scale must win over the requested one, or derived metadata
        # (outputWidth/outputHeight below, computed from job.scale) goes
        # wrong for a mismatched request/model scale.
        return VideoModelResolution(
            model_id=model_id, engine_model_name=model_id, kind=ModelKind.onnx, scale=entry.scale
        )

    async def _validate_auto_device(self, kind: ModelKind) -> None:
        if self.devices is None:
            raise ValueError("Device 'auto' requires a devices service to be configured")
        devices = await asyncio.to_thread(self.devices.list_devices)
        if not has_compatible_device(devices, kind):
            raise ValueError(
                f"No compatible device available for model kind {kind.value!r} (requested device='auto')"
            )

    def _model_kind_for_job(self, job: VideoUpscaleJob) -> ModelKind:
        if job.model_id in self.settings.model_keys:
            return ModelKind.builtin_ncnn
        if self.registry is not None:
            entry = self.registry.get(job.model_id) if job.model_id is not None else None
            if entry is not None:
                return entry.kind
        raise ValueError(f"Cannot resolve model kind for job (model_id={job.model_id!r})")

    @staticmethod
    def _validate_video_encoder(video_encoder: str) -> None:
        if video_encoder not in VIDEO_ENCODERS:
            raise ValueError(f"video_encoder must be one of {sorted(VIDEO_ENCODERS)}")

    def _validate_request(
        self,
        output_container: str,
        video_codec: str,
        video_preset: str,
        crf: int,
        fps_multiplier: int,
        target_fps: str | None,
        source_fps: Fraction,
        keep_audio: bool,
        audio_enhance: str | None,
        interp_engine: str = RIFE_ENGINE,
    ) -> None:
        if output_container not in {"mp4", "mkv"}:
            raise ValueError("Output container must be mp4 or mkv")
        if video_codec not in {"libx264", "libx265"}:
            raise ValueError("Video codec must be libx264 or libx265")
        if video_preset not in {"medium", "slow", "veryslow"}:
            raise ValueError("Video preset must be medium, slow, or veryslow")
        if crf < 10 or crf > 28:
            raise ValueError("CRF must be between 10 and 28")
        self._validate_interp_engine_choice(interp_engine)
        self._validate_fps_mode(fps_multiplier, target_fps, source_fps, interp_engine)
        self._validate_audio_enhance_mode(audio_enhance, keep_audio)

    @staticmethod
    def _validate_interp_engine_choice(interp_engine: str) -> None:
        if interp_engine not in INTERP_ENGINES:
            raise ValueError(f"interp_engine must be one of {sorted(INTERP_ENGINES)}")

    def _validate_fps_mode(
        self,
        fps_multiplier: int,
        target_fps: str | None,
        source_fps: Fraction,
        interp_engine: str = RIFE_ENGINE,
    ) -> None:
        if target_fps is not None and fps_multiplier > 1:
            raise ValueError("target_fps and fps_multiplier are mutually exclusive; provide only one")
        self._validate_fps_multiplier(fps_multiplier, interp_engine)
        if target_fps is not None:
            self._validate_target_fps(target_fps, source_fps, interp_engine)

    def _validate_target_fps(
        self, target_fps: str, source_fps: Fraction, interp_engine: str = RIFE_ENGINE
    ) -> None:
        target_fraction = parse_fps_fraction(target_fps)
        if target_fraction is None:
            raise ValueError(
                f"target_fps must be a positive fraction (e.g. '60' or '60000/1001'), got {target_fps!r}"
            )
        if target_fraction > MAX_TARGET_FPS:
            raise ValueError(f"target_fps must not exceed {MAX_TARGET_FPS}")
        if target_fraction <= source_fps:
            raise ValueError(
                f"target_fps ({target_fraction}) must be greater than the source video fps ({source_fps})"
            )
        self._validate_interpolation_enabled(interp_engine)

    def _validate_fps_multiplier(self, fps_multiplier: int, interp_engine: str = RIFE_ENGINE) -> None:
        if fps_multiplier <= 0:
            raise ValueError("fps_multiplier must be a positive integer")
        allowed_multipliers = {1, *self.settings.allowed_fps_multiplier_values}
        if fps_multiplier not in allowed_multipliers:
            raise ValueError(
                f"fps_multiplier must be 1 (off) or one of {sorted(allowed_multipliers - {1})}"
            )
        if fps_multiplier > 1:
            self._validate_interpolation_enabled(interp_engine)

    def _validate_interpolation_enabled(self, interp_engine: str = RIFE_ENGINE) -> None:
        if interp_engine == GMFSS_ENGINE:
            self._validate_gmfss_ready()
            return
        if not self.settings.enable_interpolation:
            raise ValueError(
                "Frame interpolation is disabled by configuration (set ENABLE_INTERPOLATION=true)"
            )
        if not self.settings.interpolation_available():
            raise ValueError(
                "Frame interpolation requested but RIFE is not installed "
                "(run scripts/download-rife.ps1)"
            )

    def _validate_gmfss_ready(self) -> None:
        if not self.settings.enable_gmfss:
            raise ValueError(
                "GMFSS interpolation is disabled by configuration (set ENABLE_GMFSS=true)"
            )
        if not self.settings.gmfss_available():
            raise ValueError(
                "GMFSS interpolation requested but the models are not installed "
                "(run scripts/download-gmfss-onnx.ps1)"
            )

    def _validate_audio_enhance_mode(self, audio_enhance: str | None, keep_audio: bool) -> None:
        if audio_enhance is None:
            return
        if audio_enhance not in AUDIO_ENHANCE_MODES:
            raise ValueError(f"audio_enhance must be one of {sorted(AUDIO_ENHANCE_MODES)}")
        if not keep_audio:
            raise ValueError("audio_enhance requires keep_audio to be enabled")
        self._validate_audio_enhance_enabled(audio_enhance)

    def _validate_audio_enhance_enabled(self, audio_enhance: str) -> None:
        if not self.settings.enable_audio_enhance:
            raise ValueError(
                "Audio enhancement is disabled by configuration (set ENABLE_AUDIO_ENHANCE=true)"
            )
        if not self.settings.audio_enhance_available(audio_enhance):
            raise ValueError(
                f"Audio enhance mode {audio_enhance!r} requested but not installed "
                "(run scripts/download-deepfilternet.ps1)"
            )

    def _validate_audio_restore_mode(self, audio_restore: str | None, keep_audio: bool) -> None:
        if audio_restore is None:
            return
        if audio_restore not in AUDIO_RESTORE_MODES:
            raise ValueError(f"audio_restore must be one of {sorted(AUDIO_RESTORE_MODES)}")
        if not keep_audio:
            raise ValueError("audio_restore requires keep_audio to be enabled")
        validate_restore_mode_ready(self.settings, audio_restore)

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            if job.status == JobStatus.cancelled:
                # Cancelled while waiting in the queue: skip without processing.
                self._unlink_source_if_unused(job)
                self.queue.task_done()
                continue
            if job.device == AUTO_DEVICE_ID:
                await self._run_auto_job(job)
            else:
                await self._run_pinned_job(job)

    async def _run_pinned_job(self, job: VideoUpscaleJob) -> None:
        async with self.device_semaphores.acquire(job.device):
            await self._execute_job(job)

    async def _run_auto_job(self, job: VideoUpscaleJob) -> None:
        # See JobManager._run_auto_job: device resolution is guarded on its
        # own, before any semaphore/router acquire, so a failure here fails
        # the job cleanly instead of leaving it stuck at status=queued with
        # task_done() never called.
        try:
            kind = self._model_kind_for_job(job)
            devices = await asyncio.to_thread(self.devices.list_devices)
        except Exception as exc:  # noqa: BLE001
            self._fail_dequeued_job(job, str(exc))
            return
        try:
            async with self.device_router.acquire_auto(devices, kind) as device_id:
                job.device = device_id
                await self._execute_job(job)
        except ValueError as exc:
            self._fail_dequeued_job(job, str(exc))

    async def _execute_job(self, job: VideoUpscaleJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        run_task = asyncio.ensure_future(self._run_engine(job))
        self._active[job.id] = run_task
        try:
            await run_task
            job.status = JobStatus.completed
        except asyncio.CancelledError:
            run_task.cancel()
            if asyncio.current_task().cancelling() > 0:
                # The WORKER task itself was cancelled (shutdown via stop()):
                # fail the job and re-raise so the worker actually dies.
                job.status = JobStatus.failed
                job.error = "Job cancelled"
                raise
            # Only the child engine task was cancelled (per-job cancel_job):
            # mark cancelled and let the worker live on for other jobs.
            job.status = JobStatus.cancelled
            job.error = None
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            self._active.pop(job.id, None)
            job.finished_at = utc_now()
            self._unlink_source_if_unused(job)
            self.queue.task_done()

    async def _run_engine(self, job: VideoUpscaleJob) -> None:
        job.output_path = await self.upscaler.run(job, fps_multiplier=job.fps_multiplier)

    def _fail_dequeued_job(self, job: VideoUpscaleJob, error: str) -> None:
        job.status = JobStatus.failed
        job.error = error
        job.finished_at = utc_now()
        self._unlink_source_if_unused(job)
        self.queue.task_done()

    def _unlink_source_if_unused(self, job: VideoUpscaleJob) -> None:
        # A job created from upload_token can share its source_path with sibling
        # jobs (see create_job/_resolve_source_path: each gets a fresh job_id on
        # purpose, so multiple jobs may reference the same staged upload). Only
        # unlink once no OTHER live job still references this exact path, or a
        # sibling still queued/running loses its file out from under it. Mirrors
        # RetentionSweeper._is_finished's definition of "still needed" so this
        # doesn't invent a second, divergent notion of "active".
        if self._other_job_still_needs_source(job):
            return
        self._unlink_source_safely(job.source_path)

    def _other_job_still_needs_source(self, job: VideoUpscaleJob) -> bool:
        return any(
            other.id != job.id
            and other.source_path == job.source_path
            and not self._is_job_finished(other)
            for other in self.jobs.values()
        )

    @staticmethod
    def _is_job_finished(job: VideoUpscaleJob) -> bool:
        return job.status in TERMINAL_JOB_STATUSES

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
