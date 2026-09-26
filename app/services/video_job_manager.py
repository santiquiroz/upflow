from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from app.config import AUDIO_ENHANCE_MODES, AUDIO_RESTORE_MODES, GMFSS_ENGINE, INTERP_ENGINES, RIFE_ENGINE, Settings
from app.models import TERMINAL_JOB_STATUSES, CctvOptions, JobStatus, VideoUpscaleJob
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.backend_registry import (
    get_builtin_onnx_model,
    ncnn_produces_correct_output,
    validate_backend_choice,
)
from app.services.video_encoders import VIDEO_ENCODERS
from app.services.device_router import DeviceRouter, has_compatible_device
from app.services.restorer_registry import validate_restore_mode_ready
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.media_tools import MediaTools, parse_fps_fraction, resolve_video_fps
from app.services.missing_pack import missing_pack_message
from app.services.classic_upscalers import (
    CLASSIC_SCALES,
    is_classic_upscaler,
    swscale_flag_for,
)
from app.services.job_manager_base import QueuedJobManager
from app.services.model_registry import ModelKind, ModelRegistry, ModelStatus
from app.services.target_resolution import smallest_scale_reaching
from app.services.cctv_chain import CctvChainError
from app.services.cctv_ingest import VerifiedCopyMismatch
from app.services.cctv_job_runner import frame_geometry
from app.services.cctv_job_validation import CctvJobFacts, CctvJobPlan, plan_cctv_job
from app.services.cctv_session import CctvSession, cctv_job_dir, load_session, prepare_job_dir, touch_session
from app.services.ffmpeg_capabilities import FfmpegCapabilities, cached_capabilities
from app.services.video_upscaler import VideoUpscaler
from app.exceptions import QueueFullError

logger = logging.getLogger(__name__)

MAX_TARGET_FPS = 240
SESSION_CHANGED = "cctv.error.sessionChanged"
CCTV_VIEWING_CONTAINER = "mp4"
CCTV_VIEWING_CODEC = "libx264"
CCTV_VIEWING_PRESET = "medium"
CCTV_VIEWING_CRF = 16


def session_changed_or(exc: Exception) -> Exception:
    if isinstance(exc, VerifiedCopyMismatch):
        return CctvChainError(SESSION_CHANGED, "The uploaded file changed after it was analyzed; upload it again.")
    return exc


@dataclass(frozen=True, slots=True)
class VideoModelResolution:
    model_id: str
    engine_model_name: str
    kind: ModelKind
    scale: int


class VideoJobManager(QueuedJobManager[VideoUpscaleJob]):
    queue_full_message = "Video job queue is full; try again later"
    worker_name_prefix = "video-upscale-worker"

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
        quota_service: QuotaService | None = None,
        cctv_capabilities: Callable[[], FfmpegCapabilities] | None = None,
    ) -> None:
        super().__init__(
            settings,
            quota_service=quota_service,
            worker_count=settings.max_concurrent_jobs,
        )
        self.upscaler = upscaler
        self.media_tools = media_tools
        self.registry = registry
        self.devices = devices
        self.device_semaphores = device_semaphores
        self.device_router = device_router or DeviceRouter(device_semaphores)
        self.cctv_capabilities = cctv_capabilities or (lambda: cached_capabilities(settings.ffmpeg_binary_path))

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
        target_height: int | None = None,
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
        owner: AuthenticatedUser | None = None,
    ) -> VideoUpscaleJob:
        resolved_source_path = self._resolve_source_path(source_path, upload_token)
        source_fps, probe = await self._validate_video(resolved_source_path)
        self._validate_target_height(target_height)
        validate_backend_choice(backend)
        self._validate_video_encoder(video_encoder)
        resolved_model_id = model_id if model_id is not None else model_name
        # El objetivo MANDA sobre la escala. Sin esto el modelo corria al
        # multiplicador del perfil y despues ffmpeg tiraba los pixeles: lo peor de
        # los dos mundos, el mismo costo en GPU y un resultado chico.
        scale = self._scale_for_target(probe, target_height, resolved_model_id, scale)
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
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = VideoUpscaleJob(
            target_height=target_height,
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
            owner_id=owner.id if owner is not None else None,
        )
        if container_upgrade_reason is not None:
            job.metadata["containerUpgradedReason"] = container_upgrade_reason
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    async def create_cctv_job(
        self,
        *,
        cctv: CctvOptions,
        device: str | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
    ) -> VideoUpscaleJob:
        job = await self._admit_cctv_job(cctv, device, job_id, owner)
        self._enqueue_cctv(job)
        return job

    async def run_cctv_inline(self, *, cctv: CctvOptions, device: str | None = None) -> VideoUpscaleJob:
        # Puerta del modo headless (CLI / MCP in-process): sin cola ni retencion, un fallo no deja el directorio.
        job = await self._admit_cctv_job(cctv, device, None, None)
        job.status = JobStatus.running
        try:
            job.output_path = await self.upscaler.run(job)
        except BaseException:
            job.status = JobStatus.failed
            await asyncio.to_thread(shutil.rmtree, cctv_job_dir(self.settings.outputs_path, job.id), True)
            raise
        job.status = JobStatus.completed
        return job

    async def _admit_cctv_job(
        self, cctv: CctvOptions, device: str | None, job_id: str | None, owner: AuthenticatedUser | None
    ) -> VideoUpscaleJob:
        session = await asyncio.to_thread(load_session, self.settings.video_work_path, cctv.session_token)
        plan = plan_cctv_job(cctv, await self._cctv_facts(session, cctv.task), device)
        if plan.device is not None and plan.device != AUTO_DEVICE_ID and self.devices is not None:
            await asyncio.to_thread(self.devices.validate, plan.device)
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)
        job = self._cctv_job(session, cctv, plan, owner)
        if job_id is not None:
            job.id = job_id
        await self._prepare_cctv_outputs(session, job)
        return job

    async def _cctv_facts(self, session: CctvSession, task: str) -> CctvJobFacts:
        probe = await self.media_tools.ffprobe_json(session.work)
        return CctvJobFacts(
            geometry=frame_geometry(probe),
            frame_count=session.frame_count,
            caps=await asyncio.to_thread(self.cctv_capabilities),
            restore_core_installed=bool(self.settings.restore_core_installed),
            task_available=self.upscaler.cctv_task_available(task),
            max_still_frames=self.settings.cctv_max_still_frames,
            max_roi_frames=self.settings.cctv_roi_max_frames,
        )

    @staticmethod
    def _cctv_job(
        session: CctvSession, cctv: CctvOptions, plan: CctvJobPlan, owner: AuthenticatedUser | None
    ) -> VideoUpscaleJob:
        # Los campos de codec son los de la copia de visualizacion: el job CCTV no los elige ni los valida.
        return VideoUpscaleJob(
            source_path=session.upload,
            original_filename=session.record.original_name,
            model_name=f"cctv-{cctv.task}",
            scale=1,
            output_container=CCTV_VIEWING_CONTAINER,
            video_codec=CCTV_VIEWING_CODEC,
            video_preset=CCTV_VIEWING_PRESET,
            crf=CCTV_VIEWING_CRF,
            keep_audio=True,
            device=plan.device,
            cctv=cctv,
            owner_id=owner.id if owner is not None else None,
            metadata={"cctv": {"task": cctv.task, "lane": plan.lane, "sourceSha256": session.record.sha256}},
        )

    async def _prepare_cctv_outputs(self, session: CctvSession, job: VideoUpscaleJob) -> None:
        job_dir = cctv_job_dir(self.settings.outputs_path, job.id)
        try:
            await asyncio.to_thread(prepare_job_dir, session, job_dir)
        except Exception as exc:
            await asyncio.to_thread(shutil.rmtree, job_dir, True)
            raise session_changed_or(exc) from exc
        await asyncio.to_thread(touch_session, session.directory)

    def _enqueue_cctv(self, job: VideoUpscaleJob) -> None:
        try:
            self._enqueue(job)
        except QueueFullError:
            shutil.rmtree(cctv_job_dir(self.settings.outputs_path, job.id), ignore_errors=True)
            raise
        self.jobs[job.id] = job

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

    def _source_dimensions(self, probe: dict) -> tuple[int, int] | None:
        stream = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None
        )
        if stream is None:
            return None
        width, height = stream.get("width"), stream.get("height")
        if not width or not height:
            return None
        return int(width), int(height)

    def _scale_for_target(
        self, probe: dict, target_height: int | None, model_id: str, requested_scale: int
    ) -> int:
        """La escala del modelo que hace falta para llegar al objetivo.

        Sin esto, pedir 1080p sobre una fuente 4K corria el modelo al x4 del perfil
        (15360x8640, horas) y despues reducia a 1920x1080: mismo costo que el
        multiplicador ciego y encima tirando el trabajo.

        Se acota a lo que el modelo elegido soporta: pedir una escala que no tiene
        haria fallar la resolucion del modelo mas adelante.
        """
        if is_classic_upscaler(model_id):
            # El clasico no escala por pasos enteros: swscale va a cualquier medida en
            # una sola pasada. Con un objetivo pedido la escala deja de significar algo,
            # asi que se fija en 1 y el redimensionado hace todo el trabajo. Esto es lo
            # que habilita el caso 4K a 1080p, que ningun modelo puede resolver.
            return requested_scale if target_height is None else 1
        if target_height is None:
            return requested_scale
        dimensions = self._source_dimensions(probe)
        if dimensions is None:
            # Sin dimensiones no se puede planificar; se respeta lo pedido en vez de
            # adivinar.
            return requested_scale

        _source_width, source_height = dimensions
        allowed = tuple(self._scales_for_model(model_id))
        needed = smallest_scale_reaching(source_height, target_height, allowed)
        if needed is None:
            raise ValueError(
                f"La fuente ya mide {source_height}p, asi que llegar a "
                f"{target_height}p no necesita escalado por modelo -- solo un "
                "redimensionado. Ese camino todavia no esta implementado: usa un "
                "objetivo mayor que la fuente, o el multiplicador."
            )
        return needed

    def _scales_for_model(self, model_id: str) -> list[int]:
        option = self.settings.get_model_option(model_id)
        if option and option.get("scales"):
            return list(option["scales"])
        return list(self.settings.allowed_scale_values)

    def _validate_target_height(self, target_height: int | None) -> None:
        """El alto pedido tiene que ser razonable y par.

        Se acota arriba porque un objetivo absurdo (100000) llevaria al mismo problema
        que el multiplicador ciego: un pedido enorme que nada advierte. El limite es
        8640, o sea 8K, que ya es mas de lo que cualquier pantalla usa hoy.
        """
        if target_height is None:
            return
        if not 144 <= target_height <= 8640:
            raise ValueError(
                f"target_height must be between 144 and 8640, got {target_height}"
            )
        if target_height % 2:
            # yuv420p no acepta dimensiones impares y el encode falla al final del job.
            raise ValueError(f"target_height must be even, got {target_height}")

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
        if is_classic_upscaler(model_id):
            return self._resolve_classic_upscaler(model_id, scale)
        if model_id in self.settings.model_keys:
            return self._resolve_builtin_model(model_id, scale, device)
        return self._resolve_onnx_model(model_id)

    def _resolve_classic_upscaler(self, model_id: str, scale: int) -> VideoModelResolution:
        """El reescalado clasico no tiene modelo ni motor: lo hace swscale en el encode.

        Por eso no valida dispositivo -- corre en CPU y acepta 'cpu' explicito, al
        contrario de los builtin ncnn que exigen una GPU Vulkan.
        """
        if scale not in CLASSIC_SCALES:
            raise ValueError(
                f"Classic upscaler {model_id} supports only scales {list(CLASSIC_SCALES)}"
            )
        return VideoModelResolution(
            model_id=model_id,
            engine_model_name=swscale_flag_for(model_id),
            kind=ModelKind.classic,
            scale=scale,
        )

    def _resolve_builtin_model(self, model_id: str, scale: int, device: str | None) -> VideoModelResolution:
        option = self.settings.get_model_option(model_id)
        if option and scale not in option["scales"]:
            raise ValueError(f"Model {model_id} supports only scales {option['scales']}")
        self._require_a_runtime_that_produces_this_scale(model_id, scale)
        if device == "cpu":
            raise ValueError(
                f"Device 'cpu' is not supported for builtin model {model_id!r} (requires a Vulkan GPU device)"
            )
        engine_model_name = self.settings.resolve_engine_model_name(model_id, scale)
        return VideoModelResolution(
            model_id=model_id, engine_model_name=engine_model_name, kind=ModelKind.builtin_ncnn, scale=scale
        )

    def _require_a_runtime_that_produces_this_scale(self, model_id: str, scale: int) -> None:
        """Rechaza una combinacion que ningun runtime puede entregar bien.

        realesrgan-x4plus y su variante anime ofrecen 2x y 3x, pero SOLO por ONNX: el
        binario ncnn devuelve las dimensiones pedidas con el contenido roto (magnifica
        un cuadrante), y sale con codigo 0 -- o sea que sin esta guarda el usuario
        recibiria un video destruido sin un solo error.
        """
        engine_model_name = self.settings.resolve_engine_model_name(model_id, scale)
        if ncnn_produces_correct_output(engine_model_name, scale):
            return
        onnx_model = get_builtin_onnx_model(engine_model_name, scale)
        if onnx_model is not None and (self.settings.builtin_onnx_path / onnx_model.filename).exists():
            return
        raise ValueError(
            missing_pack_message(
                "realesrgan-onnx",
                detail=(
                    f"El modelo {model_id} solo puede hacer {scale}x con el runtime ONNX. "
                    f"Tambien podes usar escala 4 o elegir otro modelo."
                ),
            )
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
        if is_classic_upscaler(job.model_id):
            return ModelKind.classic
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
                missing_pack_message("rife", detail="Se pidio interpolacion de cuadros.")
            )

    def _validate_gmfss_ready(self) -> None:
        if not self.settings.enable_gmfss:
            raise ValueError(
                "GMFSS interpolation is disabled by configuration (set ENABLE_GMFSS=true)"
            )
        if not self.settings.gmfss_available():
            raise ValueError(
                missing_pack_message("gmfss", detail="Se pidio interpolacion GMFSS.")
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
                missing_pack_message(
                    "deepfilternet",
                    detail=f"Se pidio el modo de mejora de audio {audio_enhance!r}.",
                )
            )

    def _validate_audio_restore_mode(self, audio_restore: str | None, keep_audio: bool) -> None:
        if audio_restore is None:
            return
        if audio_restore not in AUDIO_RESTORE_MODES:
            raise ValueError(f"audio_restore must be one of {sorted(AUDIO_RESTORE_MODES)}")
        if not keep_audio:
            raise ValueError("audio_restore requires keep_audio to be enabled")
        validate_restore_mode_ready(self.settings, audio_restore)

    async def _dispatch(self, job: VideoUpscaleJob) -> None:
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

    async def _run_engine(self, job: VideoUpscaleJob) -> None:
        job.output_path = await self.upscaler.run(job, fps_multiplier=job.fps_multiplier)

    def _cleanup_source(self, job: VideoUpscaleJob) -> None:
        # El upload CCTV es de la sesion y se reusa en tareas siguientes; lo barre la retencion.
        if job.cctv is not None:
            return
        self._unlink_source_if_unused(job)

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
