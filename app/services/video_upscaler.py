from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import cv2

from app.config import GMFSS_ENGINE, Settings
from app.models import VideoUpscaleJob
from app.services import video_encoders
from app.services.backend_registry import UpscaleBackend, resolve_upscale_backend
from app.services.devices_service import DevicesService
from app.services.restorer_registry import AudioRestorer
from app.services.engines.audio_enhance import AudioEnhancer
from app.services.engines.gmfss_engine import GmfssEngine
from app.services.engines.onnx_upscaler import OnnxUpscaler
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine, gpu_index_for_device
from app.services.engines.rife_ncnn import RifeNcnnEngine
from app.services.media_tools import (
    MediaTools,
    compute_interpolated_fps,
    compute_target_frame_count,
    format_fps_fraction,
    resolve_video_fps,
)
from app.services.model_registry import ModelKind, ModelRegistry
from app.services.process_runner import run_guarded_process
from app.services.progress import (
    advance_video_stage,
    apply_stage_transition,
    build_video_stages,
    complete_video_stages,
    compute_progress,
    frame_stage_fraction,
    resolve_frames_total,
)
from app.services.stall_watchdog import StallWatchdog

logger = logging.getLogger(__name__)

FRAME_POLL_INTERVAL_SECONDS = 1.0


# Not a SubprocessTimeoutError: a stall is "hung, no new output", not
# "hit the absolute 24h backstop" -- callers need to tell those apart.
class VideoStallError(RuntimeError):
    pass


def _stall_message(missing_signal: str, stall_timeout_seconds: float) -> str:
    stall_minutes = stall_timeout_seconds / 60
    return (
        f"El proceso parece estancado: sin {missing_signal} por {stall_minutes:.0f} min. "
        "Puede ser un problema del modelo/GPU."
    )


class VideoUpscaler:
    def __init__(
        self,
        settings: Settings,
        engine: RealEsrganNcnnEngine,
        media_tools: MediaTools,
        rife_engine: RifeNcnnEngine | None = None,
        gmfss_engine: GmfssEngine | None = None,
        audio_enhancers: dict[str, AudioEnhancer] | None = None,
        onnx_engine: OnnxUpscaler | None = None,
        model_registry: ModelRegistry | None = None,
        frame_poll_interval_seconds: float = FRAME_POLL_INTERVAL_SECONDS,
        frame_stall_timeout_seconds: float | None = None,
        restorers: dict[str, AudioRestorer] | None = None,
        onnx_video_engine: OnnxVideoUpscaler | None = None,
        devices: DevicesService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.media_tools = media_tools
        self.rife_engine = rife_engine
        self.gmfss_engine = gmfss_engine
        self.audio_enhancers = audio_enhancers or {}
        self.restorers = restorers or {}
        self.onnx_engine = onnx_engine
        self.onnx_video_engine = onnx_video_engine
        self.model_registry = model_registry
        self.devices = devices
        self.frame_poll_interval_seconds = frame_poll_interval_seconds
        self.frame_stall_timeout_seconds = (
            settings.frame_stall_timeout_seconds
            if frame_stall_timeout_seconds is None
            else frame_stall_timeout_seconds
        )

    def available(self) -> bool:
        return self.engine.available() and self.media_tools.available()

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        if not self.available():
            raise RuntimeError("Video pipeline is not available. Ensure Real-ESRGAN and FFmpeg are installed.")

        work_dir = self.settings.video_work_path / job.id
        frames_in = work_dir / "frames-in"
        frames_out = work_dir / "frames-out"
        audio_path = work_dir / "audio.m4a"
        work_dir.mkdir(parents=True, exist_ok=True)
        frames_in.mkdir(parents=True, exist_ok=True)
        frames_out.mkdir(parents=True, exist_ok=True)

        try:
            return await self._run_pipeline(job, frames_in, frames_out, audio_path, fps_multiplier)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _run_pipeline(
        self,
        job: VideoUpscaleJob,
        frames_in: Path,
        frames_out: Path,
        audio_path: Path,
        fps_multiplier: int = 1,
    ) -> Path:
        # Reuse the probe captured during job validation; only probe again for jobs
        # built without one (direct VideoUpscaler use / older callers).
        probe = job.probe or await self.media_tools.ffprobe_json(job.source_path)
        video_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
        has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
        if not video_stream:
            raise RuntimeError("No video stream found in the uploaded file")

        fps = str(resolve_video_fps(video_stream.get("avg_frame_rate"), video_stream.get("r_frame_rate")))
        # Set before the first advance_video_stage so build_video_stages excludes
        # the audio stages when the source carries no audio track.
        job.metadata["hasAudio"] = has_audio
        advance_video_stage(job, "probing")
        job.metadata["fps"] = fps
        job.metadata["sourceWidth"] = int(video_stream.get("width") or 0)
        job.metadata["sourceHeight"] = int(video_stream.get("height") or 0)
        job.metadata["duration"] = float(probe.get("format", {}).get("duration") or 0)
        job.metadata["framesTotal"] = resolve_frames_total(probe, video_stream, fps)

        advance_video_stage(job, "extracting_frames")
        async with self._track_frame_progress(job, frames_in, "extracting_frames"):
            await self._run_process(
                [
                    str(self.settings.ffmpeg_binary_path),
                    "-y",
                    "-i",
                    str(job.source_path),
                    "-fps_mode",
                    "passthrough",
                    "-threads",
                    str(self.settings.ffmpeg_decode_threads),
                    # Extracted frames are throwaway input for the upscaler, so pay
                    # the cheapest zlib level instead of ffmpeg's default.
                    "-compression_level",
                    "1",
                    str(frames_in / "%08d.png"),
                ]
            )

        audio_mux_path: Path | None = None
        audio_codec_args: list[str] = []
        if job.keep_audio and has_audio:
            prepared_audio_path, audio_codec_args = await self._prepare_audio(job, audio_path)
            audio_mux_path = self._usable_audio_or_none(prepared_audio_path)
        elif job.keep_audio and job.audio_enhance:
            job.metadata["audioEnhanced"] = "skipped_no_audio"
        elif job.keep_audio and job.audio_restore:
            job.metadata["audioRestored"] = "skipped_no_audio"

        output_path = self.settings.outputs_path / f"{job.id}.{job.output_container}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Interpolation runs FIRST, at SOURCE resolution: RIFE at 1080p is ~3.6x
        # faster than at the upscaled 4K (measured on a 7800 XT), and with the
        # interp done up front the raw-pipe (upscale+encode fused, no PNG
        # round-trip) becomes eligible for interpolated jobs too.
        upscale_src, encode_fps = await self._maybe_interpolate(
            job, frames_in, fps, fps_multiplier, job.target_fps
        )
        job.metadata["outputFps"] = encode_fps
        if upscale_src != frames_in:
            # Source frames are dead once interpolated; free the disk early
            # (peak footprint on a long episode is tens-hundreds of GB).
            await asyncio.to_thread(self._safe_rmtree, frames_in)

        # Raw-pipe fast path: any failure falls back to the file path.
        if await self._should_stream(job):
            streamed = await self._try_streaming(
                job, upscale_src, output_path, encode_fps, audio_mux_path, audio_codec_args
            )
            if streamed:
                self._finalize_output(job, output_path)
                await asyncio.to_thread(self._safe_rmtree, upscale_src)
                return output_path

        # The upscale denominator is the frame count it actually processes:
        # after interpolation that is more than the source framesTotal.
        upscale_frames_total = await asyncio.to_thread(self._count_frames, upscale_src)
        advance_video_stage(job, "upscaling_frames")
        async with self._track_frame_progress(
            job, frames_out, "upscaling_frames", frames_total=upscale_frames_total
        ):
            await self._upscale_frames(job, upscale_src, frames_out)

        await asyncio.to_thread(self._safe_rmtree, upscale_src)
        encode_frames_dir = frames_out

        # Resolve off the loop: for "auto" this enumerates devices (DXGI) to map
        # the job's GPU to a hardware encoder.
        encoder = await asyncio.to_thread(self._resolve_video_encoder, job)
        job.metadata["videoEncoder"] = encoder

        advance_video_stage(job, "encoding_video")
        async with self._track_encode_progress(output_path):
            await self._encode_with_fallback(
                job, encode_frames_dir, encode_fps, audio_mux_path, audio_codec_args, output_path, encoder
            )

        self._finalize_output(job, output_path)
        return output_path

    def _finalize_output(self, job: VideoUpscaleJob, output_path: Path) -> None:
        if not self._is_non_empty_file(output_path):
            raise RuntimeError("Video processing finished but no output file was produced")
        complete_video_stages(job)
        job.metadata["outputWidth"] = job.metadata["sourceWidth"] * job.scale
        job.metadata["outputHeight"] = job.metadata["sourceHeight"] * job.scale

    async def _upscale_frames(self, job: VideoUpscaleJob, frames_in: Path, frames_out: Path) -> None:
        # HF-installed ONNX models are onnx-only (arbitrary fp32 NCHW graphs) --
        # they always go through OnnxUpscaler, untouched by the runtime selector.
        if self._is_onnx_model(job.model_id):
            await self._upscale_frames_onnx(job, frames_in, frames_out)
            return
        # Builtin Real-ESRGAN model: the selector decides ncnn vs the optimized
        # onnx video engine (only the RUNTIME changes; the model is the same).
        # Off the event loop: the first resolve does a cold `import onnxruntime`
        # (native DLL load) + get_available_providers, which would otherwise stall
        # every concurrent job's progress polling on the loop thread.
        if await asyncio.to_thread(self._resolve_builtin_backend, job) == UpscaleBackend.onnx:
            await self._upscale_frames_onnx_builtin(job, frames_in, frames_out)
            return
        await self._upscale_frames_ncnn(job, frames_in, frames_out)

    def _resolve_builtin_backend(self, job: VideoUpscaleJob) -> UpscaleBackend:
        engine = self.onnx_video_engine
        onnx_model_available = (
            engine is not None and engine.available() and engine.builtin_onnx_available(job.model_name)
        )
        gpu_ep_available = engine is not None and engine.has_gpu_execution_provider()
        device = job.device or self.settings.default_device
        return resolve_upscale_backend(
            setting_backend=self.settings.upscale_backend,
            job_backend=job.backend,
            onnx_model_available=onnx_model_available,
            gpu_ep_available=gpu_ep_available,
            device=device,
        )

    def _is_onnx_model(self, model_id: str | None) -> bool:
        if model_id is None or self.model_registry is None:
            return False
        entry = self.model_registry.get(model_id)
        return entry is not None and entry.kind == ModelKind.onnx

    async def _upscale_frames_ncnn(self, job: VideoUpscaleJob, frames_in: Path, frames_out: Path) -> None:
        await self._run_process(
            [
                str(self.settings.engine_binary_path),
                "-i",
                str(frames_in),
                "-o",
                str(frames_out),
                "-n",
                job.model_name,
                "-s",
                str(job.scale),
                "-m",
                str(self.settings.engine_models_path),
                "-f",
                "png",
                "-g",
                gpu_index_for_device(job.device),
                "-j",
                self.settings.ncnn_upscale_threads,
            ]
        )

    async def _upscale_frames_onnx(self, job: VideoUpscaleJob, frames_in: Path, frames_out: Path) -> None:
        if self.onnx_engine is None:
            raise RuntimeError(f"Model {job.model_id!r} requires the ONNX engine, which is not configured")
        device = job.device or self.settings.default_device
        await self.onnx_engine.run_frames(frames_in, frames_out, job.model_id, device)

    async def _upscale_frames_onnx_builtin(
        self, job: VideoUpscaleJob, frames_in: Path, frames_out: Path
    ) -> None:
        if self.onnx_video_engine is None:
            raise RuntimeError("ONNX backend selected but the ONNX video engine is not configured")
        device = job.device or self.settings.default_device
        await self.onnx_video_engine.run_frames_builtin(frames_in, frames_out, job.model_name, device)

    async def _prepare_audio(self, job: VideoUpscaleJob, audio_path: Path) -> tuple[Path, list[str]]:
        if job.audio_enhance or job.audio_restore:
            return await self._prepare_processed_audio(job, audio_path)
        return await self._prepare_original_audio(job, audio_path)

    def _usable_audio_or_none(self, prepared_audio_path: Path) -> Path | None:
        # ffmpeg can exit 0 without producing a usable track for exotic
        # codecs; the pre-enhance pipeline gated the mux on the file existing
        # and silently encoded a muted video instead of failing the whole job,
        # so this guard keeps that contract.
        if self._is_non_empty_file(prepared_audio_path):
            return prepared_audio_path
        logger.warning(
            "Prepared audio track %s is missing or empty; encoding without audio", prepared_audio_path
        )
        return None

    async def _prepare_original_audio(self, job: VideoUpscaleJob, audio_path: Path) -> tuple[Path, list[str]]:
        advance_video_stage(job, "extracting_audio")
        await self._run_process(
            [
                str(self.settings.ffmpeg_binary_path),
                "-y",
                "-i",
                str(job.source_path),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(audio_path),
            ]
        )
        return audio_path, ["-c:a", "copy"]

    async def _prepare_processed_audio(self, job: VideoUpscaleJob, audio_path: Path) -> tuple[Path, list[str]]:
        # Extract once, then chain denoise -> restore on the WAV track (same
        # order as the standalone pipeline). Any processed track is re-encoded
        # to AAC at mux time, never copied.
        current = audio_path.with_name("audio.wav")
        await self._extract_audio_wav(job, current)

        if job.audio_enhance:
            audio_enhanced_path = audio_path.with_name("audio-enhanced.wav")
            await self._enhance_audio(job, current, audio_enhanced_path)
            current = audio_enhanced_path
            job.metadata["audioEnhanced"] = True

        if job.audio_restore:
            audio_restored_path = audio_path.with_name("audio-restored.wav")
            await self._restore_audio(job, current, audio_restored_path)
            current = audio_restored_path
            job.metadata["audioRestored"] = True

        return current, ["-c:a", "aac", "-b:a", "192k"]

    async def _extract_audio_wav(self, job: VideoUpscaleJob, audio_wav_path: Path) -> None:
        # DeepFilterNet requires 48kHz input; a lossless PCM extraction avoids
        # compounding lossy re-encodes before the enhancer runs.
        advance_video_stage(job, "extracting_audio")
        await self._run_process(
            [
                str(self.settings.ffmpeg_binary_path),
                "-y",
                "-i",
                str(job.source_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "48000",
                str(audio_wav_path),
            ]
        )

    async def _enhance_audio(self, job: VideoUpscaleJob, input_wav: Path, output_wav: Path) -> None:
        enhancer = self.audio_enhancers.get(job.audio_enhance)
        if enhancer is None:
            raise RuntimeError(
                f"Audio enhance mode {job.audio_enhance!r} requested but no engine is configured"
            )
        advance_video_stage(job, "enhancing_audio")
        await enhancer.run(input_wav, output_wav)

    async def _restore_audio(self, job: VideoUpscaleJob, input_wav: Path, output_wav: Path) -> None:
        restorer = self.restorers.get(job.audio_restore or "")
        if restorer is None:
            raise RuntimeError(
                f"audio_restore mode {job.audio_restore!r} requested but no restorer is configured"
            )
        advance_video_stage(job, "restoring_audio")
        device = job.device or self.settings.default_device
        await restorer.run(input_wav, output_wav, device)

    async def _maybe_interpolate(
        self,
        job: VideoUpscaleJob,
        frames_out: Path,
        fps: str,
        fps_multiplier: int,
        target_fps: str | None = None,
    ) -> tuple[Path, str]:
        if not self._interpolation_requested(fps_multiplier, target_fps):
            return frames_out, fps

        engine = self._resolve_interpolation_engine(job)

        advance_video_stage(job, "interpolating_frames")
        frames_interp = frames_out.parent / "frames-interp"
        source_frame_count = self._count_frames(frames_out)
        # Interpolation emits MORE frames than the source (mult>1 / higher fps),
        # so the source framesTotal would clamp this stage to 100% halfway through.
        # The RIFE/GMFSS target count is the honest denominator for this stage only.
        interp_frames_total = self._interp_frames_total(
            source_frame_count, fps, fps_multiplier, target_fps
        )
        job.metadata["interpFramesTotal"] = interp_frames_total

        async with self._track_frame_progress(
            job, frames_interp, "interpolating_frames", frames_total=interp_frames_total
        ):
            if target_fps is not None:
                return await self._interpolate_to_target_fps(
                    engine, frames_out, frames_interp, source_frame_count, fps, target_fps, job.device
                )

            return await self._interpolate_by_multiplier(
                engine, frames_out, frames_interp, source_frame_count, fps, fps_multiplier, job.device
            )

    def _resolve_interpolation_engine(self, job: VideoUpscaleJob) -> RifeNcnnEngine | GmfssEngine:
        if job.interp_engine == GMFSS_ENGINE:
            if self.gmfss_engine is None:
                raise RuntimeError("Frame interpolation requested but no GMFSS engine is configured")
            return self.gmfss_engine
        if self.rife_engine is None:
            raise RuntimeError("Frame interpolation requested but no RIFE engine is configured")
        return self.rife_engine

    @staticmethod
    def _interp_frames_total(
        source_frame_count: int, fps: str, fps_multiplier: int, target_fps: str | None
    ) -> int | None:
        if target_fps is not None:
            return compute_target_frame_count(source_frame_count, fps, target_fps)
        return source_frame_count * fps_multiplier

    @staticmethod
    def _interpolation_requested(fps_multiplier: int, target_fps: str | None) -> bool:
        return target_fps is not None or fps_multiplier > 1

    async def _interpolate_to_target_fps(
        self,
        engine: RifeNcnnEngine | GmfssEngine,
        frames_out: Path,
        frames_interp: Path,
        source_frame_count: int,
        fps: str,
        target_fps: str,
        device: str | None = None,
    ) -> tuple[Path, str]:
        target_frame_count = compute_target_frame_count(source_frame_count, fps, target_fps)
        await engine.run(
            frames_out, frames_interp, source_frame_count, target_frame_count=target_frame_count, device=device
        )
        return frames_interp, format_fps_fraction(target_fps)

    async def _interpolate_by_multiplier(
        self,
        engine: RifeNcnnEngine | GmfssEngine,
        frames_out: Path,
        frames_interp: Path,
        source_frame_count: int,
        fps: str,
        fps_multiplier: int,
        device: str | None = None,
    ) -> tuple[Path, str]:
        await engine.run(
            frames_out, frames_interp, source_frame_count, fps_multiplier, device=device
        )

        new_rate = compute_interpolated_fps(fps, fps_multiplier)
        encode_fps = f"{new_rate.numerator}/{new_rate.denominator}"
        return frames_interp, encode_fps

    @staticmethod
    def _count_frames(directory: Path) -> int:
        # os.scandir instead of glob: this runs on every progress poll over dirs
        # with thousands of entries, and glob builds a Path object per entry.
        # The `with` block matters on Windows -- a leaked dir handle blocks
        # renaming/deleting the directory afterwards.
        # Poller may start tracking before the stage creates its output dir
        # (e.g. frames-interp only appears once the RIFE engine runs).
        try:
            with os.scandir(directory) as entries:
                return sum(1 for entry in entries if entry.name.endswith(".png"))
        except (FileNotFoundError, NotADirectoryError):
            return 0

    @contextlib.asynccontextmanager
    async def _track_frame_progress(
        self, job: VideoUpscaleJob, output_dir: Path, stage_key: str, frames_total: int | None = None
    ) -> AsyncIterator[None]:
        # framesDone is job-wide but each frame stage counts ITS OWN output
        # dir from zero. Without this reset the previous stage's final count
        # (== its framesTotal) survives via setdefault in advance_video_stage,
        # so the next stage's first poll reads fraction=framesDone/framesTotal=1.0
        # and the bar freezes at that stage's ceiling. Scoping the max() to this
        # stage keeps progress honest and live.
        job.metadata["framesDone"] = 0
        stage_task = asyncio.current_task()
        stall_watchdog = StallWatchdog(self.frame_stall_timeout_seconds)
        poller = asyncio.create_task(
            self._poll_frame_progress(job, output_dir, stage_key, frames_total, stall_watchdog, stage_task)
        )
        try:
            yield
        except asyncio.CancelledError:
            if not stall_watchdog.triggered:
                raise
            raise VideoStallError(_stall_message("frames nuevos", self.frame_stall_timeout_seconds)) from None
        finally:
            await self._stop_poller(poller)
            # Final authoritative count so framesDone reaches the true total
            # even if the stage finished between two poll ticks.
            await self._refresh_frame_progress(job, output_dir, stage_key, frames_total)

    @staticmethod
    async def _stop_poller(poller: asyncio.Task[None]) -> None:
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller

    async def _poll_frame_progress(
        self,
        job: VideoUpscaleJob,
        output_dir: Path,
        stage_key: str,
        frames_total: int | None,
        stall_watchdog: StallWatchdog,
        stage_task: asyncio.Task[None],
    ) -> None:
        while True:
            await asyncio.sleep(self.frame_poll_interval_seconds)
            frames_done = await self._refresh_frame_progress(job, output_dir, stage_key, frames_total)
            if frames_done is not None and stall_watchdog.observe(frames_done):
                stage_task.cancel()
                return

    async def _refresh_frame_progress(
        self, job: VideoUpscaleJob, output_dir: Path, stage_key: str, frames_total: int | None
    ) -> int | None:
        # The whole tick is guarded (count + metadata mutation): a failure here
        # must never propagate out of the poller task, or `await poller` in the
        # finally would re-raise it and mask the stage's own exception. Returns
        # None on failure so the stall watchdog skips that tick instead of
        # misreading a transient stat error as zero progress.
        try:
            frames_done = await asyncio.to_thread(self._count_frames, output_dir)
            self._apply_frame_progress(job, stage_key, frames_done, frames_total)
            return frames_done
        except Exception:  # noqa: BLE001
            logger.exception("Frame progress poll failed for stage %s in %s", stage_key, output_dir)
            return None

    def _apply_frame_progress(
        self, job: VideoUpscaleJob, stage_key: str, frames_done: int, frames_total: int | None = None
    ) -> None:
        job.metadata["framesDone"] = max(int(job.metadata.get("framesDone") or 0), frames_done)
        stages = apply_stage_transition(build_video_stages(job), stage_key)
        # Interpolation passes its own (larger) target count; extract/upscale are
        # 1:1 with the source so they fall back to framesTotal.
        denominator = frames_total if frames_total is not None else job.metadata.get("framesTotal")
        fraction = frame_stage_fraction(job.metadata["framesDone"], denominator)
        progress = compute_progress(stages, current_fraction=fraction)
        job.metadata["progress"] = max(float(job.metadata.get("progress") or 0.0), progress)

    @contextlib.asynccontextmanager
    async def _track_encode_progress(self, output_path: Path) -> AsyncIterator[None]:
        stage_task = asyncio.current_task()
        stall_watchdog = StallWatchdog(self.frame_stall_timeout_seconds)
        poller = asyncio.create_task(self._poll_encode_progress(output_path, stall_watchdog, stage_task))
        try:
            yield
        except asyncio.CancelledError:
            if not stall_watchdog.triggered:
                raise
            raise VideoStallError(
                _stall_message("crecimiento del archivo de salida", self.frame_stall_timeout_seconds)
            ) from None
        finally:
            await self._stop_poller(poller)

    async def _poll_encode_progress(
        self, output_path: Path, stall_watchdog: StallWatchdog, stage_task: asyncio.Task[None]
    ) -> None:
        while True:
            await asyncio.sleep(self.frame_poll_interval_seconds)
            output_size = await self._safe_output_file_size(output_path)
            if output_size is not None and stall_watchdog.observe(output_size):
                stage_task.cancel()
                return

    async def _safe_output_file_size(self, output_path: Path) -> int | None:
        # Guarded the same way as _refresh_frame_progress: a transient stat
        # failure must never propagate out of the poller task.
        try:
            return await asyncio.to_thread(self._output_file_size, output_path)
        except Exception:  # noqa: BLE001
            logger.exception("Encode progress poll failed for %s", output_path)
            return None

    @staticmethod
    def _output_file_size(output_path: Path) -> int:
        return output_path.stat().st_size if output_path.exists() else 0

    @staticmethod
    def _is_non_empty_file(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    @staticmethod
    def _safe_rmtree(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def _resolve_video_encoder(self, job: VideoUpscaleJob) -> str:
        """Concrete ffmpeg encoder for this job. "software" (or any non-"auto")
        keeps the picked codec (libx264/libx265). "auto" maps the job's GPU to a
        hardware encoder, falling back to the software codec if the vendor can't
        be resolved (cpu device, unknown GPU, no devices service)."""
        if job.video_encoder != video_encoders.VIDEO_ENCODER_AUTO:
            return job.video_codec
        hw = video_encoders.resolve_hardware_encoder(self._device_name(job.device), job.video_codec)
        return hw or job.video_codec

    def _device_name(self, device_id: str | None) -> str | None:
        if self.devices is None or not device_id:
            return None
        return next((d["name"] for d in self.devices.list_devices() if d["id"] == device_id), None)

    def _build_video_encode_options(self, job: VideoUpscaleJob, encoder: str) -> list[str]:
        return video_encoders.encode_options(
            encoder=encoder,
            crf=job.crf,
            preset=job.video_preset,
            x265_pools=self.settings.ffmpeg_x265_threads,
            software_threads=self.settings.ffmpeg_encode_threads,
        )

    def _build_encode_command(
        self,
        job: VideoUpscaleJob,
        encode_frames_dir: Path,
        encode_fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        output_path: Path,
        encoder: str,
    ) -> list[str]:
        cmd = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-framerate",
            encode_fps,
            "-i",
            str(encode_frames_dir / "%08d.png"),
        ]
        next_input_index = 1
        if audio_mux_path is not None:
            cmd += ["-i", str(audio_mux_path), "-map", "0:v:0", "-map", f"{next_input_index}:a:0"]
            next_input_index += 1
        source_input_index = self._maybe_add_source_input(cmd, job, next_input_index)
        if source_input_index is not None:
            self._map_extra_audio_tracks(cmd, job, source_input_index)
            self._map_subtitles(cmd, job, source_input_index)
        cmd += self._build_video_encode_options(job, encoder)
        if audio_mux_path is not None:
            cmd += audio_codec_args
        if job.keep_subtitles:
            cmd += ["-c:s", "copy"]
        cmd.append(str(output_path))
        return cmd

    def _extra_audio_track_indices(self, job: VideoUpscaleJob) -> list[int]:
        # The PRIMARY track (index 0 of the list) is already covered by
        # audio_mux_path -- it went through enhance/restore/copy separately.
        # Only a SECOND-or-later selected track counts as "extra" here.
        if not job.audio_track_indices or len(job.audio_track_indices) <= 1:
            return []
        return job.audio_track_indices[1:]

    def _needs_source_input(self, job: VideoUpscaleJob) -> bool:
        return bool(self._extra_audio_track_indices(job)) or job.keep_subtitles

    def _maybe_add_source_input(self, cmd: list[str], job: VideoUpscaleJob, input_index: int) -> int | None:
        if not self._needs_source_input(job):
            return None
        cmd += ["-i", str(job.source_path)]
        return input_index

    def _map_extra_audio_tracks(self, cmd: list[str], job: VideoUpscaleJob, source_input_index: int) -> None:
        # Absolute ffprobe stream index (e.g. "2:3"), not the relative "2:a:N"
        # form -- it maps the original file's stream directly, sidestepping
        # having to re-enumerate which "audio Nth" an absolute index is.
        for stream_index in self._extra_audio_track_indices(job):
            cmd += ["-map", f"{source_input_index}:{stream_index}"]

    def _map_subtitles(self, cmd: list[str], job: VideoUpscaleJob, source_input_index: int) -> None:
        if not job.keep_subtitles:
            return
        cmd += ["-map", f"{source_input_index}:s?"]

    async def _encode_with_fallback(
        self,
        job: VideoUpscaleJob,
        encode_frames_dir: Path,
        encode_fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        output_path: Path,
        encoder: str,
    ) -> None:
        cmd = self._build_encode_command(
            job, encode_frames_dir, encode_fps, audio_mux_path, audio_codec_args, output_path, encoder
        )
        try:
            await self._run_process(cmd)
            return
        except RuntimeError:
            # A software encoder failure is a real error; only a hardware encoder
            # (driver/init/EP quirk) is worth retrying on the software path.
            if not video_encoders.is_hardware_encoder(encoder):
                raise
        logger.warning(
            "hardware encoder %s failed; falling back to software %s", encoder, job.video_codec
        )
        job.metadata["videoEncoder"] = job.video_codec
        job.metadata["videoEncoderFallback"] = encoder
        software_cmd = self._build_encode_command(
            job, encode_frames_dir, encode_fps, audio_mux_path, audio_codec_args, output_path, job.video_codec
        )
        await self._run_process(software_cmd)

    # --- raw-pipe streaming (upscale + encode fused, no PNG round-trip) -----

    async def _should_stream(self, job: VideoUpscaleJob) -> bool:
        if not self.settings.enable_raw_pipe or self.onnx_video_engine is None:
            return False
        # HF-installed ONNX models use OnnxUpscaler (arbitrary graphs), not the
        # builtin streaming engine, so they stay on the file path.
        if self._is_onnx_model(job.model_id):
            return False
        # Only _build_encode_command (the PNG-based file path) knows how to map
        # extra audio tracks / subtitles from the original source (Fase A Task
        # 3) -- _build_rawpipe_command doesn't, so a job that needs to preserve
        # them must not take the raw-pipe fast path.
        if self._needs_source_input(job):
            return False
        backend = await asyncio.to_thread(self._resolve_builtin_backend, job)
        return backend == UpscaleBackend.onnx

    async def _try_streaming(
        self,
        job: VideoUpscaleJob,
        frames_in: Path,
        output_path: Path,
        fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
    ) -> bool:
        """Upscale + encode by piping raw frames to ffmpeg. Returns True on success,
        False to fall back to the PNG path (a cancel / stall still propagates)."""
        encoder = await asyncio.to_thread(self._resolve_video_encoder, job)
        try:
            out_w, out_h = await asyncio.to_thread(self._output_dims, frames_in, job.scale)
        except Exception as exc:  # noqa: BLE001 - unreadable frame -> just use the file path
            logger.warning("raw-pipe: could not read input dims (%s); using PNG path", exc)
            return False

        # Below the threshold the PNG encode is cheap and the subprocess overhead
        # isn't worth it -- let the (more parallel) file path handle small outputs.
        if out_w * out_h < self.settings.raw_pipe_min_output_pixels:
            return False

        job.metadata["videoEncoder"] = encoder
        advance_video_stage(job, "upscaling_frames")
        try:
            await self._upscale_encode_streaming(
                job, frames_in, output_path, encoder, fps, audio_mux_path, audio_codec_args, out_w, out_h
            )
        except (asyncio.CancelledError, VideoStallError):
            raise
        except Exception as exc:  # noqa: BLE001 - any streaming failure -> PNG fallback
            logger.warning("raw-pipe streaming failed (%s); falling back to the PNG encode path", exc)
            job.metadata["rawPipeFallback"] = str(exc)
            output_path.unlink(missing_ok=True)
            return False

        job.metadata["rawPipe"] = True
        job.metadata["outputFps"] = fps
        return True

    @staticmethod
    def _output_dims(frames_in: Path, scale: int) -> tuple[int, int]:
        first = next(iter(sorted(frames_in.glob("*.png"))), None)
        if first is None:
            raise RuntimeError("no extracted frames to stream")
        image = cv2.imread(str(first), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read first frame: {first}")
        height, width = image.shape[:2]
        return width * scale, height * scale

    def _build_rawpipe_command(
        self,
        out_w: int,
        out_h: int,
        fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        output_path: Path,
        job: VideoUpscaleJob,
        encoder: str,
    ) -> list[str]:
        # Upscaled frames are RGB HWC uint8 (see OnnxVideoUpscaler / _load_frame),
        # so the raw input is rgb24 at the upscaled size.
        cmd = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{out_w}x{out_h}",
            "-framerate", fps,
            "-i", "-",
        ]
        if audio_mux_path is not None:
            cmd += ["-i", str(audio_mux_path), "-map", "0:v:0", "-map", "1:a:0"]
        cmd += self._build_video_encode_options(job, encoder)
        if audio_mux_path is not None:
            cmd += audio_codec_args
        cmd.append(str(output_path))
        return cmd

    async def _upscale_encode_streaming(
        self,
        job: VideoUpscaleJob,
        frames_in: Path,
        output_path: Path,
        encoder: str,
        fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        out_w: int,
        out_h: int,
    ) -> None:
        cmd = self._build_rawpipe_command(
            out_w, out_h, fps, audio_mux_path, audio_codec_args, output_path, job, encoder
        )
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL
        )
        stderr_buf: list[bytes] = []
        stderr_thread = threading.Thread(target=self._drain_stream, args=(proc.stderr, stderr_buf), daemon=True)
        stderr_thread.start()

        counter = {"n": 0}

        def write_frame(frame_hwc_rgb) -> None:
            proc.stdin.write(frame_hwc_rgb.tobytes())  # blocks on pipe backpressure (in a worker thread)
            counter["n"] += 1

        device = job.device or self.settings.default_device
        # Shield the engine task so a job cancel doesn't tear it down while a worker
        # thread is blocked writing to ffmpeg's pipe. We kill ffmpeg FIRST (which
        # unblocks that write with BrokenPipe so the engine unwinds), then await it.
        stream_task = asyncio.ensure_future(
            self.onnx_video_engine.run_frames_streaming(frames_in, job.model_name, device, write_frame)
        )
        try:
            async with self._track_streaming_progress(job, counter):
                expected = await asyncio.shield(stream_task)
            # Guard against a silent frame drop (mirrors _validate_frame_output_count
            # on the PNG path): fewer frames written than the engine reported means a
            # short video -- raise so _try_streaming falls back to the file path.
            if counter["n"] != expected:
                raise RuntimeError(f"raw-pipe wrote {counter['n']}/{expected} frames")
            proc.stdin.close()
            returncode = await asyncio.to_thread(proc.wait)
            if returncode != 0:
                raise RuntimeError(self._summarize_process_error(b"".join(stderr_buf), b""))
        except BaseException:
            proc.kill()
            with contextlib.suppress(BaseException):
                await stream_task
            with contextlib.suppress(Exception):
                await asyncio.to_thread(proc.wait)
            raise
        finally:
            stderr_thread.join(timeout=5)

    @staticmethod
    def _drain_stream(stream, sink: list[bytes]) -> None:
        # ffmpeg fills its stderr pipe; if nobody drains it, the raw stdin writer
        # deadlocks once that pipe buffer is full. Keep only the tail for errors.
        try:
            for chunk in iter(lambda: stream.read(8192), b""):
                sink.append(chunk)
                if len(sink) > 64:
                    del sink[:-64]
        except Exception:  # noqa: BLE001 - stream closed on kill
            pass

    @contextlib.asynccontextmanager
    async def _track_streaming_progress(
        self, job: VideoUpscaleJob, counter: dict[str, int]
    ) -> AsyncIterator[None]:
        # Like _track_frame_progress but reads an in-memory frame counter (there are
        # no output files to count in the raw-pipe path). Same stall-watchdog
        # semantics: cancel the stage if no new frame is written within the timeout.
        frames_total = job.metadata.get("framesTotal")
        job.metadata["framesDone"] = 0
        stage_task = asyncio.current_task()
        stall_watchdog = StallWatchdog(self.frame_stall_timeout_seconds)
        poller = asyncio.create_task(self._poll_streaming_progress(job, counter, frames_total, stall_watchdog, stage_task))
        try:
            yield
        except asyncio.CancelledError:
            if not stall_watchdog.triggered:
                raise
            raise VideoStallError(_stall_message("frames nuevos", self.frame_stall_timeout_seconds)) from None
        finally:
            await self._stop_poller(poller)

    async def _poll_streaming_progress(
        self,
        job: VideoUpscaleJob,
        counter: dict[str, int],
        frames_total: int | None,
        stall_watchdog: StallWatchdog,
        stage_task: asyncio.Task[None],
    ) -> None:
        while True:
            await asyncio.sleep(self.frame_poll_interval_seconds)
            frames_done = counter["n"]
            self._apply_frame_progress(job, "upscaling_frames", frames_done, frames_total)
            if stall_watchdog.observe(frames_done):
                stage_task.cancel()
                return

    async def _run_process(self, command: list[str]) -> None:
        stdout, stderr, returncode = await run_guarded_process(command, self.settings.subprocess_timeout)
        if returncode != 0:
            raise RuntimeError(self._summarize_process_error(stderr, stdout))

    def _summarize_process_error(self, stderr: bytes, stdout: bytes) -> str:
        text = (stderr or stdout).decode("utf-8", errors="ignore")
        lowered = text.lower()

        if "cannot open libx265" in lowered or "incorrect parameters such as bit_rate, rate, width or height" in lowered:
            return (
                "FFmpeg failed while encoding with H.265/libx265. This usually happens because the selected "
                "x265 thread settings are invalid for the current input. Try H.264/libx264, or use a lower x265 "
                "thread count."
            )

        if "nothing was written into output file" in lowered:
            return "FFmpeg could not write the output video file. Check codec/container compatibility and selected options."

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else "External process failed"
