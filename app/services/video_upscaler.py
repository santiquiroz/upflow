from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from app.config import GMFSS_ENGINE, Settings
from app.models import VideoUpscaleJob
from app.services.classic_upscalers import is_classic_upscaler, swscale_flag_for
from app.services.target_resolution import plan_for_scale, plan_for_target
from app.services import video_encoders
from app.services.backend_registry import (
    UpscaleBackend,
    ncnn_produces_correct_output,
    resolve_upscale_backend,
)
from app.services.devices_service import DevicesService
from app.services.encoder_capability_probe import EncoderCapabilityProbe
from app.services.engines.audio_enhance import AudioEnhancer
from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource
from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder
from app.services.engines.frame_workers import derive_readback_ring_capacity, load_frame
from app.services.engines.gmfss_engine import GmfssEngine
from app.services.engines.onnx_upscaler import OnnxUpscaler
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine, gpu_index_for_device
from app.services.engines.rife_ncnn import RifeNcnnEngine
from app.services.frame_pipeline import (
    FramePipeline,
    FrameStage,
    MapStage,
    derive_stream_queue_maxsizes,
    iter_png_frames,
)
from app.services.media_tools import (
    MediaTools,
    compute_interpolated_fps,
    compute_target_frame_count,
    format_fps_fraction,
    parse_fps_fraction,
    resolve_video_fps,
)
from app.services.model_registry import ModelKind, ModelRegistry
from app.services.process_runner import is_non_empty_file, run_guarded_process
from app.services.progress import (
    advance_video_stage,
    apply_stage_transition,
    build_video_stages,
    complete_video_stages,
    compute_progress,
    frame_stage_fraction,
    frames_total_is_exact,
    resolve_frames_total,
)
from app.services.restorer_registry import AudioRestorer
from app.services.scene_cuts import (
    DEFAULT_THRESHOLD as SCENE_CUT_THRESHOLD,
    build_scene_detect_command,
    parse_scene_cut_times,
    repair_interpolated_cuts,
    source_index_at_time,
)
from app.services.stall_watchdog import StallWatchdog

logger = logging.getLogger(__name__)

FRAME_POLL_INTERVAL_SECONDS = 1.0

# Cuántos renglones con señal de error se conservan al resumir un fallo de
# subproceso (ffmpeg reparte el diagnóstico en 2-3 líneas consecutivas).
_MAX_ERROR_CONTEXT_LINES = 3

# Bytes por píxel de los PNG intermedios. Son PISOS, no promedios: el guard
# responde "¿entra aunque el contenido comprima lo mejor posible?", así que
# subestimar a propósito evita bloquear jobs que sí entrarían. Medidos sobre
# material real (BDrip de animación 960x720, los flags que usa el pipeline):
#   frames-in/interp  ffmpeg -compression_level 1  ->  4.21 B/px (escribe PNG
#                     de 16 bits, por eso supera los 3 B/px de RGB8 crudo)
#   frames-out        cv2.imwrite 8 bits, compresión 1  ->  0.54 B/px (la
#                     animación upscaleada queda muy plana y comprime mucho;
#                     material fotográfico con grano sube bastante)
_PNG_SOURCE_BYTES_PER_PIXEL_FLOOR = 2.5
_PNG_UPSCALED_BYTES_PER_PIXEL_FLOOR = 0.4

# Margen sobre el espacio libre: nunca se planifica llenar el disco al ras.
_DISK_HEADROOM_FRACTION = 0.90


def estimate_png_workdir_bytes(
    *,
    source_width: int,
    source_height: int,
    scale: int,
    source_frame_count: int,
    output_frame_count: int,
    writes_upscaled_frames: bool,
    writes_interpolated_frames: bool,
) -> int:
    """Pico de disco del camino PNG clásico, en bytes.

    El work-dir llega a sostener tres sets a la vez: frames-in (resolución
    fuente), frames-interp (resolución fuente, más frames) y frames-out
    (upscaleado). El set upscaleado domina por scale² — a 4x son 16x los
    píxeles de la fuente — y es el que vuelve inviables los jobs largos.
    """
    source_pixels = max(0, source_width) * max(0, source_height)
    at_source_res = max(0, source_frame_count)
    if writes_interpolated_frames:
        at_source_res += max(0, output_frame_count)
    total = source_pixels * at_source_res * _PNG_SOURCE_BYTES_PER_PIXEL_FLOOR
    if writes_upscaled_frames:
        upscaled_pixels = source_pixels * (max(1, scale) ** 2)
        total += upscaled_pixels * max(0, output_frame_count) * _PNG_UPSCALED_BYTES_PER_PIXEL_FLOOR
    return int(total)

# Modos del stream pipeline (spec 2026-07-25-stream-frame-pipeline-design.md):
# "full" = decode→(GMFSS)→upscale→encode sin ningún PNG; "hybrid" = RIFE
# conserva su tramo PNG y solo upscale→encode va en streaming; None = clásico.
STREAM_MODE_FULL = "full"
STREAM_MODE_HYBRID = "hybrid"


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
        self.encoder_probe = EncoderCapabilityProbe(settings.ffmpeg_binary_path)
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
        # nb_frames del contenedor vs round(duration*fps): solo el primero es
        # exacto y sirve para validar por igualdad o para el plan de GMFSS.
        frames_total_exact = frames_total_is_exact(video_stream)

        output_path = self.settings.outputs_path / f"{job.id}.{job.output_container}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stream_mode = await self._resolve_stream_pipeline_mode(job, fps_multiplier, frames_total_exact)
        audio_mux_path: Path | None = None
        audio_codec_args: list[str] = []
        audio_prepared = False
        if stream_mode == STREAM_MODE_FULL:
            audio_mux_path, audio_codec_args = await self._prepare_audio_mux(job, has_audio, audio_path)
            audio_prepared = True
            if await self._try_stream_pipeline_full(
                job,
                fps_multiplier,
                output_path,
                format_fps_fraction(fps),
                audio_mux_path,
                audio_codec_args,
                frames_total_exact,
            ):
                self._finalize_output(job, output_path)
                return output_path
            stream_mode = None  # fallback: camino clásico desde cero

        # Antes de escribir el primer PNG: si el camino clásico no entra en el
        # disco, fallar acá con un mensaje accionable en vez de a mitad del
        # upscale con frames truncados.
        self._ensure_disk_room_for_png_path(job, frames_in.parent, fps_multiplier, stream_mode)

        advance_video_stage(job, "extracting_frames")
        async with self._track_frame_progress(job, frames_in, "extracting_frames"):
            await self._run_process(
                self._build_extract_frames_command(job.source_path, frames_in, fps)
            )

        if not audio_prepared:
            audio_mux_path, audio_codec_args = await self._prepare_audio_mux(job, has_audio, audio_path)

        encode_frames_dir, encode_fps = await self._interpolate_and_upscale(
            job, frames_in, frames_out, fps, fps_multiplier, output_path,
            audio_mux_path, audio_codec_args, stream_mode
        )
        # The raw-pipe path already produced AND finalized output_path; there is
        # no PNG directory left to encode.
        if encode_frames_dir is None:
            return output_path

        # Resolve off the loop: for "auto" this enumerates devices (DXGI) to map
        # the job's GPU to a hardware encoder, and probes that it can do the
        # output resolution before committing to it.
        out_w, out_h = self._expected_output_dims(job)
        encoder = await asyncio.to_thread(self._resolve_video_encoder, job, out_w, out_h)
        job.metadata["videoEncoder"] = encoder
        self._stamp_upscale_diagnostics(job)

        advance_video_stage(job, "encoding_video")
        async with self._track_encode_progress(output_path):
            await self._encode_with_fallback(
                job, encode_frames_dir, encode_fps, audio_mux_path, audio_codec_args, output_path, encoder
            )

        self._finalize_output(job, output_path)
        return output_path

    async def _interpolate_and_upscale(
        self,
        job: VideoUpscaleJob,
        frames_in: Path,
        frames_out: Path,
        fps: str,
        fps_multiplier: int,
        output_path: Path,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        stream_mode: str | None = None,
    ) -> tuple[Path | None, str]:
        """Interpolate (optional) then upscale into frames_out, returning
        (encode_frames_dir, encode_fps). encode_frames_dir is None when the
        raw-pipe path already produced AND finalized output_path (the caller
        returns output_path directly, skipping the PNG encode).

        Interpolation runs FIRST, at SOURCE resolution: RIFE at 1080p is ~3.6x
        faster than at the upscaled 4K (measured on a 7800 XT), and with the
        interp done up front the raw-pipe (upscale+encode fused, no PNG
        round-trip) becomes eligible for interpolated jobs too.
        """
        upscale_src, encode_fps = await self._maybe_interpolate(
            job, frames_in, fps, fps_multiplier, job.target_fps
        )
        job.metadata["outputFps"] = encode_fps
        if upscale_src != frames_in:
            # Source frames are dead once interpolated; free the disk early
            # (peak footprint on a long episode is tens-hundreds of GB).
            await asyncio.to_thread(self._safe_rmtree, frames_in)

        if is_classic_upscaler(job.model_id):
            # Sin IA no hay etapa de escalado: swscale reescala DENTRO del encode que
            # igual iba a correr, asi que los frames de entrada YA son los de salida.
            # Copiarlos a frames_out solo para no escalarlos duplicaria el disco (en 4K,
            # decenas de GB) sin cambiar un pixel.
            #
            # Sale ANTES del tramo streaming a proposito: ese camino existe para fusionar
            # escalado y encode, y aca no hay escalado que fusionar -- entrar correria el
            # motor de un modelo que este job no eligio.
            return upscale_src, encode_fps

        # Tramo streaming: con el pipeline nuevo activo (modo híbrido) va por
        # FramePipeline; si no (flag OFF / no elegible / fallback del modo
        # full), el raw-pipe legacy queda intacto. Ambos caen al camino PNG
        # ante cualquier fallo — el if/elif garantiza que un fallo del híbrido
        # NO reintenta con el raw-pipe legacy (iría directo al PNG clásico).
        if stream_mode == STREAM_MODE_HYBRID:
            if await self._try_stream_pipeline_from_dir(
                job, upscale_src, output_path, encode_fps, audio_mux_path, audio_codec_args
            ):
                self._finalize_output(job, output_path)
                await asyncio.to_thread(self._safe_rmtree, upscale_src)
                return None, encode_fps
        elif await self._should_stream(job):
            streamed = await self._try_streaming(
                job, upscale_src, output_path, encode_fps, audio_mux_path, audio_codec_args
            )
            if streamed:
                self._finalize_output(job, output_path)
                await asyncio.to_thread(self._safe_rmtree, upscale_src)
                return None, encode_fps

        # The upscale denominator is the frame count it actually processes:
        # after interpolation that is more than the source framesTotal.
        upscale_frames_total = await asyncio.to_thread(self._count_frames, upscale_src)
        advance_video_stage(job, "upscaling_frames")
        async with self._track_frame_progress(
            job, frames_out, "upscaling_frames", frames_total=upscale_frames_total
        ):
            await self._upscale_frames(job, upscale_src, frames_out)

        await asyncio.to_thread(self._safe_rmtree, upscale_src)
        return frames_out, encode_fps

    def _finalize_output(self, job: VideoUpscaleJob, output_path: Path) -> None:
        if not is_non_empty_file(output_path):
            raise RuntimeError("Video processing finished but no output file was produced")
        complete_video_stages(job)
        width, height = self._final_output_dims(job)
        job.metadata["outputWidth"] = width
        job.metadata["outputHeight"] = height

    def _final_output_dims(self, job: VideoUpscaleJob) -> tuple[int, int]:
        """Lo que el archivo REALMENTE mide.

        Con un alto objetivo el tamaño final es el del plan, no el del multiplicador:
        reportar sourceWidth*scale mentiria en la UI y en la cola de jobs.
        """
        source_width = job.metadata["sourceWidth"]
        source_height = job.metadata["sourceHeight"]
        if job.target_height is None:
            return source_width * job.scale, source_height * job.scale
        plan = plan_for_target(source_width, source_height, job.target_height)
        return plan.output_width, plan.output_height

    def _resize_filter_args(self, job: VideoUpscaleJob) -> list[str]:
        """El -vf que lleva a la medida exacta, o nada si no hace falta.

        Cumple dos papeles distintos. Con un modelo de IA es un AJUSTE: el modelo solo
        hace escalados enteros y un objetivo como 1080p casi nunca cae justo. Con un
        upscaler clasico es TODO el trabajo -- no corrio ningun motor, los frames siguen
        en resolucion de fuente, y este filtro es el que escala.
        """
        classic = is_classic_upscaler(job.model_id)
        if job.target_height is None and not classic:
            return []
        source_width = job.metadata.get("sourceWidth")
        source_height = job.metadata.get("sourceHeight")
        if not source_width or not source_height:
            return []

        if job.target_height is None:
            plan = plan_for_scale(source_width, source_height, job.scale)
        else:
            plan = plan_for_target(source_width, source_height, job.target_height)

        # Lo que ya se logro antes del encode. En el camino clasico no corrio nada, asi
        # que los frames siguen midiendo lo que la fuente.
        reached = (
            (source_width, source_height)
            if classic
            else (source_width * job.scale, source_height * job.scale)
        )
        if reached == (plan.output_width, plan.output_height):
            # Ya esta en la medida pedida: una pasada de ffmpeg al vicio.
            return []
        # El flag elegido cuando es clasico (es la opcion del usuario); lanczos para el
        # ajuste post-modelo, porque ahi el paso casi siempre es una REDUCCION y es el
        # que menos detalle pierde.
        flags = swscale_flag_for(job.model_id) if classic else "lanczos"
        return ["-vf", f"scale={plan.output_width}:{plan.output_height}:flags={flags}"]

    @staticmethod
    def _interpolated_encode_fps(fps: str, fps_multiplier: int, target_fps: str | None) -> str:
        # Mismos valores de encode-fps que devuelven _interpolate_to_target_fps /
        # _interpolate_by_multiplier, para que el pipeline encodee exactamente a
        # la misma tasa que el camino de dos pasadas.
        if target_fps is not None:
            return format_fps_fraction(target_fps)
        new_rate = compute_interpolated_fps(fps, fps_multiplier)
        return f"{new_rate.numerator}/{new_rate.denominator}"

    async def _upscale_frames(self, job: VideoUpscaleJob, frames_in: Path, frames_out: Path) -> None:
        # HF-installed ONNX models are onnx-only (arbitrary fp32 NCHW graphs) --
        # they always go through OnnxUpscaler, untouched by the runtime selector.
        if self._is_onnx_model(job.model_id):
            self._stamp_upscale_backend(job, "onnx")
            await self._upscale_frames_onnx(job, frames_in, frames_out)
            return
        # Builtin Real-ESRGAN model: the selector decides ncnn vs the optimized
        # onnx video engine (only the RUNTIME changes; the model is the same).
        # Off the event loop: the first resolve does a cold `import onnxruntime`
        # (native DLL load) + get_available_providers, which would otherwise stall
        # every concurrent job's progress polling on the loop thread.
        if await asyncio.to_thread(self._resolve_builtin_backend, job) == UpscaleBackend.onnx:
            self._stamp_upscale_backend(job, "onnx")
            await self._upscale_frames_onnx_builtin(job, frames_in, frames_out)
            return
        self._stamp_upscale_backend(job, "ncnn")
        await self._upscale_frames_ncnn(job, frames_in, frames_out)

    def _resolve_builtin_backend(self, job: VideoUpscaleJob) -> UpscaleBackend:
        # ncnn devuelve contenido ROTO para las escalas no nativas de un modelo (medido:
        # x4plus con -s 2 magnifica un cuadrante y sale con codigo 0). Ahi el runtime no
        # es una preferencia sino la unica opcion correcta, asi que la regla auto y
        # cualquier preferencia por ncnn quedan de lado. VideoJobManager ya rechazo el
        # job si el export ONNX no estaba, asi que llegar aca significa que existe.
        if not ncnn_produces_correct_output(job.model_name, job.scale):
            return UpscaleBackend.onnx

        engine = self.onnx_video_engine
        onnx_model_available = (
            engine is not None
            and engine.available()
            and engine.builtin_onnx_available(job.model_name, job.scale)
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
        await self.onnx_video_engine.run_frames_builtin(
            frames_in, frames_out, job.model_name, device, job.scale
        )

    async def _prepare_audio_mux(
        self, job: VideoUpscaleJob, has_audio: bool, audio_path: Path
    ) -> tuple[Path | None, list[str]]:
        # Extraído del cuerpo de _run_pipeline para poder prepararlo ANTES del
        # stream pipeline completo (que encodea sin pasar por el camino PNG).
        if job.keep_audio and has_audio:
            prepared_audio_path, audio_codec_args = await self._prepare_audio(job, audio_path)
            return self._usable_audio_or_none(prepared_audio_path), audio_codec_args
        if job.keep_audio and job.audio_enhance:
            job.metadata["audioEnhanced"] = "skipped_no_audio"
        elif job.keep_audio and job.audio_restore:
            job.metadata["audioRestored"] = "skipped_no_audio"
        return None, []

    async def _prepare_audio(self, job: VideoUpscaleJob, audio_path: Path) -> tuple[Path, list[str]]:
        if job.audio_enhance or job.audio_restore:
            return await self._prepare_processed_audio(job, audio_path)
        return await self._prepare_original_audio(job, audio_path)

    def _usable_audio_or_none(self, prepared_audio_path: Path) -> Path | None:
        # ffmpeg can exit 0 without producing a usable track for exotic
        # codecs; the pre-enhance pipeline gated the mux on the file existing
        # and silently encoded a muted video instead of failing the whole job,
        # so this guard keeps that contract.
        if is_non_empty_file(prepared_audio_path):
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
                *self._primary_audio_map_args(job),
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

        if self._wants_lossless_audio(job):
            return current, ["-c:a", "flac"]
        return current, ["-c:a", "aac", "-b:a", "192k"]

    @staticmethod
    def _wants_lossless_audio(job: VideoUpscaleJob) -> bool:
        # Mirrors VideoJobManager._resolve_output_container's wants_flac rule:
        # "auto" only upgrades to FLAC when a restore actually ran (enhance
        # alone stays lossy AAC); an explicit "flac" always wants it.
        return job.audio_output_format == "flac" or (
            job.audio_output_format == "auto" and job.audio_restore is not None
        )

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
                *self._primary_audio_map_args(job),
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
                resultado = await self._interpolate_to_target_fps(
                    engine, frames_out, frames_interp, source_frame_count, fps, target_fps, job.device
                )
            else:
                resultado = await self._interpolate_by_multiplier(
                    engine, frames_out, frames_interp, source_frame_count, fps, fps_multiplier, job.device
                )

        # Fuera del bloque de progreso: la reparacion no produce cuadros y
        # contarla como avance mentiria sobre lo que falta.
        await self._repair_scene_cuts(
            job,
            resultado[0],
            source_frame_count=source_frame_count,
            output_frame_count=interp_frames_total or source_frame_count,
        )
        return resultado

    async def _repair_scene_cuts(
        self,
        job: VideoUpscaleJob,
        frames_dir: Path,
        *,
        source_frame_count: int,
        output_frame_count: int,
    ) -> None:
        """Cambia por una copia los cuadros que la interpolacion invento a
        caballo de un corte.

        En un corte NO hay movimiento que estimar: las dos imagenes no comparten
        nada, y lo que sale es una mezcla. Duplicar el ultimo cuadro de la escena
        que termina es lo que hace cualquier interpolador serio.
        """
        try:
            cortes = await self._detect_scene_cuts(job.source_path, job.metadata.get("fps"))
            if not cortes:
                return
            reparados = await asyncio.to_thread(
                repair_interpolated_cuts,
                frames_dir,
                cut_indices=cortes,
                source_count=source_frame_count,
                output_count=output_frame_count,
            )
        except Exception:  # noqa: BLE001
            # Un video con fantasmas sigue siendo un video: perderlo entero
            # porque la deteccion fallo seria cambiar una molestia por un
            # desastre.
            logger.warning("scene-cut repair failed; leaving the interpolated frames as they are")
            return
        if reparados:
            job.metadata["sceneCutsRepaired"] = reparados

    async def _detect_scene_cuts(self, video: Path, fps: object) -> list[int]:
        fraccion = parse_fps_fraction(str(fps)) if fps is not None else None
        if not fraccion:
            return []
        cuadros_por_segundo = float(fraccion)
        command = build_scene_detect_command(
            ffmpeg=str(self.settings.ffmpeg_binary_path),
            video=video,
            threshold=SCENE_CUT_THRESHOLD,
        )
        # run_guarded_process y no create_subprocess_exec pelado: el scan
        # recorre el video entero y era el unico paso de este archivo sin
        # timeout ni kill-on-cancel.
        _stdout, stderr, _returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        tiempos = parse_scene_cut_times(stderr.decode("utf-8", "replace"))
        return [source_index_at_time(t, fps=cuadros_por_segundo) for t in tiempos]

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
    def _safe_rmtree(path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def _resolve_video_encoder(self, job: VideoUpscaleJob, out_width: int = 0, out_height: int = 0) -> str:
        """Concrete ffmpeg encoder for this job. "software" (or any non-"auto")
        keeps the picked codec. "auto" maps the job's GPU to a hardware encoder,
        falling back to the software codec if the vendor can't be resolved (cpu
        device, unknown GPU, no devices service) OR if that encoder can't handle
        the output resolution on this machine.

        El chequeo de resolución es un probe real, no una tabla: los techos
        varían por familia y generación (h264_amf falla a 7680x4320 y hevc_amf
        no, medido). Descubrirlo ANTES evita encodear en vano y que
        _encode_with_fallback rehaga todo por software horas después."""
        if job.video_encoder != video_encoders.VIDEO_ENCODER_AUTO:
            return job.video_codec
        hw = video_encoders.resolve_hardware_encoder(self._device_name(job.device), job.video_codec)
        if hw is None:
            return job.video_codec
        if out_width > 0 and out_height > 0 and not self.encoder_probe.supports(hw, out_width, out_height):
            logger.info(
                "hardware encoder %s no soporta %dx%d; se usa %s",
                hw, out_width, out_height, job.video_codec,
            )
            return job.video_codec
        return hw

    def _expected_output_dims(self, job: VideoUpscaleJob) -> tuple[int, int]:
        """Resolución de salida según el probe (0,0 si no se conoce todavía).

        Con un alto objetivo NO es fuente*escala: el encoder se elige probando que
        soporte esta resolución, así que pasarle la del multiplicador descartaría
        encoders por hardware capaces de la salida real.
        """
        width = int(job.metadata.get("sourceWidth") or 0)
        height = int(job.metadata.get("sourceHeight") or 0)
        if width <= 0 or height <= 0:
            return 0, 0
        if job.target_height is not None:
            plan = plan_for_target(width, height, job.target_height)
            return plan.output_width, plan.output_height
        return width * job.scale, height * job.scale

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

    def _stamp_upscale_diagnostics(self, job: VideoUpscaleJob) -> None:
        """Deja en job.metadata POR QUE este job corrio rapido o lento.

        Claves informativas, como streamPipeline / videoEncoder: no son de
        progreso y el frontend no cambia. Sin esto, un reporte de "va lentisimo
        en otra maquina" es adivinar: correr fp32 en vez de fp16 se midio 7.26x
        mas lento y caer a tiling 2.3x, y ninguna de las dos cosas se veia.
        """
        engine = self.onnx_video_engine
        if engine is None:
            return
        precision = getattr(engine, "last_precision", None)
        if precision is not None:
            job.metadata["upscalePrecision"] = precision
        job.metadata["upscaleTiled"] = bool(getattr(engine, "last_tiled", False))

    def _stamp_upscale_backend(self, job: VideoUpscaleJob, backend: str) -> None:
        """Que motor escalo. Sin esto, un job en ncnn no dejaba ninguna huella y
        "va lentisimo en otra maquina" seguia siendo adivinar: medido a 1080p->4x
        en una RX 7800 XT, ONNX fp16 da 1.58 fps y ncnn 0.72."""
        job.metadata["upscaleBackend"] = backend

    def _build_extract_frames_command(
        self, source_path: Path, frames_in: Path, fps: str
    ) -> list[str]:
        """Extrae los frames NORMALIZANDO a tasa constante.

        Con -fps_mode passthrough se conservaban los timestamps VFR de la fuente,
        pero el encode posterior los ignora y asume fps fijo: el video salía más
        corto que el audio y los subtítulos. En un BDrip real de cadencia mixta
        (23.976 + 29.97 declarando 29.97) eran 39.925 frames donde la tasa
        declarada implica 45.195, o sea 177 s de deriva en 25 minutos.

        Con -fps_mode cfr ffmpeg duplica/descarta según los timestamps reales, y
        a partir de ahí frames == duración × fps por construcción. Para una
        fuente que YA es CFR esto no cambia nada.
        """
        return [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(source_path),
            "-fps_mode",
            "cfr",
            "-r",
            fps,
            "-threads",
            str(self.settings.ffmpeg_decode_threads),
            # Extracted frames are throwaway input for the upscaler, so pay
            # the cheapest zlib level instead of ffmpeg's default.
            "-compression_level",
            "1",
            str(frames_in / "%08d.png"),
        ]

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
        # TODOS los -i van primero y los -map DESPUÉS. En ffmpeg -map es opción
        # de salida: intercalada entre dos inputs se aplica al -i siguiente y el
        # comando entero se rechaza con "Option map cannot be applied to input
        # url ...". Solo se disparaba con audio preparado + fuente extra
        # (subtítulos o pistas adicionales), y el fallo llegaba al final del job.
        next_input_index = 1
        audio_input_index: int | None = None
        if audio_mux_path is not None:
            cmd += ["-i", str(audio_mux_path)]
            audio_input_index = next_input_index
            next_input_index += 1
        source_input_index = self._maybe_add_source_input(cmd, job, next_input_index)

        maps: list[str] = []
        if audio_input_index is not None:
            maps += ["-map", "0:v:0", "-map", f"{audio_input_index}:a:0"]
        if source_input_index is not None:
            # Any -map switches ffmpeg to explicit-map mode and drops every
            # unmapped stream -- including the frames video. When audio_mux_path
            # was present the video map was already emitted above; when it was
            # not (keep_audio=False + keep_subtitles, or no usable source audio)
            # it must be emitted here or the output loses its video track.
            if audio_input_index is None:
                maps += ["-map", "0:v:0"]
            self._map_extra_audio_tracks(maps, job, source_input_index)
            self._map_subtitles(maps, job, source_input_index)
        cmd += maps
        cmd += self._resize_filter_args(job)
        cmd += self._build_video_encode_options(job, encoder)
        if audio_mux_path is not None:
            cmd += audio_codec_args
            cmd += self._extra_audio_copy_args(job)
        if job.keep_subtitles:
            cmd += ["-c:s", "copy"]
        cmd.append(str(output_path))
        return cmd

    def _extra_audio_copy_args(self, job: VideoUpscaleJob) -> list[str]:
        # audio_codec_args uses the unindexed "-c:a" specifier, which applies
        # to EVERY output audio stream. The extras were mapped raw from the
        # source and must be copied verbatim (enhance/restore only ever touches
        # the primary). A per-output-position "-c:a:<pos> copy" overrides the
        # general codec for that stream; ffmpeg resolves conflicting codec
        # specifiers as "last match wins", so these MUST come after
        # audio_codec_args. Output audio 0 is the primary (from audio_mux_path);
        # the extras occupy positions 1..N in the same order they were mapped.
        args: list[str] = []
        for position, _ in enumerate(self._extra_audio_track_indices(job), start=1):
            args += [f"-c:a:{position}", "copy"]
        return args

    def _primary_audio_track_index(self, job: VideoUpscaleJob) -> int | None:
        # The PRIMARY track is the first of the selected list; it is extracted
        # (and enhanced/restored/copied) into audio_mux_path separately, and is
        # mapped into the extraction ffmpeg command by this ABSOLUTE index.
        if not job.audio_track_indices:
            return None
        return job.audio_track_indices[0]

    def _primary_audio_map_args(self, job: VideoUpscaleJob) -> list[str]:
        # Absolute ffprobe stream index (e.g. "0:3"), never the relative
        # "0:a:N" form -- audio_track_indices stores absolute stream indices,
        # so an audio-relative specifier would select the wrong stream.
        index = self._primary_audio_track_index(job)
        if index is None:
            return []
        return ["-map", f"0:{index}"]

    def _extra_audio_track_indices(self, job: VideoUpscaleJob) -> list[int]:
        # The PRIMARY track (audio_track_indices[0]) is muxed separately via
        # audio_mux_path (extracted by absolute index in the extraction step).
        # The extras are every OTHER selected track; a repeat of the primary
        # later in the list is dropped so the same stream is never mapped twice.
        # keep_audio=False must silently drop ALL audio, extras included --
        # otherwise _needs_source_input still fires and the extras get mapped
        # with no codec args (audio_mux_path is None), so ffmpeg re-encodes
        # them into the output despite the user turning audio off.
        if not job.keep_audio:
            return []
        if not job.audio_track_indices or len(job.audio_track_indices) <= 1:
            return []
        primary = job.audio_track_indices[0]
        return [index for index in job.audio_track_indices[1:] if index != primary]

    def _ensure_disk_room_for_png_path(
        self, job: VideoUpscaleJob, work_dir: Path, fps_multiplier: int, stream_mode: str | None
    ) -> None:
        """Falla ANTES de extraer si el camino PNG no entra en el disco.

        Sin esto un 4x de dos horas descubre que le faltan terabytes recién a
        mitad del upscale, y el síntoma que ve el usuario es un error de ffmpeg
        sin relación aparente (frames truncados) horas después de arrancar.
        """
        frames_total = job.metadata.get("framesTotal")
        width = int(job.metadata.get("sourceWidth") or 0)
        height = int(job.metadata.get("sourceHeight") or 0)
        # Sin conteo honesto (VFR) o sin dimensiones no hay estimación honesta.
        if not isinstance(frames_total, int) or isinstance(frames_total, bool) or frames_total <= 0:
            return
        if width <= 0 or height <= 0:
            return

        interpolating = self._interpolation_requested(fps_multiplier, job.target_fps)
        output_frame_count = frames_total * fps_multiplier if interpolating else frames_total
        needed = estimate_png_workdir_bytes(
            source_width=width,
            source_height=height,
            scale=job.scale,
            source_frame_count=frames_total,
            output_frame_count=output_frame_count,
            # El híbrido streamea el upscale: nunca materializa el set grande.
            # El clásico tampoco: no corre ningún motor, así que los frames de entrada
            # SON los del encode y no se escribe un segundo set. Sin esta condición el
            # guard le pedía a un 4K clásico el disco de un 4x que nunca va a ocurrir
            # (scale² veces más) y rechazaba justo el camino barato.
            writes_upscaled_frames=stream_mode is None and not is_classic_upscaler(job.model_id),
            writes_interpolated_frames=interpolating,
        )
        usable = int(shutil.disk_usage(work_dir).free * _DISK_HEADROOM_FRACTION)
        if needed <= usable:
            return
        needed_gb = needed / 1024**3
        usable_gb = usable / 1024**3
        raise RuntimeError(
            f"No hay espacio suficiente para procesar este video: el camino con frames PNG "
            f"necesita ~{needed_gb:,.0f} GB y hay ~{usable_gb:,.0f} GB utilizables. "
            f"Probá con una escala menor, un recorte más corto, o liberá espacio en disco."
        )

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

    async def _resolve_stream_pipeline_mode(
        self, job: VideoUpscaleJob, fps_multiplier: int, frames_total_exact: bool = False
    ) -> str | None:
        """Elige el camino del stream pipeline. Requiere metadata del probe ya
        estampada (sourceWidth/sourceHeight/framesTotal). None = camino clásico.
        """
        if not self.settings.enable_stream_pipeline or self.onnx_video_engine is None:
            return None
        # Modelos HF-ONNX van por OnnxUpscaler (grafos fp32 arbitrarios), no por
        # el motor builtin de streaming — misma exclusión que el raw-pipe.
        if self._is_onnx_model(job.model_id):
            return None
        # Las pistas de audio extra y los subtítulos YA NO excluyen del
        # streaming: _build_rawpipe_command agrega el source como input
        # adicional y los mapea igual que el camino PNG. Antes esos jobs
        # (justamente los que más se benefician) quedaban condenados a
        # materializar cientos de GB de frames intermedios.
        out_pixels = (
            int(job.metadata.get("sourceWidth") or 0)
            * int(job.metadata.get("sourceHeight") or 0)
            * job.scale
            * job.scale
        )
        # Bajo el umbral el encode PNG es barato y el overhead no compensa —
        # mismo criterio (y mismo setting) que el raw-pipe.
        if out_pixels < self.settings.raw_pipe_min_output_pixels:
            return None
        # Off the loop: el primer resolve puede hacer un import frío de
        # onnxruntime + get_available_providers (mismo motivo que _should_stream).
        if await asyncio.to_thread(self._resolve_builtin_backend, job) != UpscaleBackend.onnx:
            return None
        if not self._interpolation_requested(fps_multiplier, job.target_fps):
            return STREAM_MODE_FULL
        if job.interp_engine == GMFSS_ENGINE:
            # El plan GMFSS necesita el conteo fuente exacto: sin framesTotal
            # honesto (VFR) no hay plan — clásico.
            #
            # PENDIENTE (decisión de producto, ver review final de la rama): con
            # un framesTotal ESTIMADO (round(duration*fps), cuando el contenedor
            # no trae nb_frames) el plan puede quedar corrido y el desajuste solo
            # se detecta en el flush() de la etapa, con todo el trabajo ya hecho.
            # Exigir `frames_total_exact` acá lo evitaría, a costa de mandar al
            # camino clásico a toda fuente sin nb_frames. No se aplica todavía
            # porque angosta el ruteo de forma visible para el usuario.
            frames_total = job.metadata.get("framesTotal")
            if (
                self.gmfss_engine is None
                or not isinstance(frames_total, int)
                or isinstance(frames_total, bool)
                or frames_total <= 0
            ):
                return None
            return STREAM_MODE_FULL
        # RIFE: el binario exige el directorio PNG entero — híbrido (su tramo
        # queda en PNG; upscale→encode va en streaming).
        return STREAM_MODE_HYBRID

    async def _run_stream_pipeline(
        self,
        job: VideoUpscaleJob,
        *,
        source_factory: "Callable[[threading.Event], Iterator[np.ndarray]]",
        gmfss_stage_factory: "Callable[[str], FrameStage] | None",
        width: int,
        height: int,
        expected_output_count: int | None,
        output_path: Path,
        encode_fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        encoder_name: str,
    ) -> None:
        """Corre el FramePipeline completo en un worker thread con el patrón
        shield+cancel_event+watchdog del raw-pipe: el cancel señala y ESPERA a
        que el worker desenrolle (threads joineados, procesos ffmpeg muertos)
        antes de propagar, para que la limpieza del work-dir no corra contra
        escrituras vivas."""
        out_w, out_h = width * job.scale, height * job.scale
        command = self._build_rawpipe_command(
            out_w, out_h, encode_fps, audio_mux_path, audio_codec_args, output_path, job, encoder_name
        )
        counter = {"n": 0}
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_stream_pipeline_blocking,
                job,
                source_factory,
                gmfss_stage_factory,
                job.device or self.settings.default_device,
                width,
                height,
                expected_output_count,
                command,
                counter,
                cancel_event,
            )
        )
        try:
            async with self._track_streaming_progress(job, counter):
                await asyncio.shield(worker)
        except BaseException:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise
        # Un exit 0 de ffmpeg no garantiza un archivo utilizable. Validarlo acá
        # (y no recién en _finalize_output, que corre FUERA del try del caller)
        # es lo que hace que ese caso caiga al camino clásico en vez de fallar
        # el job con streamPipeline=true ya estampado.
        if not is_non_empty_file(output_path):
            raise RuntimeError("el stream pipeline terminó sin dejar un archivo de salida utilizable")
        if expected_output_count is None:
            # El denominador era un estimado (o no había): publicar lo realmente
            # entregado para que el stepper cierre en N/N y no en N/estimado.
            job.metadata["framesTotal"] = counter["n"]

    def _run_stream_pipeline_blocking(
        self,
        job: VideoUpscaleJob,
        source_factory: "Callable[[threading.Event], Iterator[np.ndarray]]",
        gmfss_stage_factory: "Callable[[str], FrameStage] | None",
        device: str,
        width: int,
        height: int,
        expected_output_count: int | None,
        command: list[str],
        counter: dict[str, int],
        cancel_event: threading.Event,
    ) -> None:
        stages: list[FrameStage] = []
        if gmfss_stage_factory is not None:
            # GMFSS primero, upscale después: el acquire del upscaler evicta la
            # ENTRADA de cache de GMFSS pero el driver ya retiene sus sesiones.
            # Ambos sets quedan residentes en VRAM durante el run — mismo
            # trade-off que documentaba la fusión eliminada; vigilarlo en el
            # smoke real. La serialización sigue en el coordinator/semáforos.
            stages.append(gmfss_stage_factory(device))
        input_bytes = width * height * 3
        output_bytes = input_bytes * job.scale * job.scale
        budget_bytes = max(1, self.settings.onnx_video_max_pipeline_mb) * 1024 * 1024
        # Las colas se derivan ANTES de armar la etapa de upscale: su anillo de
        # readback debe cubrir los frames en vuelo aguas abajo (última cola +
        # sink) o un buffer reusado corrompería un frame todavía encolado.
        maxsizes = derive_stream_queue_maxsizes(
            input_bytes, output_bytes, len(stages) + 1, budget_bytes
        )
        upscale_frame = self.onnx_video_engine.build_frame_upscaler(
            job.model_name,
            device,
            job.scale,
            readback_ring_capacity=derive_readback_ring_capacity(maxsizes[-1], 1),
        )
        stages.append(MapStage(upscale_frame))

        encoder = RawPipeEncoder(
            command, summarize_error=lambda stderr: self._summarize_process_error(stderr, b"")
        )
        encoder.start()

        def sink(frame_nhwc: np.ndarray) -> None:
            encoder.write_frame(frame_nhwc[0])  # NHWC -> HWC; bloquea con backpressure del pipe
            counter["n"] = encoder.frames_written

        pipeline = FramePipeline(source_factory(cancel_event), stages, sink, maxsizes)
        try:
            delivered = pipeline.run(cancel_event)
        except BaseException:
            encoder.kill()
            raise
        if cancel_event.is_set():
            encoder.kill()
            return
        if delivered == 0:
            encoder.kill()
            raise RuntimeError("el stream pipeline no entregó ningún frame")
        if expected_output_count is not None and delivered != expected_output_count:
            encoder.kill()
            raise RuntimeError(
                f"el stream pipeline entregó {delivered}/{expected_output_count} frames"
            )
        encoder.finish()

    async def _try_stream_pipeline_full(
        self,
        job: VideoUpscaleJob,
        fps_multiplier: int,
        output_path: Path,
        fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        frames_total_exact: bool = False,
    ) -> bool:
        """Pipeline completo decode→(GMFSS)→upscale→encode, sin ningún PNG.
        True = output_path finalizado; False = fallback clásico DESDE CERO
        (cancel/stall SÍ propagan). fps_multiplier lo consume la rama de
        interpolación (Task 11)."""
        width = int(job.metadata.get("sourceWidth") or 0)
        height = int(job.metadata.get("sourceHeight") or 0)
        if width <= 0 or height <= 0:
            return False
        had_frames_total = "framesTotal" in job.metadata
        original_frames_total = job.metadata.get("framesTotal")
        try:
            probe_w, probe_h = self._expected_output_dims(job)
            encoder_name = await asyncio.to_thread(
                self._resolve_video_encoder, job, probe_w, probe_h
            )
            job.metadata["videoEncoder"] = encoder_name
            self._stamp_upscale_diagnostics(job)
            # Sin interp la salida iguala a la fuente, pero solo se valida por
            # igualdad si el conteo es EXACTO (nb_frames). Un estimado
            # round(duration*fps) difiere +-1 seguido, y rechazar por eso
            # tiraría a la basura un decode+upscale+encode ya completo para
            # rehacerlo por el clásico: se deja pasar y se publica lo entregado.
            expected_output_count = (
                job.metadata.get("framesTotal") if frames_total_exact else None
            )
            encode_fps = fps
            gmfss_stage_factory = None
            if self._interpolation_requested(fps_multiplier, job.target_fps):
                # El gate garantiza framesTotal conocido y gmfss_engine presente
                # para este camino (RIFE nunca llega acá: es modo híbrido).
                source_count = int(job.metadata["framesTotal"])
                expected_output_count = (
                    compute_target_frame_count(source_count, fps, job.target_fps)
                    if job.target_fps is not None
                    else source_count * fps_multiplier
                )
                encode_fps = self._interpolated_encode_fps(fps, fps_multiplier, job.target_fps)
                # Denominador honesto del stepper colapsado: la salida interpolada.
                job.metadata["framesTotal"] = expected_output_count
                gmfss_stage_factory = lambda device: self.gmfss_engine.build_stream_stage(  # noqa: E731
                    source_count, expected_output_count, device
                )
            # Etapa colapsada (decisión de stepper del plan): extract/interp
            # quedan "done" al avanzar; framesDone lo cuenta el sink de encode.
            advance_video_stage(job, "upscaling_frames")
            source = FfmpegFrameSource(
                self.settings.ffmpeg_binary_path,
                job.source_path,
                width,
                height,
                self.settings.ffmpeg_decode_threads,
                fps,
            )
            await self._run_stream_pipeline(
                job,
                source_factory=source.frames,
                gmfss_stage_factory=gmfss_stage_factory,
                width=width,
                height=height,
                expected_output_count=expected_output_count,
                output_path=output_path,
                encode_fps=encode_fps,
                audio_mux_path=audio_mux_path,
                audio_codec_args=audio_codec_args,
                encoder_name=encoder_name,
            )
        except (asyncio.CancelledError, VideoStallError):
            raise
        except Exception as exc:  # noqa: BLE001 - CUALQUIER fallo -> clásico desde cero
            logger.warning(
                "stream pipeline (completo) falló (%s); se usa el camino clásico desde cero", exc
            )
            if had_frames_total:
                job.metadata["framesTotal"] = original_frames_total
            else:
                job.metadata.pop("framesTotal", None)
            job.metadata["streamPipelineFallback"] = str(exc)
            output_path.unlink(missing_ok=True)
            return False
        job.metadata["streamPipeline"] = True
        job.metadata["outputFps"] = encode_fps
        return True

    @staticmethod
    def _probe_png_dir(frames_dir: Path) -> tuple[int, int, int]:
        paths = sorted(frames_dir.glob("*.png"))
        if not paths:
            raise RuntimeError("no hay frames para streamear")
        first = load_frame(paths[0])
        _, height, width, _ = first.shape
        return len(paths), width, height

    async def _try_stream_pipeline_from_dir(
        self,
        job: VideoUpscaleJob,
        frames_dir: Path,
        output_path: Path,
        encode_fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
    ) -> bool:
        """Tramo streaming del modo híbrido (RIFE): lee los PNGs interpolados y
        streamea upscale→encode. True = output_path finalizado por el pipeline;
        False = fallback al camino PNG clásico (cancel/stall SÍ propagan)."""
        had_frames_total = "framesTotal" in job.metadata
        original_frames_total = job.metadata.get("framesTotal")
        try:
            frame_count, width, height = await asyncio.to_thread(self._probe_png_dir, frames_dir)
            probe_w, probe_h = self._expected_output_dims(job)
            encoder_name = await asyncio.to_thread(
                self._resolve_video_encoder, job, probe_w, probe_h
            )
            job.metadata["videoEncoder"] = encoder_name
            self._stamp_upscale_diagnostics(job)
            # Etapa colapsada (decisión de stepper del plan): denominador
            # honesto = frames reales del tramo (ya interpolados).
            job.metadata["framesTotal"] = frame_count
            advance_video_stage(job, "upscaling_frames")
            await self._run_stream_pipeline(
                job,
                source_factory=lambda cancel_event: iter_png_frames(frames_dir, cancel_event),
                gmfss_stage_factory=None,
                width=width,
                height=height,
                expected_output_count=frame_count,
                output_path=output_path,
                encode_fps=encode_fps,
                audio_mux_path=audio_mux_path,
                audio_codec_args=audio_codec_args,
                encoder_name=encoder_name,
            )
        except (asyncio.CancelledError, VideoStallError):
            raise
        except Exception as exc:  # noqa: BLE001 - CUALQUIER fallo -> camino clásico
            logger.warning("stream pipeline (híbrido) falló (%s); se usa el camino PNG clásico", exc)
            if had_frames_total:
                job.metadata["framesTotal"] = original_frames_total
            else:
                job.metadata.pop("framesTotal", None)
            job.metadata["streamPipelineFallback"] = str(exc)
            output_path.unlink(missing_ok=True)
            return False
        job.metadata["streamPipeline"] = True
        job.metadata["outputFps"] = encode_fps
        return True

    async def _should_stream(self, job: VideoUpscaleJob) -> bool:
        if not self.settings.enable_raw_pipe or self.onnx_video_engine is None:
            return False
        # HF-installed ONNX models use OnnxUpscaler (arbitrary graphs), not the
        # builtin streaming engine, so they stay on the file path.
        if self._is_onnx_model(job.model_id):
            return False
        # Extra audio tracks / subtitles no longer force the PNG path:
        # _build_rawpipe_command adds the source as an extra input and maps them
        # the same way _build_encode_command does.
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
        try:
            out_w, out_h = await asyncio.to_thread(self._output_dims, frames_in, job.scale)
            encoder = await asyncio.to_thread(self._resolve_video_encoder, job, out_w, out_h)
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
        # Upscaled frames are RGB HWC uint8 (see OnnxVideoUpscaler / frame_workers.load_frame),
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
        # Mismo esquema que _build_encode_command: TODOS los -i primero y los
        # -map después (en ffmpeg -map es opción de salida). El archivo fuente
        # entra como input adicional para conservar pistas de audio extra y
        # subtítulos SIN caer al camino PNG, que a 4x materializa cientos de GB
        # de frames intermedios solo para poder mapearlos.
        next_input_index = 1
        audio_input_index: int | None = None
        if audio_mux_path is not None:
            cmd += ["-i", str(audio_mux_path)]
            audio_input_index = next_input_index
            next_input_index += 1
        source_input_index = self._maybe_add_source_input(cmd, job, next_input_index)

        maps: list[str] = []
        if audio_input_index is not None:
            maps += ["-map", "0:v:0", "-map", f"{audio_input_index}:a:0"]
        if source_input_index is not None:
            if audio_input_index is None:
                maps += ["-map", "0:v:0"]
            self._map_extra_audio_tracks(maps, job, source_input_index)
            self._map_subtitles(maps, job, source_input_index)
        cmd += maps
        # Mismo ajuste al objetivo que el camino PNG. Sin esto los tres caminos de
        # streaming ignoraban target_height por completo: el archivo salia en
        # fuente*escala mientras _final_output_dims reportaba el objetivo, o sea que la
        # metadata, la API y la vista previa del frontend mentian sobre el resultado.
        # Los frames que entran por el pipe ya vienen en fuente*escala, que es
        # exactamente lo que _resize_filter_args toma como punto de partida.
        cmd += self._resize_filter_args(job)

        cmd += self._build_video_encode_options(job, encoder)
        if audio_mux_path is not None:
            cmd += audio_codec_args
            cmd += self._extra_audio_copy_args(job)
        if job.keep_subtitles:
            cmd += ["-c:s", "copy"]
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
        # Mismo resumen de errores que el encode PNG (mensajes x265 amigables, etc.).
        pipe_encoder = RawPipeEncoder(cmd, summarize_error=lambda stderr: self._summarize_process_error(stderr, b""))
        pipe_encoder.start()

        counter = {"n": 0}

        def write_frame(frame_hwc_rgb) -> None:
            pipe_encoder.write_frame(frame_hwc_rgb)  # blocks on pipe backpressure (worker thread)
            counter["n"] = pipe_encoder.frames_written

        device = job.device or self.settings.default_device
        # Shield the engine task so a job cancel doesn't tear it down while a worker
        # thread is blocked writing to ffmpeg's pipe. We kill ffmpeg FIRST (which
        # unblocks that write with BrokenPipe so the engine unwinds), then await it.
        stream_task = asyncio.ensure_future(
            self.onnx_video_engine.run_frames_streaming(
                frames_in, job.model_name, device, write_frame, job.scale
            )
        )
        try:
            async with self._track_streaming_progress(job, counter):
                expected = await asyncio.shield(stream_task)
            if counter["n"] != expected:
                raise RuntimeError(f"raw-pipe wrote {counter['n']}/{expected} frames")
            await asyncio.to_thread(pipe_encoder.finish)
        except BaseException:
            pipe_encoder.kill()
            with contextlib.suppress(BaseException):
                await stream_task
            raise

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
            # Publicar el contador final: un clip que termina antes del primer
            # tick del poller quedaba en framesDone=0 pese a haber entregado
            # todos sus frames (mismo cierre que hace _track_frame_progress).
            self._apply_frame_progress(job, "upscaling_frames", counter["n"], frames_total)

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
        if not lines:
            return "External process failed"
        # El último renglón de ffmpeg suele NO ser el diagnóstico: en un fallo de
        # input imprime "Error opening input file <RUTA>." y recién después el
        # resumen sin ruta, y en un fallo de encode las estadísticas del codec
        # van DESPUÉS del error (lines[-1] devolvía cosas como "kb/s:52.80").
        # Se conservan los últimos renglones que sí traen señal de error.
        error_lines = [line for line in lines if "error" in line.lower()]
        if not error_lines:
            return lines[-1]
        deduped = list(dict.fromkeys(error_lines[-_MAX_ERROR_CONTEXT_LINES:]))
        return " | ".join(deduped)
