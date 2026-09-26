"""Carril IA de CCTV (tarea "enhance", spec §4.7): gate propio, siempre por stream.

Decode con los pasos pre-IA -> etapa compuesta (DRUNet, escalador ONNX y OSD) en un
solo FramePipeline -> encode por pipe con los pasos post-IA y la banda del rotulo.
Nunca pasa por el gate del reescalado ni por el camino PNG. Lo que comparte con el
pipeline de video (elegir encoder, armar el raw-pipe, progreso con watchdog) llega
por EnhanceVideoTools, que arma VideoUpscaler.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import MODEL_CATALOG, Settings
from app.core.version import get_app_version
from app.models import VideoUpscaleJob, utc_now
from app.services.cctv_ai_models import catalog_generative
from app.services.cctv_chain import AiLanePlan, ResolvedStep, ai_lane_plan
from app.services.cctv_clarify_runner import ClarifyTools, StageProgress
from app.services.cctv_enhance_finish import (
    EnhanceContext,
    EnhanceFinishTools,
    EnhanceOutputs,
    finish_enhance,
    finished_json,
    with_finished,
)
from app.services.cctv_enhance_outputs import (
    AUDIO_LABEL,
    EnhanceLabel,
    EnhanceRuns,
    ModelFile,
    labeled_size,
    onnx_runtime_info,
    process_run,
    restore_model_file,
    stream_runs,
    upscale_json,
    upscaler_model_file,
    write_enhance_label,
)
from app.services.cctv_enhance_plan import (
    AI_LANE,
    EnhancePlan,
    audio_trim,
    build_audio_command,
    build_enhance_plan,
    enhance_metadata,
    enhance_source,
    enhanced_output_path,
    source_json,
    stream_json,
)
from app.services.cctv_ingest import MediaTools as CctvMediaTools
from app.services.cctv_job_runner import IngestedSource, ingest_job_source
from app.services.cctv_job_validation import resolve_steps
from app.services.cctv_report import HostFacts, ProcessRun, host_facts
from app.services.cctv_report_model import GpuInfo
from app.services.cctv_session import cctv_job_dir
from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder
from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource
from app.services.engines.frame_restorer import (
    ComposedStage,
    ComposedStageReport,
    FrameRestorer,
    UpscalerFactory,
    build_composed_stage,
)
from app.services.engines.frame_workers import derive_readback_ring_capacity
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.ffmpeg_capabilities import cached_capabilities
from app.services.frame_pipeline import FramePipeline, derive_stream_queue_maxsizes
from app.services.handover_package import PROCESSED_DIRNAME, place_file
from app.services.process_runner import is_non_empty_file
from app.services.progress import advance_video_stage, complete_video_stages

CCTV_ENHANCE_TASK = "enhance"
CCTV_ENHANCE_STAGE = "restoring_frames"
CCTV_AUDIO_NAME = "audio.m4a"
CCTV_LABEL_DIRNAME = "label"

EncoderResolver = Callable[[VideoUpscaleJob, int, int], str]
ProgressTracker = Callable[[VideoUpscaleJob, dict[str, int], str], AbstractAsyncContextManager[None]]


@dataclass(frozen=True, slots=True)
class EnhanceVideoTools:
    resolve_encoder: EncoderResolver
    rawpipe_command: Callable[..., list[str]]
    track_progress: ProgressTracker
    run_process: Callable[[list[str]], Awaitable[bytes]]
    summarize_error: Callable[[bytes, bytes], str]
    usable_audio: Callable[[Path], Path | None]
    gpu_info: Callable[[str | None], tuple[GpuInfo, ...]]


def unchanged_frame(frame: np.ndarray) -> np.ndarray:
    return frame


def prepend_frame(first: np.ndarray, rest: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    # Generador (no itertools.chain): al cerrarlo el pipeline tambien cierra el decode y mata su ffmpeg.
    yield first
    yield from rest


def close_quietly(frames: Iterator[np.ndarray]) -> None:
    # Si un thread del pipeline no llego a soltar el generador, cerrarlo tira ValueError y taparia el error real.
    with contextlib.suppress(ValueError):
        frames.close()


def counting_sink(encoder: RawPipeEncoder, counter: dict[str, int]) -> Callable[[np.ndarray], None]:
    def sink(frame_nhwc: np.ndarray) -> None:
        encoder.write_frame(frame_nhwc[0])
        counter["n"] = encoder.frames_written

    return sink


def finish_encoded_pipeline(
    pipeline: FramePipeline, encoder: RawPipeEncoder, cancel_event: threading.Event
) -> int | None:
    try:
        delivered = pipeline.run(cancel_event)
    except BaseException:
        encoder.kill()
        raise
    if cancel_event.is_set():
        encoder.kill()
        return None
    if delivered == 0:
        encoder.kill()
        raise RuntimeError("The enhancement stream delivered no frames.")
    encoder.finish()
    return delivered


def no_frames_to_enhance(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        return None
    raise RuntimeError("The working copy produced no frames to enhance.")


class CctvEnhanceRunner:
    def __init__(
        self,
        settings: Settings,
        frame_restorer: FrameRestorer,
        onnx_video_engine: OnnxVideoUpscaler | None,
        tools: EnhanceVideoTools,
    ) -> None:
        self.settings = settings
        self.frame_restorer = frame_restorer
        self.onnx_video_engine = onnx_video_engine
        self.tools = tools

    async def run(self, job: VideoUpscaleJob, work_dir: Path, on_stage: StageProgress) -> Path:
        lane = ai_lane_plan(resolve_steps(job.cctv, AI_LANE))
        job_dir = cctv_job_dir(self.settings.outputs_path, job.id)
        source = await ingest_job_source(self._media_tools(), job_dir, work_dir, on_stage)
        plan = build_enhance_plan(lane, enhance_source(source), job.cctv.osd_boxes, job.scale)
        label = await asyncio.to_thread(
            write_enhance_label, work_dir / CCTV_LABEL_DIRNAME, plan, self._app_version(), job.id
        )
        # Se encodea en video-work y se mueve al final: un job que falla no deja un video a medias en outputs.
        encoded = enhanced_output_path(work_dir, job.output_container)
        audio_mux_path, audio_codec_args, audio_run = await self._audio(job, source, lane.decode, work_dir)
        command = await self._encode_command(job, plan, label, encoded, audio_mux_path, audio_codec_args)
        frame_source = self._frame_source(plan, source.work)
        started = utc_now()
        report = await self._run_stream(job, plan, frame_source, command, encoded)
        frames = (plan.frames_in, job.metadata["framesTotal"])
        runs = stream_runs(frame_source.build_command(), command, (started, utc_now()), frames, audio_run)
        output_path = place_file(encoded, enhanced_output_path(job_dir / PROCESSED_DIRNAME, job.output_container))
        context = self._context(job, lane, plan, source, output_path, label, runs, report)
        outputs = await finish_enhance(self._finish_tools(job.device), context, on_stage)
        self._stamp(job, context, report, outputs)
        return output_path

    def _app_version(self) -> str:
        return get_app_version(self.settings.update_package_name)

    def _media_tools(self) -> CctvMediaTools:
        return CctvMediaTools(self.settings.ffmpeg_binary_path, self.settings.ffprobe_binary_path)

    async def _audio(
        self, job: VideoUpscaleJob, source: IngestedSource, decode: tuple[ResolvedStep, ...], work_dir: Path
    ) -> tuple[Path | None, list[str], ProcessRun | None]:
        if not job.keep_audio or not source.ingest.audio:
            return None, [], None
        audio_path = work_dir / CCTV_AUDIO_NAME
        atrim = audio_trim(decode, source.frames)
        command = build_audio_command(self.settings.ffmpeg_binary_path, source.work, atrim, audio_path)
        started = utc_now()
        await self.tools.run_process(command)
        run = process_run(AUDIO_LABEL, command, (started, utc_now()), (None, None))
        return self.tools.usable_audio(audio_path), ["-c:a", "copy"], run

    async def _encode_command(
        self,
        job: VideoUpscaleJob,
        plan: EnhancePlan,
        label: EnhanceLabel,
        output_path: Path,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
    ) -> list[str]:
        # El encoder se elige para el cuadro final: los pasos post-IA y la banda cambian el tamaño.
        encoder = await asyncio.to_thread(self.tools.resolve_encoder, job, *labeled_size(plan, label))
        job.metadata["videoEncoder"] = encoder
        width, height = plan.output_size
        return self.tools.rawpipe_command(
            width, height, plan.rate_text, audio_mux_path, audio_codec_args, output_path, job, encoder,
            video_filter_args=label.args,
        )  # fmt: skip

    def _frame_source(self, plan: EnhancePlan, work: Path) -> FfmpegFrameSource:
        return FfmpegFrameSource(
            self.settings.ffmpeg_binary_path,
            work,
            plan.decoded.width,
            plan.decoded.height,
            self.settings.ffmpeg_decode_threads,
            plan.rate_text,
            prefilter_args=plan.prefilter_args,
        )

    async def _run_stream(
        self,
        job: VideoUpscaleJob,
        plan: EnhancePlan,
        source: FfmpegFrameSource,
        command: list[str],
        output_path: Path,
    ) -> ComposedStageReport:
        advance_video_stage(job, CCTV_ENHANCE_STAGE)
        job.metadata["framesTotal"] = plan.frames_in
        counter = {"n": 0}
        cancel_event = threading.Event()
        blocking = functools.partial(self._run_stream_blocking, job, plan, source, command, counter, cancel_event)
        report = await self._await_worker(job, counter, cancel_event, blocking)
        if report is None or not is_non_empty_file(output_path):
            raise RuntimeError("The enhanced video was not produced.")
        job.metadata["framesTotal"] = counter["n"]
        return report

    async def _await_worker(
        self,
        job: VideoUpscaleJob,
        counter: dict[str, int],
        cancel_event: threading.Event,
        blocking: Callable[[], ComposedStageReport | None],
    ) -> ComposedStageReport | None:
        # Mismo patron que el stream pipeline: el cancel senala y espera a que el worker mate sus procesos.
        worker = asyncio.ensure_future(asyncio.to_thread(blocking))
        try:
            async with self.tools.track_progress(job, counter, CCTV_ENHANCE_STAGE):
                return await asyncio.shield(worker)
        except BaseException:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

    def _run_stream_blocking(
        self,
        job: VideoUpscaleJob,
        plan: EnhancePlan,
        source: FfmpegFrameSource,
        command: list[str],
        counter: dict[str, int],
        cancel_event: threading.Event,
    ) -> ComposedStageReport | None:
        frames = source.frames(cancel_event)
        try:
            return self._pipe_frames(job, plan, frames, command, counter, cancel_event)
        finally:
            close_quietly(frames)

    def _pipe_frames(
        self,
        job: VideoUpscaleJob,
        plan: EnhancePlan,
        frames: Iterator[np.ndarray],
        command: list[str],
        counter: dict[str, int],
        cancel_event: threading.Event,
    ) -> ComposedStageReport | None:
        first = next(frames, None)
        if first is None:
            return no_frames_to_enhance(cancel_event)
        maxsizes = self._queue_sizes(plan)
        # El primer cuadro decodificado es la muestra del canario y fija la forma del job.
        stage = self._composed_stage(job, plan, first, maxsizes, cancel_event)
        encoder = RawPipeEncoder(command, summarize_error=lambda stderr: self.tools.summarize_error(stderr, b""))
        encoder.start()
        pipeline = FramePipeline(prepend_frame(first, frames), [stage], counting_sink(encoder, counter), maxsizes)
        delivered = finish_encoded_pipeline(pipeline, encoder, cancel_event)
        return None if delivered is None else stage.report()

    def _queue_sizes(self, plan: EnhancePlan) -> list[int]:
        input_bytes = plan.decoded.width * plan.decoded.height * 3
        budget_bytes = max(1, self.settings.onnx_video_max_pipeline_mb) * 1024 * 1024
        return derive_stream_queue_maxsizes(input_bytes, input_bytes * plan.upscale * plan.upscale, 1, budget_bytes)

    def _composed_stage(
        self,
        job: VideoUpscaleJob,
        plan: EnhancePlan,
        sample: np.ndarray,
        maxsizes: list[int],
        cancel_event: threading.Event,
    ) -> ComposedStage:
        device = job.device or self.settings.default_device
        upscaler = self._upscaler_factory(job, plan, device, maxsizes)
        if plan.strength is None:
            upscale = None if upscaler is None else upscaler()
            return build_composed_stage(unchanged_frame, upscale, osd_boxes=plan.osd_boxes)
        return self.frame_restorer.build_stage(
            device,
            sample,
            plan.strength,
            upscaler_factory=upscaler,
            osd_boxes=plan.osd_boxes,
            cancel_event=cancel_event,
        )

    def _upscaler_factory(
        self, job: VideoUpscaleJob, plan: EnhancePlan, device: str, maxsizes: list[int]
    ) -> UpscalerFactory | None:
        if plan.upscale == 1:
            return None
        if self.onnx_video_engine is None:
            raise RuntimeError("AI upscale needs the ONNX video engine.")
        ring = derive_readback_ring_capacity(maxsizes[-1], 1)
        engine = self.onnx_video_engine
        return lambda: engine.build_frame_upscaler(job.model_id, device, plan.upscale, readback_ring_capacity=ring)

    def _context(
        self,
        job: VideoUpscaleJob,
        lane: AiLanePlan,
        plan: EnhancePlan,
        source: IngestedSource,
        enhanced: Path,
        label: EnhanceLabel,
        runs: EnhanceRuns,
        report: ComposedStageReport,
    ) -> EnhanceContext:
        return EnhanceContext(
            job_id=job.id,
            options=job.cctv,
            lane=lane,
            plan=plan,
            source=source,
            job_dir=cctv_job_dir(self.settings.outputs_path, job.id),
            enhanced=enhanced,
            frames_out=job.metadata["framesTotal"],
            label=label,
            runs=runs,
            models={**self._restore_model(report), **self._upscaler_model(job, plan)},
            generative=plan.upscale > 1 and catalog_generative(MODEL_CATALOG, job.model_id),
            app_version=self._app_version(),
        )

    def _restore_model(self, report: ComposedStageReport) -> dict[str, ModelFile]:
        restore = report.restore
        if restore is None:
            return {}
        spec, path = self.frame_restorer.model_file(restore.model_id, restore.precision)
        return {"ai_deblock": restore_model_file(spec, path)}

    def _upscaler_model(self, job: VideoUpscaleJob, plan: EnhancePlan) -> dict[str, ModelFile]:
        if plan.upscale == 1:
            return {}
        precision = getattr(self.onnx_video_engine, "last_precision", None)
        onnx_dir = self.settings.builtin_onnx_path
        return {"ai_upscale": upscaler_model_file(job.model_id, plan.upscale, precision, onnx_dir, MODEL_CATALOG)}

    def _finish_tools(self, device: str | None) -> EnhanceFinishTools:
        ffmpeg, ffprobe = self.settings.ffmpeg_binary_path, self.settings.ffprobe_binary_path
        return EnhanceFinishTools(
            media=self._media_tools(),
            stills=ClarifyTools(ffmpeg, ffprobe),
            capabilities=functools.partial(cached_capabilities, ffmpeg),
            host=functools.partial(self._host, device),
        )

    def _host(self, device: str | None) -> HostFacts:
        return host_facts(self.tools.gpu_info(device), onnx_runtime_info())

    def _stamp(
        self, job: VideoUpscaleJob, context: EnhanceContext, report: ComposedStageReport, outputs: EnhanceOutputs
    ) -> None:
        plan, source = context.plan, context.source
        base = {**job.metadata.get("cctv", {}), "task": job.cctv.task, **source_json(source.record)}
        stream = stream_json(plan, report, job.metadata["framesTotal"])
        enhanced = context.enhanced.relative_to(context.job_dir).as_posix()
        metadata = enhance_metadata(base, stream, enhanced, source.ingest.warnings)
        upscale = upscale_json(job.model_id, plan.upscale, context.generative)
        job.metadata["cctv"] = with_finished(metadata, finished_json(context, outputs), upscale)
        job.metadata["streamPipeline"] = True
        job.metadata["outputFps"] = plan.rate_text
        job.metadata["outputWidth"], job.metadata["outputHeight"] = labeled_size(plan, context.label)
        complete_video_stages(job)


def enhance_runner_entry(
    settings: Settings,
    frame_restorer: FrameRestorer | None,
    onnx_video_engine: OnnxVideoUpscaler | None,
    tools: EnhanceVideoTools,
) -> dict[str, CctvEnhanceRunner]:
    # Sin la etapa compuesta de restauracion el carril IA no esta disponible: no hay runner.
    if frame_restorer is None:
        return {}
    return {CCTV_ENHANCE_TASK: CctvEnhanceRunner(settings, frame_restorer, onnx_video_engine, tools)}

