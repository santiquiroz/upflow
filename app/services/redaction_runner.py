"""Job "Redacted copy" de CCTV (P4-REDACT).

Reusa la ingesta del job (copia verificada re-hasheada, `work.mkv` e indice),
decodifica sin re-temporizar (el cuadro N es el N del indice, el mismo que se
eligio en la vista de cuadros), pixela o desenfoca las cajas de cada cuadro y
codifica una copia H.264 rotulada en `05_redacted/`, fuera de la lista de
carpetas del paquete de entrega. Deja `redaction.json` y `SHA256SUMS.txt`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.models import CctvOptions, RedactionRequest, VideoUpscaleJob
from app.services.cctv_chain import ResolvedStep
from app.services.cctv_clarify_runner import FrameCountMismatch, StageProgress, trim_step
from app.services.cctv_enhance_plan import source_rate
from app.services.cctv_ingest import FRAME_INDEX_NAME, ORIGINAL_DIRNAME, sha256_file
from app.services.cctv_job_runner import (
    CLASSIC_LANE,
    CctvRunnerConfig,
    IngestedSource,
    ingest_job_source,
    relative,
)
from app.services.cctv_job_validation import resolve_steps
from app.services.cctv_report import write_sha256sums
from app.services.cctv_session import cctv_job_dir
from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder
from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource
from app.services.ffmpeg_filters import FrameGeometry, build_filter
from app.services.handover_package import ascii_stem, check_job_id
from app.services.label_band import LabelAssets, label_band_args, write_label_assets
from app.services.redaction import (
    REDACTED_DIRNAME,
    REDACTED_EXTENSION,
    REDACTED_LABEL,
    REDACTED_TAG,
    REDACTION_LOG_NAME,
    RedactionLogFacts,
    redact_regions,
    redaction_log,
    regions_at,
)

STAGE_REDACTING = "redacting"
STAGE_REPORTING = "reporting"
LABEL_DIRNAME = "redact"
RENDER_NAME = "redacted.mp4"
X264_CRF = "18"
X264_PRESET = "medium"
DECODE_THREADS = 2
PROGRESS_STEPS = 100


@dataclass(frozen=True, slots=True)
class RedactionPlan:
    work: Path
    geometry: FrameGeometry
    first_frame: int
    last_frame: int
    rate: Fraction
    request: RedactionRequest
    prefilters: tuple[str, ...]

    @property
    def expected_frames(self) -> int:
        return self.last_frame - self.first_frame + 1


# --- Plan y comandos ---


def trim_range_of(steps: Sequence[ResolvedStep], frame_count: int) -> tuple[int, int]:
    trim = trim_step(steps)
    if trim is None:
        return 0, frame_count - 1
    return int(trim.params["start_frame"]), int(trim.params["end_frame"])


def decode_prefilters(steps: Sequence[ResolvedStep]) -> tuple[str, ...]:
    trim = trim_step(steps)
    return () if trim is None else (build_filter(trim),)


def rate_text(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


def redaction_plan(options: CctvOptions, steps: Sequence[ResolvedStep], source: IngestedSource) -> RedactionPlan:
    index = source.ingest.index
    first, last = trim_range_of(steps, index.summary.frame_count)
    rate = source_rate(index.summary, source.ingest.video.header_rate)
    return RedactionPlan(source.work, source.geometry, first, last, rate, options.redaction, decode_prefilters(steps))


class PassthroughFrameSource(FfmpegFrameSource):
    # Sin -fps_mode cfr: duplicar o descartar cuadros correria las cajas respecto del indice.
    def build_command(self) -> list[str]:
        vf = ["-vf", ",".join(self._prefilter_args)] if self._prefilter_args else []
        return [
            str(self._ffmpeg_binary), "-hide_banner", "-nostdin", "-v", "error",
            "-threads", str(self._decode_threads), "-i", str(self._source_path),
            "-map", "0:v:0", *vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]  # fmt: skip


def frame_source(ffmpeg: Path, plan: RedactionPlan) -> PassthroughFrameSource:
    geometry = plan.geometry
    return PassthroughFrameSource(
        ffmpeg, plan.work, geometry.width, geometry.height, DECODE_THREADS, rate_text(plan.rate), plan.prefilters
    )


def even_pad(geometry: FrameGeometry) -> tuple[str, ...]:
    width, height = geometry.width + geometry.width % 2, geometry.height + geometry.height % 2
    if (width, height) == (geometry.width, geometry.height):
        return ()
    return (f"pad=w={width}:h={height}:x=0:y=0:color=black",)


def encode_prefix(geometry: FrameGeometry) -> tuple[str, ...]:
    sar = geometry.sar
    return (*even_pad(geometry), f"setsar={sar.numerator}/{sar.denominator}")


@dataclass(frozen=True, slots=True)
class EncodeTarget:
    ffmpeg: Path
    output: Path
    x264_threads: int
    app_version: str
    job_id: str


def build_encode_command(target: EncodeTarget, plan: RedactionPlan, assets: LabelAssets) -> list[str]:
    geometry, threads = plan.geometry, target.x264_threads
    label = label_band_args(
        assets, target.app_version, target.job_id, prefix=encode_prefix(geometry), texts=REDACTED_LABEL
    )
    return [
        str(target.ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{geometry.width}x{geometry.height}",
        "-framerate", rate_text(plan.rate), "-i", "pipe:0", "-map", "0:v:0", *label,
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", X264_CRF, "-pix_fmt", "yuv420p",
        "-threads", str(threads), "-x264-params", f"threads={threads}:lookahead_threads=1",
        "-an", "-movflags", "+faststart", "-flags", "+bitexact", "-fflags", "+bitexact", str(target.output),
    ]  # fmt: skip


def redacted_name(original_name: str, job_id: str) -> str:
    return f"{ascii_stem(original_name)}{REDACTED_TAG}{check_job_id(job_id)}{REDACTED_EXTENSION}"


# --- Cuadros ---


class FrameWriter(Protocol):
    def start(self) -> None: ...

    def write_frame(self, frame_hwc: np.ndarray) -> None: ...

    def finish(self) -> None: ...

    def kill(self) -> None: ...


def progress_every(total: int) -> int:
    return max(1, total // PROGRESS_STEPS)


def close_quietly(frames: Iterator[np.ndarray]) -> None:
    with contextlib.suppress(ValueError):
        frames.close()


def write_redacted(
    frames: Iterator[np.ndarray],
    plan: RedactionPlan,
    encoder: FrameWriter,
    on_frame: Callable[[int], None],
) -> int:
    written = 0
    for offset, batch in enumerate(frames):
        frame = batch[0]
        encoder.write_frame(redact_regions(frame, regions_at(plan.request.tracks, plan.first_frame + offset), plan.request.style))
        written += 1
        on_frame(written)
    return written


def finish_writer(encoder: FrameWriter, cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        encoder.kill()
        return
    encoder.finish()


def render_redacted(
    frames: Iterator[np.ndarray],
    plan: RedactionPlan,
    encoder: FrameWriter,
    cancel_event: threading.Event,
    on_frame: Callable[[int], None],
) -> int:
    encoder.start()
    try:
        written = write_redacted(frames, plan, encoder, on_frame)
    except OSError:
        # El encoder murio (en Windows el pipe roto llega como EINVAL): finish trae su stderr, que explica mas.
        encoder.finish()
        raise
    except BaseException:
        encoder.kill()
        raise
    finally:
        close_quietly(frames)
    finish_writer(encoder, cancel_event)
    return written


def check_frame_count(plan: RedactionPlan, written: int) -> None:
    if written != plan.expected_frames:
        raise FrameCountMismatch(plan.expected_frames, written)


def stage_reporter(on_stage: StageProgress, total: int) -> Callable[[int], None]:
    every = progress_every(total)

    def report(written: int) -> None:
        if written % every == 0 or written == total:
            on_stage(STAGE_REDACTING, min(1.0, written / total))

    return report


# --- Salidas ---


def write_log(job_dir: Path, log: dict[str, Any]) -> Path:
    path = job_dir / REDACTION_LOG_NAME
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def original_files(job_dir: Path) -> list[str]:
    folder = job_dir / ORIGINAL_DIRNAME
    return sorted(path.relative_to(job_dir).as_posix() for path in folder.rglob("*") if path.is_file())


def checksum_paths(job_dir: Path, output: Path) -> tuple[str, ...]:
    return (*original_files(job_dir), output.relative_to(job_dir).as_posix(), FRAME_INDEX_NAME, REDACTION_LOG_NAME)


def log_facts(job: VideoUpscaleJob, source: IngestedSource, plan: RedactionPlan, frames: int, output: Path, version: str) -> RedactionLogFacts:
    job_dir = output.parent.parent
    return RedactionLogFacts(
        app_version=version,
        job_id=job.id,
        source_name=source.record.original_name,
        source_sha256=source.record.sha256,
        first_frame=plan.first_frame,
        last_frame=plan.last_frame,
        frames_out=frames,
        rate=rate_text(plan.rate),
        output_file=output.relative_to(job_dir).as_posix(),
        output_sha256=sha256_file(output),
    )


def redaction_metadata(job: VideoUpscaleJob, source: IngestedSource, plan: RedactionPlan, output: Path, job_dir: Path) -> dict[str, Any]:
    return {
        "task": job.cctv.task,
        "lane": CLASSIC_LANE,
        "sourceSha256": source.record.sha256,
        "receivedAt": {"utc": source.record.received_at.utc, "local": source.record.received_at.local},
        "framesIn": plan.expected_frames,
        "framesOut": plan.expected_frames,
        "outputs": {
            "redacted": relative(output, job_dir),
            "redaction": REDACTION_LOG_NAME,
            "package": None,
        },
        "redaction": {"style": plan.request.style, "boxes": len(plan.request.tracks)},
        "warnings": list(dict.fromkeys(source.ingest.warnings)),
    }


# --- Orquestacion ---


class RedactionRunner:
    def __init__(self, config: CctvRunnerConfig) -> None:
        self.config = config

    async def run(self, job: VideoUpscaleJob, work_dir: Path, on_stage: StageProgress) -> Path:
        steps = resolve_steps(job.cctv, CLASSIC_LANE)
        job_dir = cctv_job_dir(self.config.outputs_root, job.id)
        source = await ingest_job_source(self.config.media, job_dir, work_dir, on_stage)
        plan = redaction_plan(job.cctv, steps, source)
        output = job_dir / REDACTED_DIRNAME / redacted_name(source.record.original_name, job.id)
        frames = await self._render(job, plan, work_dir, output, on_stage)
        check_frame_count(plan, frames)
        await self._finish(job, source, plan, output, on_stage)
        job.metadata["cctv"] = redaction_metadata(job, source, plan, output, job_dir)
        return output

    async def _render(
        self, job: VideoUpscaleJob, plan: RedactionPlan, work_dir: Path, output: Path, on_stage: StageProgress
    ) -> int:
        on_stage(STAGE_REDACTING, 0.0)
        label_dir = work_dir / LABEL_DIRNAME
        label_dir.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        config, geometry = self.config, plan.geometry
        assets = await asyncio.to_thread(
            write_label_assets, label_dir, geometry.width, geometry.height, config.app_version, job.id,
            texts=REDACTED_LABEL,
        )  # fmt: skip
        target = EncodeTarget(config.clarify.ffmpeg, output, config.threads.x264_threads, config.app_version, job.id)
        encoder = RawPipeEncoder(build_encode_command(target, plan, assets))
        source = frame_source(config.clarify.ffmpeg, plan)
        return await self._run_blocking(source, plan, encoder, stage_reporter(on_stage, plan.expected_frames))

    async def _run_blocking(
        self,
        source: PassthroughFrameSource,
        plan: RedactionPlan,
        encoder: RawPipeEncoder,
        on_frame: Callable[[int], None],
    ) -> int:
        cancel_event = threading.Event()

        def blocking() -> int:
            return render_redacted(source.frames(cancel_event), plan, encoder, cancel_event, on_frame)

        # Mismo patron que el carril IA: el cancel senala y espera a que el worker mate sus procesos.
        worker = asyncio.ensure_future(asyncio.to_thread(blocking))
        try:
            return await asyncio.shield(worker)
        except BaseException:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

    async def _finish(
        self, job: VideoUpscaleJob, source: IngestedSource, plan: RedactionPlan, output: Path, on_stage: StageProgress
    ) -> None:
        on_stage(STAGE_REPORTING, 0.0)
        job_dir = output.parent.parent
        facts = await asyncio.to_thread(log_facts, job, source, plan, plan.expected_frames, output, self.config.app_version)
        await asyncio.to_thread(write_log, job_dir, redaction_log(plan.request, facts))
        await asyncio.to_thread(write_sha256sums, job_dir, checksum_paths(job_dir, output))
        on_stage(STAGE_REPORTING, 1.0)
