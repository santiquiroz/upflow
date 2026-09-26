"""Analisis de una sesion CCTV: ingesta hasta el indice + diagnostico (spec §4.2, §4.3, §5.5).

La sesion `video-work/cctv-{token}/` guarda el upload, `source.json`, `work.mkv`
y `frame_index.csv`; los jobs la reusan por token. El SHA-256 del upload se
calcula antes de cualquier otra operacion sobre el archivo (`ingest_source`).
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.services.cctv_chain import CctvChainError
from app.services.cctv_ingest import (
    IngestError,
    IngestTools,
    MediaTools,
    NoVideoStream,
    SourceRecord,
    WorkingIngest,
    ingest_source,
    ingest_working_copy,
    safe_original_name,
    source_facts,
    validate_frame_rate,
    write_source_record,
)
from app.services.cctv_job_validation import MODE_UNAVAILABLE
from app.services.cctv_presets import PresetContext, preset_steps
from app.services.cctv_session import UPLOAD_DIRNAME, session_dir
from app.services.ffmpeg_capabilities import (
    FILTER_UNAVAILABLE,
    FfmpegCapabilities,
    FfmpegProbeError,
    mode_unavailable_reason,
    unavailable_filters,
    unavailable_steps,
)
from app.services.video_analysis import CctvDiagnosis, VideoAnalysisError, analyze_video, is_interlaced

logger = logging.getLogger(__name__)

DEFAULT_UPLOAD_NAME = "upload.bin"
INVALID_UPLOAD = "cctv.error.invalidUpload"
UPLOAD_NOT_FOUND = "cctv.error.uploadNotFound"
INGEST_FAILED = "cctv.error.ingestFailed"
NO_VIDEO_STREAM = "cctv.error.noVideoStream"
ANALYSIS_FAILED = "cctv.error.analysisFailed"
FFMPEG_UNAVAILABLE = "cctv.error.ffmpegUnavailable"
LANES_PROPOSED = ("classic", "ai")

_UPLOAD_TOKEN = re.compile(r"[0-9a-f]{32}")

Diagnose = Callable[..., Awaitable[CctvDiagnosis]]


class CctvAnalysisError(RuntimeError):
    def __init__(self, key: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.key = key
        self.status = status


@dataclass(frozen=True)
class AnalysisTools:
    media: MediaTools
    capabilities: Callable[[], FfmpegCapabilities]
    ingest: IngestTools = IngestTools()
    diagnose: Diagnose = analyze_video


@dataclass(frozen=True)
class AnalysisRequest:
    token: str
    directory: Path
    upload: Path
    frame_rate: str | None = None


# --- Sesion y upload ---


def new_session(work_root: Path) -> tuple[str, Path]:
    token = uuid4().hex
    directory = session_dir(work_root, token)
    (directory / UPLOAD_DIRNAME).mkdir(parents=True)
    return token, directory


def upload_destination(directory: Path, filename: str | None) -> Path:
    try:
        name = safe_original_name(filename or DEFAULT_UPLOAD_NAME)
    except ValueError:
        name = DEFAULT_UPLOAD_NAME
    return directory / UPLOAD_DIRNAME / name


def staged_upload(uploads_root: Path, upload_token: str) -> Path:
    if not _UPLOAD_TOKEN.fullmatch(upload_token):
        raise CctvAnalysisError(UPLOAD_NOT_FOUND, "The upload token is not valid.")
    matches = sorted(path for path in uploads_root.glob(f"{upload_token}-*") if path.is_file())
    if len(matches) != 1:
        raise CctvAnalysisError(UPLOAD_NOT_FOUND, "The staged upload was not found or has expired; upload it again.")
    return matches[0]


def adopt_staged_upload(staged: Path, directory: Path, upload_token: str) -> Path:
    # Se mueve y no se copia: copiar leeria el archivo antes de hashearlo (spec §4.2 paso 1).
    destination = upload_destination(directory, staged.name[len(upload_token) + 1 :])
    shutil.move(str(staged), str(destination))
    return destination


def checked_frame_rate(raw: str | None) -> str | None:
    if raw is None or not raw.strip():
        return None
    try:
        return validate_frame_rate(raw)
    except ValueError as exc:
        raise CctvAnalysisError(INVALID_UPLOAD, str(exc)) from exc


def discard_session(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)


# --- Respuesta ---


def lite_sample_aspect(ingest: WorkingIngest) -> tuple[int, int] | None:
    if ingest.lite is None:
        return None
    return ingest.lite.sar.numerator, ingest.lite.sar.denominator


def proposed_steps(diagnosis: CctvDiagnosis, ingest: WorkingIngest) -> dict[str, list[dict[str, Any]]]:
    context = PresetContext(interlaced=is_interlaced(diagnosis.interlace), sample_aspect=lite_sample_aspect(ingest))
    return {lane: preset_steps(diagnosis.suggested_preset, lane, context) for lane in LANES_PROPOSED}


def capability_warnings(caps: FfmpegCapabilities) -> list[str]:
    checks = (
        (mode_unavailable_reason(caps) is not None, MODE_UNAVAILABLE),
        (bool(unavailable_filters(caps)), FILTER_UNAVAILABLE),
    )
    return [key for triggered, key in checks if triggered]


def analysis_warnings(ingest: WorkingIngest, diagnosis: CctvDiagnosis | None, caps: FfmpegCapabilities) -> list[str]:
    found = [*ingest.warnings, *(diagnosis.warnings if diagnosis else ()), *capability_warnings(caps)]
    return list(dict.fromkeys(found))


def video_json(ingest: WorkingIngest) -> dict[str, Any]:
    return {**ingest.video.to_json(), "lite": None if ingest.lite is None else ingest.lite.to_json()}


def working_copy_json(ingest: WorkingIngest) -> dict[str, Any]:
    return {**ingest.working_copy.to_json(), "decode": ingest.decode.to_json()}


def index_json(ingest: WorkingIngest) -> dict[str, Any] | None:
    return None if ingest.index is None else WorkingIngest.index_json(ingest.index)


def gop_json(ingest: WorkingIngest) -> dict[str, Any] | None:
    return None if ingest.index is None else ingest.index.summary.gop.to_json()


def diagnosis_parts(diagnosis: CctvDiagnosis | None, ingest: WorkingIngest) -> dict[str, Any]:
    if diagnosis is None:
        return {"quality": None, "suggestedPreset": None, "proposedSteps": None}
    return {
        "quality": diagnosis.model_dump(mode="json", by_alias=True, exclude={"suggested_preset", "warnings"}),
        "suggestedPreset": diagnosis.suggested_preset,
        "proposedSteps": proposed_steps(diagnosis, ingest),
    }


def capability_parts(caps: FfmpegCapabilities) -> dict[str, Any]:
    return {
        "modeAvailable": mode_unavailable_reason(caps) is None,
        "modeUnavailableReason": mode_unavailable_reason(caps),
        "unavailableSteps": list(unavailable_steps(caps)),
        "unavailableFilters": [item.to_json() for item in unavailable_filters(caps)],
    }


def analysis_json(
    token: str,
    record: SourceRecord,
    ingest: WorkingIngest,
    diagnosis: CctvDiagnosis | None,
    caps: FfmpegCapabilities,
) -> dict[str, Any]:
    return {
        "token": token,
        "sourceSha256": record.sha256,
        "receivedAt": {"utc": record.received_at.utc, "local": record.received_at.local},
        "originalName": record.original_name,
        "sizeBytes": record.size_bytes,
        "container": {"kind": record.container.kind, "label": record.container.label},
        "workingCopy": working_copy_json(ingest),
        "video": video_json(ingest),
        "audio": [track.to_json() for track in ingest.audio],
        "frameIndex": index_json(ingest),
        "gop": gop_json(ingest),
        "decodeFailed": ingest.decode.decode_failed,
        **diagnosis_parts(diagnosis, ingest),
        **capability_parts(caps),
        "warnings": analysis_warnings(ingest, diagnosis, caps),
    }


# --- Orquestacion ---


async def diagnose_if_decodable(tools: AnalysisTools, ingest: WorkingIngest) -> CctvDiagnosis | None:
    # Un archivo que no decodifica no paga el diagnostico: el job no va a seguir (spec §4.2 paso 6).
    if ingest.decode.decode_failed or ingest.index is None:
        return None
    media = tools.media
    return await tools.diagnose(media.ffmpeg, ingest.working_copy.path, source_facts(ingest), run=media.run)


async def analyze_session(request: AnalysisRequest, tools: AnalysisTools) -> dict[str, Any]:
    record = await asyncio.to_thread(ingest_source, request.upload, request.upload.name, tools.ingest)
    await asyncio.to_thread(write_source_record, request.directory, record)
    caps = await asyncio.to_thread(tools.capabilities)
    ingest = await ingest_working_copy(
        tools.media, request.upload, request.directory, record.container, request.frame_rate
    )
    diagnosis = await diagnose_if_decodable(tools, ingest)
    return analysis_json(request.token, record, ingest, diagnosis, caps)


def analysis_error(exc: Exception) -> CctvAnalysisError:
    if isinstance(exc, CctvAnalysisError):
        return exc
    if isinstance(exc, NoVideoStream):
        return CctvAnalysisError(NO_VIDEO_STREAM, "The file has no video stream.")
    if isinstance(exc, IngestError):
        return CctvAnalysisError(INGEST_FAILED, f"The file could not be read as video ({exc.step}).")
    if isinstance(exc, FfmpegProbeError):
        return CctvAnalysisError(FFMPEG_UNAVAILABLE, "ffmpeg is not available for CCTV mode.", 503)
    if isinstance(exc, (CctvChainError, ValueError)):
        return CctvAnalysisError(INVALID_UPLOAD, str(exc))
    if isinstance(exc, VideoAnalysisError):
        return CctvAnalysisError(ANALYSIS_FAILED, "The video diagnosis failed.", 500)
    logger.exception("Unexpected error while analyzing a CCTV upload", exc_info=exc)
    return CctvAnalysisError(ANALYSIS_FAILED, "The CCTV analysis failed.", 500)


async def run_session_analysis(request: AnalysisRequest, tools: AnalysisTools) -> dict[str, Any]:
    try:
        return await analyze_session(request, tools)
    except Exception as exc:
        # Una sesion a medio ingerir no sirve para ningun job: se borra con su upload.
        await asyncio.to_thread(discard_session, request.directory)
        raise analysis_error(exc) from exc


def analysis_tools(ffmpeg: Path, ffprobe: Path, capabilities: Callable[[], FfmpegCapabilities]) -> AnalysisTools:
    return AnalysisTools(media=MediaTools(ffmpeg, ffprobe), capabilities=capabilities)
