"""Sesion de analisis CCTV en `video-work/cctv-{token}/` (spec §4.2).

El analisis (P2-14) guarda ahi el upload, `source.json` (hash calculado antes de
todo), `work.mkv` y `frame_index.csv`. Los jobs la reusan por token y nunca
borran nada: el mismo token sirve para Clarify, despues la ROI y despues mas
cuadros. El barrido de retencion la borra por inactividad.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.services.cctv_chain import CctvChainError
from app.services.cctv_ingest import (
    FRAME_INDEX_NAME,
    WORK_COPY_NAME,
    SourceRecord,
    VerifiedCopy,
    make_verified_copy,
    read_source_record,
    source_record_path,
    write_source_record,
)

SESSION_PREFIX = "cctv-"
JOB_DIR_SUFFIX = ".cctv"
UPLOAD_DIRNAME = "upload"
SESSION_NOT_FOUND = "cctv.error.sessionNotFound"
SESSION_NOT_ANALYZED = "cctv.error.sessionNotAnalyzed"
INVALID_TOKEN = "cctv.error.invalidSessionToken"

_TOKEN = re.compile(r"[A-Za-z0-9_-]{8,64}")


@dataclass(frozen=True, slots=True)
class CctvSession:
    directory: Path
    upload: Path
    record: SourceRecord
    work: Path
    frame_count: int


def check_token(token: str) -> str:
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise CctvChainError(INVALID_TOKEN, "The CCTV session token is not valid.")
    return token


def session_dir(work_root: Path, token: str) -> Path:
    return work_root / f"{SESSION_PREFIX}{check_token(token)}"


def is_session_dir_name(name: str) -> bool:
    return name.startswith(SESSION_PREFIX)


def _not_analyzed(what: str) -> CctvChainError:
    return CctvChainError(SESSION_NOT_ANALYZED, f"The CCTV session has no {what}; analyze the file first.")


def session_upload(directory: Path) -> Path:
    uploads = sorted(path for path in (directory / UPLOAD_DIRNAME).glob("*") if path.is_file())
    if len(uploads) != 1:
        raise _not_analyzed("single uploaded file")
    return uploads[0]


def count_indexed_frames(csv_text: str) -> int:
    rows = [line for line in csv_text.splitlines() if line.strip()]
    return max(0, len(rows) - 1)


def indexed_frame_count(directory: Path) -> int:
    index = directory / FRAME_INDEX_NAME
    if not index.is_file():
        raise _not_analyzed("frame index")
    count = count_indexed_frames(index.read_text(encoding="utf-8"))
    if count == 0:
        raise _not_analyzed("decoded frames")
    return count


def working_copy(directory: Path) -> Path:
    work = directory / WORK_COPY_NAME
    if not work.is_file():
        raise _not_analyzed("working copy")
    return work


def session_record(directory: Path) -> SourceRecord:
    if not source_record_path(directory).is_file():
        raise _not_analyzed("source hash record")
    return read_source_record(directory)


def load_session(work_root: Path, token: str) -> CctvSession:
    directory = session_dir(work_root, token)
    if not directory.is_dir():
        raise CctvChainError(SESSION_NOT_FOUND, "The CCTV session was not found or has expired; upload the file again.")
    return CctvSession(
        directory=directory,
        upload=session_upload(directory),
        record=session_record(directory),
        work=working_copy(directory),
        frame_count=indexed_frame_count(directory),
    )


def touch_session(directory: Path) -> None:
    # La retencion mide la inactividad por el mtime del directorio.
    os.utime(directory)


def cctv_job_dir(outputs_root: Path, job_id: str) -> Path:
    return outputs_root / f"{job_id}{JOB_DIR_SUFFIX}"


def prepare_job_dir(session: CctvSession, job_dir: Path) -> VerifiedCopy:
    # Todo lo que tiene que sobrevivir al job va a outputs/{id}.cctv: video-work/{id} se borra al terminar.
    verified = make_verified_copy(session.upload, session.record, job_dir)
    write_source_record(job_dir, session.record)
    return verified
