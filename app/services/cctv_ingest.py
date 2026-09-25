"""Ingesta de un video CCTV, parte 1 (spec §4.2 pasos 1-3).

El orden es obligatorio: el SHA-256 del archivo tal como Upflow lo recibio se
calcula antes de cualquier otra operacion sobre el archivo (sniff, ffprobe,
copia o remux). El registro queda en la sesion de analisis
(`video-work/cctv-{token}/source.json`) y cada job hace desde ahi su copia
verificada en `outputs/{job.id}.cctv/01_original/`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path, PureWindowsPath
from typing import Any

from app.models import utc_now
from app.services.json_store import write_json_atomically
from app.services.media_signature import ContainerGuess, container_guess, sniff_file

HASH_CHUNK_SIZE = 8 * 1024 * 1024
ORIGINAL_DIRNAME = "01_original"
SOURCE_RECORD_NAME = "source.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{n}" for n in range(1, 10)} | {f"LPT{n}" for n in range(1, 10)}
)


class VerifiedCopyMismatch(RuntimeError):
    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(f"Verified copy hash {actual} does not match the received file hash {expected}")
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class ReceivedAt:
    utc: str
    local: str


def sha256_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class IngestTools:
    hash_file: Callable[[Path], str] = sha256_file
    sniff: Callable[[Path], ContainerGuess] = sniff_file
    copy_file: Callable[[Path, Path], object] = shutil.copyfile
    now: Callable[[], datetime] = utc_now
    local_tz: tzinfo | None = None


@dataclass(frozen=True)
class SourceRecord:
    original_name: str
    size_bytes: int
    modified_at: str
    sha256: str
    received_at: ReceivedAt
    container: ContainerGuess

    def to_json(self) -> dict[str, Any]:
        return {
            "originalName": self.original_name,
            "sizeBytes": self.size_bytes,
            "mtime": self.modified_at,
            "sourceSha256": self.sha256,
            "receivedAt": {"utc": self.received_at.utc, "local": self.received_at.local},
            "container": {"kind": self.container.kind, "label": self.container.label},
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> SourceRecord:
        record = parse_source_record(payload)
        validate_source_record(record)
        return record


@dataclass(frozen=True)
class VerifiedCopy:
    path: Path
    sha256: str

    def to_json(self, job_dir: Path) -> dict[str, str]:
        return {"path": self.path.relative_to(job_dir).as_posix(), "verifiedCopySha256": self.sha256}


def iso_utc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def iso_local(moment: datetime, local_tz: tzinfo | None = None) -> str:
    return moment.astimezone(local_tz).isoformat(timespec="milliseconds")


def received_at(moment: datetime, local_tz: tzinfo | None = None) -> ReceivedAt:
    if moment.tzinfo is None:
        raise ValueError("receivedAt needs a timezone-aware datetime")
    return ReceivedAt(utc=iso_utc(moment), local=iso_local(moment, local_tz))


def modified_at(path: Path) -> str:
    return iso_utc(datetime.fromtimestamp(path.stat().st_mtime, timezone.utc))


def safe_original_name(raw: str) -> str:
    name = _UNSAFE_NAME_CHARS.sub("_", PureWindowsPath(raw).name).strip().rstrip(". ")
    if not name:
        raise ValueError(f"File name has no usable file part: {raw!r}")
    return f"_{name}" if name.split(".", 1)[0].upper() in _RESERVED_STEMS else name


def ingest_source(path: Path, original_name: str, tools: IngestTools = IngestTools()) -> SourceRecord:
    received = received_at(tools.now(), tools.local_tz)
    digest = tools.hash_file(path)
    return SourceRecord(
        original_name=safe_original_name(original_name),
        size_bytes=path.stat().st_size,
        modified_at=modified_at(path),
        sha256=digest,
        received_at=received,
        container=tools.sniff(path),
    )


def verified_copy_path(job_dir: Path, original_name: str) -> Path:
    return job_dir / ORIGINAL_DIRNAME / safe_original_name(original_name)


def make_verified_copy(
    source: Path, record: SourceRecord, job_dir: Path, tools: IngestTools = IngestTools()
) -> VerifiedCopy:
    destination = verified_copy_path(job_dir, record.original_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tools.copy_file(source, destination)
    digest = tools.hash_file(destination)
    if digest != record.sha256:
        destination.unlink(missing_ok=True)
        raise VerifiedCopyMismatch(record.sha256, digest)
    return VerifiedCopy(path=destination, sha256=digest)


def parse_source_record(payload: dict[str, Any]) -> SourceRecord:
    try:
        received = payload["receivedAt"]
        return SourceRecord(
            original_name=str(payload["originalName"]),
            size_bytes=int(payload["sizeBytes"]),
            modified_at=str(payload["mtime"]),
            sha256=str(payload["sourceSha256"]),
            received_at=ReceivedAt(utc=str(received["utc"]), local=str(received["local"])),
            container=container_guess(str(payload["container"]["kind"])),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed source record: missing or invalid {exc}") from exc


def validate_source_record(record: SourceRecord) -> None:
    if not _SHA256.fullmatch(record.sha256):
        raise ValueError("sourceSha256 must be 64 lowercase hex characters")
    if record.size_bytes < 0:
        raise ValueError("sizeBytes must not be negative")
    if safe_original_name(record.original_name) != record.original_name:
        raise ValueError(f"originalName is not a safe file name: {record.original_name!r}")


def source_record_path(session_dir: Path) -> Path:
    return session_dir / SOURCE_RECORD_NAME


def write_source_record(session_dir: Path, record: SourceRecord) -> Path:
    path = source_record_path(session_dir)
    write_json_atomically(path, record.to_json())
    return path


def read_source_record(session_dir: Path) -> SourceRecord:
    payload = json.loads(source_record_path(session_dir).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source record must be a JSON object")
    return SourceRecord.from_json(payload)
