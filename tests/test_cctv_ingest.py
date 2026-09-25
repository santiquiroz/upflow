from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services import cctv_ingest
from app.services.cctv_ingest import (
    HASH_CHUNK_SIZE,
    ORIGINAL_DIRNAME,
    IngestTools,
    ReceivedAt,
    SourceRecord,
    VerifiedCopyMismatch,
    ingest_source,
    make_verified_copy,
    read_source_record,
    received_at,
    safe_original_name,
    sha256_file,
    verified_copy_path,
    write_source_record,
)
from app.services.media_signature import sniff_file

IMKH_BYTES = b"IMKH" + bytes(36) + b"\x00\x00\x01\xba" + bytes(4000)
BOGOTA = timezone(timedelta(hours=-5))
MOMENT = datetime(2026, 9, 25, 15, 4, 5, 123456, tzinfo=timezone.utc)


def _upload(tmp_path: Path, data: bytes = IMKH_BYTES, name: str = "upload.bin") -> Path:
    path = tmp_path / "session" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _tools(**overrides) -> IngestTools:
    return dataclasses.replace(IngestTools(now=lambda: MOMENT, local_tz=BOGOTA), **overrides)


class Spy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tools(self) -> IngestTools:
        return _tools(hash_file=self.hash_file, sniff=self.sniff, copy_file=self.copy_file)

    def hash_file(self, path: Path) -> str:
        self.calls.append(f"hash:{path.parent.name}")
        return sha256_file(path)

    def sniff(self, path: Path):
        self.calls.append("sniff")
        return sniff_file(path)

    def copy_file(self, source: Path, destination: Path) -> None:
        self.calls.append("copy")
        shutil.copyfile(source, destination)


def test_hash_matches_hashlib_on_a_file_larger_than_one_chunk(tmp_path: Path) -> None:
    data = bytes(range(256)) * 5000
    path = _upload(tmp_path, data)

    assert sha256_file(path, chunk_size=1000) == hashlib.sha256(data).hexdigest()


def test_hash_streams_in_8_mib_blocks_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _upload(tmp_path, b"x" * 10)
    requested: list[int] = []
    real_open = Path.open

    def recording_open(self: Path, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_read = handle.read

        def recording_read(size: int = -1) -> bytes:
            requested.append(size)
            return real_read(size)

        handle.read = recording_read
        return handle

    monkeypatch.setattr(Path, "open", recording_open)

    assert sha256_file(path) == hashlib.sha256(b"x" * 10).hexdigest()
    assert HASH_CHUNK_SIZE == 8 * 1024 * 1024
    assert requested and set(requested) == {HASH_CHUNK_SIZE}


def test_hash_of_an_empty_file_is_the_empty_digest(tmp_path: Path) -> None:
    assert sha256_file(_upload(tmp_path, b"")) == hashlib.sha256(b"").hexdigest()


def test_hash_happens_before_any_other_file_operation(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    spy = Spy()

    record = ingest_source(upload, "export.mp4", spy.tools())
    make_verified_copy(upload, record, tmp_path / "job.cctv", spy.tools())

    assert spy.calls == ["hash:session", "sniff", "copy", f"hash:{ORIGINAL_DIRNAME}"]


def test_hash_is_computed_on_the_file_as_received(tmp_path: Path) -> None:
    upload = _upload(tmp_path)

    record = ingest_source(upload, "export.mp4", _tools())

    assert record.sha256 == hashlib.sha256(IMKH_BYTES).hexdigest()
    assert record.size_bytes == len(IMKH_BYTES)
    assert record.original_name == "export.mp4"


def test_hash_record_identifies_the_container_by_content_not_extension(tmp_path: Path) -> None:
    record = ingest_source(_upload(tmp_path, name="upload.mkv"), "export.mp4", _tools())

    assert record.container.kind == "hikvision_ps"
    assert record.container.label == "Hikvision PS"


def test_hash_record_carries_received_at_in_utc_and_local_offset(tmp_path: Path) -> None:
    record = ingest_source(_upload(tmp_path), "export.mp4", _tools())

    assert record.received_at == ReceivedAt(utc="2026-09-25T15:04:05.123Z", local="2026-09-25T10:04:05.123-05:00")


def test_hash_record_mtime_is_utc_iso(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    stamp = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
    os.utime(upload, (stamp, stamp))

    assert ingest_source(upload, "a.dav", _tools()).modified_at == "2024-01-02T03:04:05.000Z"


def test_hash_received_at_rejects_a_naive_datetime() -> None:
    with pytest.raises(ValueError):
        received_at(datetime(2026, 9, 25, 15, 4, 5))


def test_hash_received_at_converts_a_non_utc_moment() -> None:
    moment = datetime(2026, 9, 25, 10, 4, 5, tzinfo=BOGOTA)

    assert received_at(moment, BOGOTA) == ReceivedAt(utc="2026-09-25T15:04:05.000Z", local="2026-09-25T10:04:05.000-05:00")


def test_hash_record_survives_the_session_round_trip(tmp_path: Path) -> None:
    session = tmp_path / "cctv-token"
    record = ingest_source(_upload(tmp_path), "export.mp4", _tools())

    write_source_record(session, record)

    assert read_source_record(session) == record


def test_hash_record_json_uses_the_report_field_names(tmp_path: Path) -> None:
    record = ingest_source(_upload(tmp_path), "export.mp4", _tools())

    payload = record.to_json()

    assert payload["sourceSha256"] == record.sha256
    assert payload["receivedAt"] == {"utc": "2026-09-25T15:04:05.123Z", "local": "2026-09-25T10:04:05.123-05:00"}
    assert payload["container"] == {"kind": "hikvision_ps", "label": "Hikvision PS"}
    assert set(payload) == {"originalName", "sizeBytes", "mtime", "sourceSha256", "receivedAt", "container"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(sourceSha256="not-a-hash"),
        lambda p: p.update(sizeBytes=-1),
        lambda p: p.update(originalName="../evil.mp4"),
        lambda p: p["container"].update(kind="nope"),
        lambda p: p.pop("receivedAt"),
    ],
)
def test_hash_record_from_json_rejects_malformed_payloads(tmp_path: Path, mutate) -> None:
    payload = ingest_source(_upload(tmp_path), "export.mp4", _tools()).to_json()
    mutate(payload)

    with pytest.raises(ValueError):
        SourceRecord.from_json(payload)


def test_hash_record_read_of_a_missing_session_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_source_record(tmp_path / "cctv-missing")


def test_verified_copy_is_byte_identical_and_rehashed(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    record = ingest_source(upload, "export.mp4", _tools())
    job_dir = tmp_path / "outputs" / "job-1.cctv"

    copy = make_verified_copy(upload, record, job_dir, _tools())

    assert copy.path == job_dir / ORIGINAL_DIRNAME / "export.mp4"
    assert copy.path.read_bytes() == IMKH_BYTES
    assert copy.sha256 == record.sha256
    assert upload.exists()


def test_verified_copy_json_carries_the_verified_hash(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    record = ingest_source(upload, "export.mp4", _tools())
    job_dir = tmp_path / "job.cctv"

    copy = make_verified_copy(upload, record, job_dir, _tools())

    assert copy.to_json(job_dir) == {"path": f"{ORIGINAL_DIRNAME}/export.mp4", "verifiedCopySha256": record.sha256}


def test_verified_copy_can_be_made_twice_from_the_same_session(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    record = ingest_source(upload, "export.mp4", _tools())

    first = make_verified_copy(upload, record, tmp_path / "job-1.cctv", _tools())
    second = make_verified_copy(upload, record, tmp_path / "job-2.cctv", _tools())

    assert first.sha256 == second.sha256 == record.sha256
    assert first.path != second.path


def test_verified_copy_mismatch_raises_and_removes_the_copy(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    record = ingest_source(upload, "export.mp4", _tools())
    upload.write_bytes(IMKH_BYTES + b"tampered")
    job_dir = tmp_path / "job.cctv"

    with pytest.raises(VerifiedCopyMismatch) as caught:
        make_verified_copy(upload, record, job_dir, _tools())

    assert record.sha256 in str(caught.value)
    assert not (job_dir / ORIGINAL_DIRNAME / "export.mp4").exists()


def test_verified_copy_mismatch_when_the_copy_itself_is_corrupted(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    record = ingest_source(upload, "export.mp4", _tools())

    def corrupting_copy(source: Path, destination: Path) -> None:
        destination.write_bytes(source.read_bytes()[:-1])

    with pytest.raises(VerifiedCopyMismatch):
        make_verified_copy(upload, record, tmp_path / "job.cctv", _tools(copy_file=corrupting_copy))


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("export.mp4", "export.mp4"),
        ("C:\\Users\\x\\Videos\\ch01_20260101.mp4", "ch01_20260101.mp4"),
        ("/tmp/../ch02.dav", "ch02.dav"),
        ("cam:1?.mp4", "cam_1_.mp4"),
        ("  spaced .mp4  ", "spaced .mp4"),
        ("CON.mp4", "_CON.mp4"),
        ("nul", "_nul"),
    ],
)
def test_verified_copy_uses_a_safe_original_name(raw: str, safe: str) -> None:
    assert safe_original_name(raw) == safe


@pytest.mark.parametrize("raw", ["", "   ", ".", "..", "C:\\", "C:", "/"])
def test_verified_copy_rejects_names_without_a_file_part(raw: str) -> None:
    with pytest.raises(ValueError):
        safe_original_name(raw)


def test_verified_copy_path_stays_inside_the_job_directory(tmp_path: Path) -> None:
    job_dir = tmp_path / "job.cctv"

    path = verified_copy_path(job_dir, "..\\..\\outside.mp4")

    assert path == job_dir / ORIGINAL_DIRNAME / "outside.mp4"


def test_hash_default_tools_use_the_real_clock_and_helpers() -> None:
    tools = IngestTools()

    assert tools.hash_file is cctv_ingest.sha256_file
    assert tools.sniff is sniff_file
    assert tools.copy_file is shutil.copyfile
    assert tools.now().tzinfo is not None
