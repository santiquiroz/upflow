from __future__ import annotations

import dataclasses
import hashlib
import os
import random
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from app.config import Settings
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
from app.services.media_signature import DAHUA_DAV, HIKVISION_PS, MP4, RAW_H264, RAW_HEVC, sniff_file
from ffmpeg_support import needs_ffmpeg

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


# --- Parte 2: remux, decodificacion, indice de cuadros, lite y audio ---

WORK = cctv_ingest.WORK_COPY_NAME
SUCCESS = (b"", b"", 0)
FAILURE = (b"", b"Invalid data found when processing input", 1)


class FakeRunner:
    def __init__(self, results: dict[str, tuple[bytes, bytes, int]], write_output: bool = True) -> None:
        self.results = results
        self.write_output = write_output
        self.commands: list[list[str]] = []

    async def __call__(self, command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        self.commands.append(command)
        stdout, stderr, code = self.results.get(self.label(command), FAILURE)
        if code == 0 and self.write_output:
            Path(command[-1]).write_bytes(b"mkv")
        return stdout, stderr, code

    @staticmethod
    def label(command: list[str]) -> str:
        head = command[: command.index("-i")]
        return head[head.index("-f") + 1] if "-f" in head else "auto"


def _media_tools(run) -> cctv_ingest.MediaTools:
    return cctv_ingest.MediaTools(ffmpeg=Path("ffmpeg.exe"), ffprobe=Path("ffprobe.exe"), run=run)


async def _remux(runner: FakeRunner, source: Path, container, frame_rate: str | None, copytb: bool = False):
    tools = _media_tools(runner)
    return await cctv_ingest.remux_working_copy(tools, source, source.parent, container, frame_rate, copytb)


def _video_probe(r: str, avg: str) -> dict:
    return {"streams": [{"codec_type": "video", "r_frame_rate": r, "avg_frame_rate": avg}]}


@pytest.mark.parametrize(
    ("raw", "value"),
    [("25/1", 25), ("30000/1001", Fraction(30000, 1001)), ("15", 15), ("12.5", Fraction(25, 2))],
)
def test_rate_parses_ffprobe_rates(raw: str, value) -> None:
    assert cctv_ingest.parse_rate(raw) == value


@pytest.mark.parametrize("raw", [None, "", "0/0", "0", "N/A", "25/0", "-5", "abc", "nan", "inf"])
def test_rate_rejects_missing_or_non_positive_rates(raw) -> None:
    assert cctv_ingest.parse_rate(raw) is None


def test_rate_given_by_the_user_must_be_positive() -> None:
    assert cctv_ingest.validate_frame_rate(" 15 ") == "15"
    with pytest.raises(ValueError):
        cctv_ingest.validate_frame_rate("0")


def test_header_rate_prefers_the_average_over_the_field_rate_guess() -> None:
    assert cctv_ingest.header_frame_rate(_video_probe("50/1", "25/1")) == "25/1"
    assert cctv_ingest.header_frame_rate(_video_probe("15/1", "0/0")) == "15/1"
    assert cctv_ingest.header_frame_rate({"streams": [{"codec_type": "audio"}]}) is None


def test_header_suggests_vfr_when_rates_disagree_by_more_than_ten_percent() -> None:
    assert cctv_ingest.header_suggests_vfr(_video_probe("25/1", "40/3"))
    assert not cctv_ingest.header_suggests_vfr(_video_probe("25/1", "25/1"))
    assert not cctv_ingest.header_suggests_vfr(_video_probe("25/1", "0/0"))
    assert not cctv_ingest.header_suggests_vfr({})


def test_remux_command_copies_every_stream_without_reencoding(tmp_path: Path) -> None:
    command = cctv_ingest.build_remux_command(Path("ffmpeg.exe"), tmp_path / "in.mp4", tmp_path / WORK, (), False)

    assert command[command.index("-fflags") + 1] == "+genpts"
    assert command.index("-fflags") < command.index("-i")
    assert command[command.index("-c") + 1] == "copy"
    assert command[command.index("-map") : command.index("-map") + 2] == ["-map", "0"]
    assert "-copytb" not in command and "-vf" not in command and "-c:v" not in command
    assert command[-1] == str(tmp_path / WORK)


def test_remux_command_adds_copytb_for_vfr_and_input_options_before_the_input(tmp_path: Path) -> None:
    command = cctv_ingest.build_remux_command(
        Path("ffmpeg.exe"), tmp_path / "in.bin", tmp_path / WORK, ("-f", "h264", "-r", "15"), copytb=True
    )

    assert command[command.index("-copytb") + 1] == "1"
    assert command.index("-f") < command.index("-r") < command.index("-i")


def test_remux_plans_follow_the_spec_order() -> None:
    plans = cctv_ingest.remux_plans(HIKVISION_PS, "25/1")

    assert [p.label for p in plans] == ["auto", "mpeg", "dhav", "h264"]
    assert plans[-1].input_args == ("-f", "h264", "-r", "25/1")
    assert cctv_ingest.remux_plans(RAW_HEVC, "15")[-1].input_args == ("-f", "hevc", "-r", "15")


def test_remux_plan_for_raw_streams_is_skipped_without_a_frame_rate() -> None:
    raw = cctv_ingest.remux_plans(RAW_H264, None)[-1]

    assert raw.skip_reason is not None and raw.input_args == ()


async def test_remux_first_attempt_success_records_one_attempt(tmp_path: Path) -> None:
    runner = FakeRunner({"auto": SUCCESS})

    copy = await _remux(runner, tmp_path / "in.mp4", MP4, None, copytb=False)

    assert copy.method == "auto"
    assert [a.label for a in copy.attempts] == ["auto"] and copy.attempts[0].ok
    assert copy.path == tmp_path / WORK
    assert copy.sha256 == hashlib.sha256(b"mkv").hexdigest()


async def test_remux_retries_in_order_until_one_works(tmp_path: Path) -> None:
    runner = FakeRunner({"dhav": SUCCESS})

    copy = await _remux(runner, tmp_path / "in.dav", DAHUA_DAV, None, copytb=False)

    assert [FakeRunner.label(c) for c in runner.commands] == ["auto", "mpeg", "dhav"]
    assert copy.method == "dhav"
    assert [(a.label, a.ok) for a in copy.attempts] == [("auto", False), ("mpeg", False), ("dhav", True)]
    assert "Invalid data" in copy.attempts[0].detail


async def test_remux_raw_retry_uses_the_given_frame_rate(tmp_path: Path) -> None:
    runner = FakeRunner({"h264": SUCCESS})

    copy = await _remux(runner, tmp_path / "in.bin", RAW_H264, "12.5", copytb=True)

    assert copy.method == "h264" and copy.copytb
    last = runner.commands[-1]
    assert last[last.index("-r") + 1] == "12.5"
    assert "-copytb" in last


async def test_remux_treats_an_empty_output_as_a_failed_attempt(tmp_path: Path) -> None:
    runner = FakeRunner({"auto": SUCCESS, "mpeg": SUCCESS}, write_output=False)

    with pytest.raises(cctv_ingest.RemuxFailed) as caught:
        await _remux(runner, tmp_path / "in.mp4", MP4, "25", copytb=False)

    assert [a.ok for a in caught.value.attempts] == [False, False, False, False]
    assert "no output" in caught.value.attempts[0].detail


async def test_remux_failure_lists_every_attempt_and_removes_the_partial_copy(tmp_path: Path) -> None:
    (tmp_path / WORK).write_bytes(b"partial")
    runner = FakeRunner({})

    with pytest.raises(cctv_ingest.RemuxFailed) as caught:
        await _remux(runner, tmp_path / "in.bin", RAW_H264, None, copytb=False)

    attempts = caught.value.attempts
    assert [a.label for a in attempts] == ["auto", "mpeg", "dhav", "h264"]
    assert attempts[-1].detail == cctv_ingest.NO_FRAME_RATE_DETAIL
    assert len(runner.commands) == 3
    assert not (tmp_path / WORK).exists()


async def test_remux_attempt_detail_names_files_without_their_folders(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"

    async def run(command: list[str], timeout: float):
        return b"", f"{source}: Invalid data found when processing input".encode(), 1

    plan = cctv_ingest.RemuxPlan("auto", ())
    attempt = await cctv_ingest.run_remux_attempt(_media_tools(run), plan, source, tmp_path / WORK, copytb=False)

    assert attempt.detail == "in.mp4: Invalid data found when processing input"


def test_remux_attempt_json_has_no_machine_paths() -> None:
    attempt = cctv_ingest.RemuxAttempt(label="mpeg", input_args=("-f", "mpeg"), ok=True, detail="ok")

    assert attempt.to_json() == {"method": "mpeg", "inputOptions": ["-f", "mpeg"], "ok": True, "detail": "ok"}


def test_working_copy_json_carries_hash_method_attempts_and_the_report_note(tmp_path: Path) -> None:
    attempt = cctv_ingest.RemuxAttempt(label="auto", input_args=(), ok=True, detail="ok")
    copy = cctv_ingest.WorkingCopy(tmp_path / WORK, sha256="a" * 64, method="auto", attempts=(attempt,), copytb=False)

    assert copy.to_json() == {
        "sha256": "a" * 64,
        "method": "auto",
        "copytb": False,
        "attempts": [attempt.to_json()],
        "note": cctv_ingest.REMUX_NOTE,
    }


async def test_frame_index_is_written_to_the_session_and_hashed(tmp_path: Path) -> None:
    text = b"".join(
        f"key_frame={int(n == 0)},best_effort_timestamp_time={n / 25:.6f},pkt_size=900,pict_type=P\n".encode()
        for n in range(3)
    )

    async def run(command: list[str], timeout: float):
        return text, b"", 0

    index = await cctv_ingest.build_frame_index(_media_tools(run), tmp_path / WORK, tmp_path)

    assert index.csv_path == tmp_path / cctv_ingest.FRAME_INDEX_NAME
    assert index.csv_sha256 == sha256_file(index.csv_path)
    assert index.summary.frame_count == 3
    assert len(index.frames) == 3


async def test_frame_index_failure_raises_an_ingest_error(tmp_path: Path) -> None:
    async def run(command: list[str], timeout: float):
        return b"", b"moov atom not found", 1

    with pytest.raises(cctv_ingest.IngestError, match="moov atom"):
        await cctv_ingest.build_frame_index(_media_tools(run), tmp_path / WORK, tmp_path)


def test_decode_check_command_reads_only_the_first_100_frames(tmp_path: Path) -> None:
    command = cctv_ingest.build_decode_check_command(Path("ffmpeg.exe"), tmp_path / WORK)

    assert command[command.index("-frames:v") + 1] == "100"
    assert command[command.index("-v") + 1] == "error"
    assert command[-3:] == ["-f", "null", "-"]


def test_decode_check_parses_the_last_progress_frame_count() -> None:
    assert cctv_ingest.parse_progress_frames(b"frame=10\nprogress=continue\nframe=100\nprogress=end\n") == 100
    assert cctv_ingest.parse_progress_frames(b"") == 0


@pytest.mark.parametrize(
    ("decoded", "errors", "failed"),
    [(100, 0, False), (100, 50, False), (100, 51, True), (0, 0, True), (10, 6, True)],
)
def test_decode_check_fails_on_zero_frames_or_errors_on_most_frames(decoded: int, errors: int, failed: bool) -> None:
    assert cctv_ingest.is_decode_failed(decoded, errors) is failed


async def test_decode_check_counts_error_lines(tmp_path: Path) -> None:
    async def run(command: list[str], timeout: float):
        return b"frame=4\nprogress=end\n", b"[h264] no frame!\n\n[h264] error while decoding MB 6 0\n[h264] x\n", 1

    check = await cctv_ingest.check_decodable(_media_tools(run), tmp_path / WORK)

    assert check == cctv_ingest.DecodeCheck(frames_decoded=4, error_lines=3, decode_failed=True)


@pytest.mark.parametrize(("size", "display"), [((960, 1080), (1920, 1080)), ((640, 720), (1280, 720))])
def test_lite_sizes_propose_a_2_to_1_aspect_fix(size: tuple[int, int], display: tuple[int, int]) -> None:
    lite = cctv_ingest.lite_aspect(*size)

    assert lite is not None and lite.sar == 2
    assert lite.display_size == display
    expected = {"storedSize": list(size), "displaySize": list(display), "sar": "2:1", "filter": "setsar=2"}
    assert lite.to_json() == expected


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 960), (704, 576)])
def test_lite_is_not_proposed_for_other_sizes(size: tuple[int, int]) -> None:
    assert cctv_ingest.lite_aspect(*size) is None


@pytest.mark.parametrize(
    ("codec", "family"), [("pcm_alaw", "g711"), ("pcm_mulaw", "g711"), ("aac", "aac"), ("mp2", "other")]
)
def test_audio_family_recognises_g711_and_aac(codec: str, family: str) -> None:
    assert cctv_ingest.audio_family(codec) == family


def test_audio_tracks_read_codec_rate_and_channels() -> None:
    probe = {
        "streams": [
            {"codec_type": "video"},
            {"codec_type": "audio", "codec_name": "pcm_alaw", "sample_rate": "8000", "channels": 1},
        ]
    }

    assert cctv_ingest.audio_tracks(probe) == (
        cctv_ingest.AudioTrack(codec="pcm_alaw", sample_rate=8000, channels=1, family="g711"),
    )


def test_audio_is_never_enhanced_only_decoded_per_lane() -> None:
    assert cctv_ingest.audio_codec_for_lane("classic") == "flac"
    assert cctv_ingest.audio_codec_for_lane("visual") == "aac"


def test_video_stream_reads_the_first_video_stream() -> None:
    probe = {
        "streams": [
            {"codec_type": "audio"},
            {"codec_type": "video", "codec_name": "hevc", "width": 960, "height": 1080, "avg_frame_rate": "25/1"},
        ]
    }

    expected = cctv_ingest.VideoStream(codec="hevc", width=960, height=1080, header_rate="25/1")
    assert cctv_ingest.video_stream(probe) == expected
    assert cctv_ingest.video_stream({"streams": []}) is None


async def test_probe_falls_back_to_the_sniffed_demuxer(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    async def run(command: list[str], timeout: float):
        seen.append(command)
        return (b'{"streams": []}', b"", 0) if "-f" in command else (b"", b"Invalid data", 1)

    probe = await cctv_ingest.probe_source(_media_tools(run), tmp_path / "in.mp4", HIKVISION_PS)

    assert probe == {"streams": []}
    assert len(seen) == 2 and seen[1][seen[1].index("-f") + 1] == "mpeg"


async def test_probe_failure_without_a_fallback_raises(tmp_path: Path) -> None:
    async def run(command: list[str], timeout: float):
        return b"", b"Invalid data", 1

    with pytest.raises(cctv_ingest.IngestError, match="ffprobe"):
        await cctv_ingest.probe_source(_media_tools(run), tmp_path / "in.mp4", MP4)


# --- Parte 2 con ffmpeg real sobre clips sinteticos ---

SRC_320 = ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25"]
X264 = ["-c:v", "libx264", "-preset", "ultrafast"]


def _real_tools() -> cctv_ingest.MediaTools:
    settings = Settings()
    return cctv_ingest.MediaTools(ffmpeg=settings.ffmpeg_binary_path, ffprobe=settings.ffprobe_binary_path)


def _make(workdir: Path, name: str, args: list[str]) -> Path:
    subprocess.run(
        [str(Settings().ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y", *args, name],
        cwd=workdir,
        check=True,
        capture_output=True,
    )
    return workdir / name


async def _ingest(tmp_path: Path, clip: Path, frame_rate: str | None = None) -> cctv_ingest.WorkingIngest:
    session = tmp_path / "cctv-token"
    session.mkdir(exist_ok=True)
    return await cctv_ingest.ingest_working_copy(_real_tools(), clip, session, sniff_file(clip), frame_rate)


def _scramble_slices(data: bytes, seed: int = 7) -> bytes:
    # Simula un export cifrado: NAL intactos, carga de los slices ilegible y sin bytes 0 (no crea start codes).
    rng = random.Random(seed)
    out = bytearray()
    for part in re.split(rb"(\x00\x00\x01)", data):
        if part and part != b"\x00\x00\x01" and part[0] & 0x1F in (1, 5):
            part = part[:2] + bytes(rng.randint(1, 255) for _ in range(len(part) - 2))
        out += part
    return bytes(out)


@needs_ffmpeg
async def test_real_imkh_program_stream_is_remuxed_and_indexed(tmp_path: Path) -> None:
    sine = ["-f", "lavfi", "-i", "sine=sample_rate=16000"]
    video = [*X264, "-g", "12", "-bf", "0"]
    ps = _make(tmp_path, "ps.mpg", [*SRC_320, *sine, "-t", "2", *video, "-c:a", "mp2", "-shortest", "-f", "vob"])
    upload = tmp_path / "export.mp4"
    upload.write_bytes(b"IMKH" + bytes(36) + ps.read_bytes())

    result = await _ingest(tmp_path, upload)

    assert sniff_file(upload).kind == "hikvision_ps"
    assert result.working_copy.path.name == WORK and result.working_copy.attempts[-1].ok
    assert result.working_copy.sha256 == sha256_file(result.working_copy.path)
    assert result.index is not None and result.index.summary.frame_count == 50
    assert result.index.summary.measured_fps == pytest.approx(25.0)
    assert result.index.summary.gop.max_length == 12 and result.index.summary.gop.b_ratio == 0
    assert [track.codec for track in result.audio] == ["mp2"]
    assert not result.decode.decode_failed and result.warnings == ()


@needs_ffmpeg
async def test_real_frame_index_csv_matches_the_working_copy(tmp_path: Path) -> None:
    clip = _make(tmp_path, "gop.mkv", [*SRC_320, "-t", "2", *X264, "-g", "12", "-bf", "2"])

    result = await _ingest(tmp_path, clip)

    rows = result.index.csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "n,pts_time,key_frame,pict_type,pkt_size" and len(rows) == 51
    assert result.index.csv_sha256 == sha256_file(result.index.csv_path)
    assert result.index.summary.gop.b_ratio > 0 and result.index.summary.gop.keyframes >= 4


@needs_ffmpeg
async def test_real_vfr_clip_is_classified_and_measured(tmp_path: Path) -> None:
    jitter = ["-vf", "settb=1/1000,setpts='(N/25+0.012*eq(mod(N,4),1))/TB'", "-fps_mode", "passthrough"]
    jitter += ["-enc_time_base:v", "1/1000"]
    clip = _make(tmp_path, "vfr.mkv", [*SRC_320, "-t", "2", *jitter, *X264, "-bf", "0"])

    summary = (await _ingest(tmp_path, clip)).index.summary

    assert summary.is_vfr
    assert summary.measured_fps == pytest.approx(25.0, rel=0.05)
    assert summary.gaps == ()


@needs_ffmpeg
async def test_real_dropped_frames_show_up_as_a_gap(tmp_path: Path) -> None:
    drop = ["-vf", "select='not(between(n,10,19))'", "-fps_mode", "passthrough"]
    clip = _make(tmp_path, "gap.mkv", [*SRC_320, "-t", "2", *drop, *X264, "-bf", "0"])

    summary = (await _ingest(tmp_path, clip)).index.summary

    assert summary.frame_count == 40 and not summary.is_vfr
    assert [(g.after_frame, round(g.end - g.start, 2)) for g in summary.gaps] == [(9, 0.44)]


@needs_ffmpeg
async def test_real_frozen_section_shows_up_as_probable_duplicates(tmp_path: Path) -> None:
    freeze = ["-filter_complex", "split[a][b];[a][b]freezeframes=first=25:last=44:replace=24"]
    clip = _make(tmp_path, "dup.mkv", [*SRC_320, "-t", "3", *freeze, "-c:v", "libx264", "-bf", "0", "-g", "250"])

    summary = (await _ingest(tmp_path, clip)).index.summary

    assert 15 <= summary.probable_duplicates <= 20


@needs_ffmpeg
async def test_real_raw_h264_is_remuxed_with_the_given_frame_rate(tmp_path: Path) -> None:
    raw = _make(tmp_path, "raw.bin", [*SRC_320, "-t", "2", *X264, "-bf", "0", "-f", "h264"])
    session = tmp_path / "cctv-token"
    session.mkdir()
    plan = cctv_ingest.remux_plans(RAW_H264, "15")[-1]

    attempt = await cctv_ingest.run_remux_attempt(_real_tools(), plan, raw, session / WORK, copytb=False)
    index = await cctv_ingest.build_frame_index(_real_tools(), session / WORK, session)

    assert attempt.ok
    assert index.summary.frame_count == 50
    assert index.summary.measured_fps == pytest.approx(15.0, rel=0.05)


@needs_ffmpeg
async def test_real_scrambled_recording_is_undecodable_and_not_indexed(tmp_path: Path) -> None:
    raw = _make(tmp_path, "raw.264", [*SRC_320, "-t", "2", *X264, "-bf", "0", "-f", "h264"])
    upload = tmp_path / "encrypted.bin"
    upload.write_bytes(_scramble_slices(raw.read_bytes()))

    result = await _ingest(tmp_path, upload, frame_rate="25")

    assert result.decode.decode_failed
    assert result.index is None
    assert result.warnings == (cctv_ingest.UNDECODABLE_WARNING,)
    assert not (tmp_path / "cctv-token" / cctv_ingest.FRAME_INDEX_NAME).exists()
    assert result.to_json()["decodeFailed"] and result.to_json()["frameIndex"] is None
    assert cctv_ingest.source_facts(result).frame_count is None


@needs_ffmpeg
async def test_real_lite_recording_and_g711_audio(tmp_path: Path) -> None:
    lite_source = ["-f", "lavfi", "-i", "testsrc2=size=960x1080:rate=25", "-f", "lavfi", "-i", "sine=sample_rate=8000"]
    clip = _make(tmp_path, "lite.mkv", [*lite_source, "-t", "0.4", *X264, "-c:a", "pcm_alaw", "-shortest"])

    result = await _ingest(tmp_path, clip)

    assert result.lite is not None and result.lite.display_size == (1920, 1080)
    assert result.audio == (cctv_ingest.AudioTrack(codec="pcm_alaw", sample_rate=8000, channels=1, family="g711"),)
    assert result.warnings == (cctv_ingest.LITE_WARNING,)
    facts = cctv_ingest.source_facts(result)
    assert (facts.width, facts.height, facts.is_lite, facts.frame_count) == (960, 1080, True, 10)


@needs_ffmpeg
async def test_real_file_without_video_is_rejected(tmp_path: Path) -> None:
    clip = _make(tmp_path, "audio.mkv", ["-f", "lavfi", "-i", "sine=sample_rate=8000", "-t", "0.5", "-c:a", "pcm_alaw"])

    with pytest.raises(cctv_ingest.NoVideoStream):
        await _ingest(tmp_path, clip)
