"""Ingesta de un video CCTV (spec §4.2).

Parte 1 (pasos 1-3): el SHA-256 del archivo tal como Upflow lo recibio se
calcula antes de cualquier otra operacion sobre el archivo (sniff, ffprobe,
copia o remux). El registro queda en la sesion de analisis
(`video-work/cctv-{token}/source.json`) y cada job hace desde ahi su copia
verificada en `outputs/{job.id}.cctv/01_original/`.

Parte 2 (pasos 4-8): remux sin re-encodear a `work.mkv` con su cadena de
reintentos, prueba de decodificacion, indice de cuadros (`frame_index.csv`) con
fps medidos, CFR/VFR, huecos, duplicados probables y GOP, modo lite y audio.
La prueba de decodificacion va antes del indice porque el indice decodifica el
video entero y un archivo cifrado no debe pagar ese costo. Los umbrales y
`LITE_STORAGE_SIZES` son provisionales hasta ver exports reales (P2-00/P2-VAL).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from fractions import Fraction
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from app.models import utc_now
from app.services.cctv_frame_index import (
    FrameEntry,
    FrameIndexSummary,
    frame_index_csv,
    optional_int,
    parse_frame_index,
    summarize_frame_index,
)
from app.services.json_store import write_json_atomically
from app.services.media_signature import ContainerGuess, container_guess, sniff_file
from app.services.process_runner import is_non_empty_file, run_guarded_process
from app.services.video_analysis import ProcessRunner, SourceFacts

HASH_CHUNK_SIZE = 8 * 1024 * 1024
ORIGINAL_DIRNAME = "01_original"
SOURCE_RECORD_NAME = "source.json"
WORK_COPY_NAME = "work.mkv"
FRAME_INDEX_NAME = "frame_index.csv"
FRAME_INDEX_ENTRIES = "frame=best_effort_timestamp_time,pkt_size,key_frame,pict_type"
REMUX_NOTE = "Source was remuxed to Matroska without re-encoding"
NO_FRAME_RATE_DETAIL = "Skipped: a raw stream needs a frame rate and neither the header nor the user gave one"
UNDECODABLE_WARNING = "cctv.undecodable"
LITE_WARNING = "cctv.lite"
STEP_TIMEOUT_SECONDS = 3600.0
STDERR_TAIL_CHARS = 2000

RATE_DISAGREEMENT = 0.10
DECODE_CHECK_FRAMES = 100
DECODE_ERROR_RATIO = 0.5
LITE_STORAGE_SIZES: dict[tuple[int, int], Fraction] = {(960, 1080): Fraction(2), (640, 720): Fraction(2)}
CLASSIC_AUDIO_CODEC = "flac"
VISUAL_AUDIO_CODEC = "aac"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{n}" for n in range(1, 10)} | {f"LPT{n}" for n in range(1, 10)}
)
_G711_CODECS = frozenset({"pcm_alaw", "pcm_mulaw"})
_FORCED_DEMUXERS = ("mpeg", "dhav")
_RATE = re.compile(r"\d+(?:\.\d+)?(?:/\d+)?")

AudioFamily = Literal["g711", "aac", "other"]
Lane = Literal["classic", "visual"]


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


class IngestError(RuntimeError):
    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"CCTV ingest step '{step}' failed: {detail}")
        self.step = step
        self.detail = detail


class NoVideoStream(IngestError):
    def __init__(self) -> None:
        super().__init__("probe", "the file has no video stream")


@dataclass(frozen=True)
class MediaTools:
    ffmpeg: Path
    ffprobe: Path
    run: ProcessRunner = run_guarded_process
    timeout: float = STEP_TIMEOUT_SECONDS


def stderr_tail(stderr: bytes) -> str:
    return stderr.decode("utf-8", errors="replace")[-STDERR_TAIL_CHARS:].strip()


def last_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


def failure_detail(stderr: bytes, returncode: int) -> str:
    return stderr_tail(stderr) or f"exit code {returncode}"


# --- Tasas de cuadros ---


def parse_rate(raw: str | None) -> Fraction | None:
    text = (raw or "").strip()
    if not _RATE.fullmatch(text):
        return None
    try:
        rate = Fraction(text)
    except ZeroDivisionError:
        return None
    return rate if rate > 0 else None


def validate_frame_rate(raw: str) -> str:
    text = raw.strip()
    if parse_rate(text) is None:
        raise ValueError(f"Frame rate must be a positive number or fraction: {raw!r}")
    return text


def first_stream(probe: dict[str, Any], codec_type: str) -> dict[str, Any] | None:
    return next((s for s in probe.get("streams", []) if s.get("codec_type") == codec_type), None)


def header_frame_rate(probe: dict[str, Any]) -> str | None:
    stream = first_stream(probe, "video") or {}
    # avg_frame_rate primero: en H.264 crudo r_frame_rate suele ser la tasa de campos (50 para 25 fps).
    candidates = (stream.get("avg_frame_rate"), stream.get("r_frame_rate"))
    return next((str(rate) for rate in candidates if parse_rate(rate) is not None), None)


def header_suggests_vfr(probe: dict[str, Any]) -> bool:
    stream = first_stream(probe, "video") or {}
    real = parse_rate(stream.get("r_frame_rate"))
    average = parse_rate(stream.get("avg_frame_rate"))
    if real is None or average is None:
        return False
    return abs(real - average) / average > RATE_DISAGREEMENT


# --- ffprobe ---


def build_probe_command(ffprobe: Path, path: Path, demuxer: str | None = None) -> list[str]:
    forced = ["-f", demuxer] if demuxer else []
    return [str(ffprobe), "-v", "error", *forced, "-show_format", "-show_streams", "-of", "json", str(path)]


def parse_probe(stdout: bytes) -> dict[str, Any]:
    payload = json.loads(stdout.decode("utf-8", errors="replace") or "{}")
    if not isinstance(payload, dict):
        raise IngestError("probe", "ffprobe did not return a JSON object")
    return payload


async def probe_media(tools: MediaTools, path: Path, demuxer: str | None = None) -> dict[str, Any]:
    stdout, stderr, returncode = await tools.run(build_probe_command(tools.ffprobe, path, demuxer), tools.timeout)
    if returncode != 0:
        raise IngestError("ffprobe", failure_detail(stderr, returncode))
    return parse_probe(stdout)


async def probe_source(tools: MediaTools, path: Path, container: ContainerGuess) -> dict[str, Any]:
    try:
        return await probe_media(tools, path)
    except IngestError:
        if container.ffmpeg_format is None:
            raise
    return await probe_media(tools, path, container.ffmpeg_format)


# --- Remux a work.mkv ---


@dataclass(frozen=True)
class RemuxPlan:
    label: str
    input_args: tuple[str, ...]
    skip_reason: str | None = None


@dataclass(frozen=True)
class RemuxAttempt:
    label: str
    input_args: tuple[str, ...]
    ok: bool
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {"method": self.label, "inputOptions": list(self.input_args), "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class WorkingCopy:
    path: Path
    sha256: str
    method: str
    attempts: tuple[RemuxAttempt, ...]
    copytb: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "method": self.method,
            "copytb": self.copytb,
            "attempts": [attempt.to_json() for attempt in self.attempts],
            "note": REMUX_NOTE,
        }


class RemuxFailed(IngestError):
    def __init__(self, attempts: tuple[RemuxAttempt, ...]) -> None:
        super().__init__("remux", "; ".join(f"{a.label}: {last_line(a.detail)}" for a in attempts))
        self.attempts = attempts


def raw_demuxer(container: ContainerGuess) -> str:
    return "hevc" if container.kind == "raw_hevc" else "h264"


def raw_remux_plan(container: ContainerGuess, frame_rate: str | None) -> RemuxPlan:
    demuxer = raw_demuxer(container)
    if frame_rate is None:
        return RemuxPlan(demuxer, (), NO_FRAME_RATE_DETAIL)
    return RemuxPlan(demuxer, ("-f", demuxer, "-r", frame_rate))


def remux_plans(container: ContainerGuess, frame_rate: str | None) -> tuple[RemuxPlan, ...]:
    forced = tuple(RemuxPlan(name, ("-f", name)) for name in _FORCED_DEMUXERS)
    return (RemuxPlan("auto", ()), *forced, raw_remux_plan(container, frame_rate))


def build_remux_command(
    ffmpeg: Path, source: Path, output: Path, input_args: tuple[str, ...], copytb: bool
) -> list[str]:
    timebase = ["-copytb", "1"] if copytb else []
    return [
        str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-y",
        "-fflags", "+genpts", *input_args, "-i", str(source),
        # Matroska rechaza streams de datos (p. ej. privados del PS de Hikvision); el original verificado los conserva.
        "-map", "0", "-map", "-0:d?", "-c", "copy", *timebase, str(output),
    ]


def without_paths(detail: str, *paths: Path) -> str:
    # El detalle va al informe y al paquete: sin rutas de la maquina, solo nombres de archivo.
    for path in paths:
        detail = detail.replace(str(path), path.name)
    return detail


def skipped_attempt(plan: RemuxPlan) -> RemuxAttempt:
    return RemuxAttempt(plan.label, plan.input_args, ok=False, detail=plan.skip_reason or "")


def remux_outcome(output: Path, stderr: bytes, returncode: int) -> tuple[bool, str]:
    if returncode != 0:
        return False, failure_detail(stderr, returncode)
    if not is_non_empty_file(output):
        return False, "ffmpeg exited cleanly but produced no output"
    return True, "ok"


async def run_remux_attempt(
    tools: MediaTools, plan: RemuxPlan, source: Path, output: Path, copytb: bool
) -> RemuxAttempt:
    if plan.skip_reason is not None:
        return skipped_attempt(plan)
    command = build_remux_command(tools.ffmpeg, source, output, plan.input_args, copytb)
    _, stderr, returncode = await tools.run(command, tools.timeout)
    ok, detail = remux_outcome(output, stderr, returncode)
    return RemuxAttempt(plan.label, plan.input_args, ok=ok, detail=without_paths(detail, source, output))


async def remux_working_copy(
    tools: MediaTools,
    source: Path,
    session_dir: Path,
    container: ContainerGuess,
    frame_rate: str | None,
    copytb: bool,
) -> WorkingCopy:
    output = session_dir / WORK_COPY_NAME
    attempts: list[RemuxAttempt] = []
    for plan in remux_plans(container, frame_rate):
        attempts.append(await run_remux_attempt(tools, plan, source, output, copytb))
        if attempts[-1].ok:
            return WorkingCopy(output, sha256_file(output), plan.label, tuple(attempts), copytb)
    output.unlink(missing_ok=True)
    raise RemuxFailed(tuple(attempts))


# --- Prueba de decodificacion ---


@dataclass(frozen=True)
class DecodeCheck:
    frames_decoded: int
    error_lines: int
    decode_failed: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "framesDecoded": self.frames_decoded,
            "errorLines": self.error_lines,
            "decodeFailed": self.decode_failed,
        }


def build_decode_check_command(ffmpeg: Path, video: Path) -> list[str]:
    return [
        str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-v", "error",
        "-i", str(video), "-map", "0:v:0", "-frames:v", str(DECODE_CHECK_FRAMES),
        "-progress", "pipe:1", "-f", "null", "-",
    ]


def parse_progress_frames(stdout: bytes) -> int:
    lines = stdout.decode("utf-8", errors="replace").splitlines()
    counts = [line.removeprefix("frame=").strip() for line in lines if line.startswith("frame=")]
    return int(counts[-1]) if counts and counts[-1].isdigit() else 0


def count_error_lines(stderr: bytes) -> int:
    return sum(1 for line in stderr.decode("utf-8", errors="replace").splitlines() if line.strip())


def is_decode_failed(frames_decoded: int, error_lines: int) -> bool:
    return frames_decoded == 0 or error_lines > frames_decoded * DECODE_ERROR_RATIO


async def check_decodable(tools: MediaTools, video: Path) -> DecodeCheck:
    # El codigo de salida no decide: con un archivo cifrado ffmpeg puede salir en 0; deciden los cuadros.
    stdout, stderr, _ = await tools.run(build_decode_check_command(tools.ffmpeg, video), tools.timeout)
    frames, errors = parse_progress_frames(stdout), count_error_lines(stderr)
    return DecodeCheck(frames_decoded=frames, error_lines=errors, decode_failed=is_decode_failed(frames, errors))


# --- Indice de cuadros ---


@dataclass(frozen=True)
class FrameIndex:
    frames: tuple[FrameEntry, ...]
    summary: FrameIndexSummary
    csv_path: Path
    csv_sha256: str


def build_frame_index_command(ffprobe: Path, video: Path) -> list[str]:
    return [
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries", FRAME_INDEX_ENTRIES, "-of", "csv=p=0:nk=0", str(video),
    ]


def write_frame_index(session_dir: Path, frames: Sequence[FrameEntry]) -> tuple[Path, str]:
    path = session_dir / FRAME_INDEX_NAME
    path.write_text(frame_index_csv(frames), encoding="utf-8", newline="\n")
    return path, sha256_file(path)


async def build_frame_index(tools: MediaTools, video: Path, session_dir: Path) -> FrameIndex:
    stdout, stderr, returncode = await tools.run(build_frame_index_command(tools.ffprobe, video), tools.timeout)
    if returncode != 0:
        raise IngestError("frame_index", failure_detail(stderr, returncode))
    frames = parse_frame_index(stdout.decode("utf-8", errors="replace"))
    csv_path, csv_sha256 = write_frame_index(session_dir, frames)
    return FrameIndex(frames, summarize_frame_index(frames), csv_path, csv_sha256)


# --- Modo lite, video y audio ---


@dataclass(frozen=True)
class LiteAspect:
    stored_size: tuple[int, int]
    sar: Fraction

    @property
    def display_size(self) -> tuple[int, int]:
        width, height = self.stored_size
        return int(width * self.sar), height

    def to_json(self) -> dict[str, Any]:
        return {
            "storedSize": list(self.stored_size),
            "displaySize": list(self.display_size),
            "sar": f"{self.sar.numerator}:{self.sar.denominator}",
            "filter": f"setsar={self.sar}",
        }


def lite_aspect(width: int, height: int) -> LiteAspect | None:
    sar = LITE_STORAGE_SIZES.get((width, height))
    return LiteAspect((width, height), sar) if sar is not None else None


@dataclass(frozen=True)
class VideoStream:
    codec: str
    width: int
    height: int
    header_rate: str | None

    def to_json(self) -> dict[str, Any]:
        return {"codec": self.codec, "width": self.width, "height": self.height, "headerRate": self.header_rate}


def video_stream(probe: dict[str, Any]) -> VideoStream | None:
    stream = first_stream(probe, "video")
    if stream is None:
        return None
    return VideoStream(
        codec=str(stream.get("codec_name", "unknown")),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        header_rate=header_frame_rate(probe),
    )


@dataclass(frozen=True)
class AudioTrack:
    codec: str
    sample_rate: int | None
    channels: int | None
    family: AudioFamily

    def to_json(self) -> dict[str, Any]:
        return {"codec": self.codec, "sampleRate": self.sample_rate, "channels": self.channels, "family": self.family}


def audio_family(codec: str) -> AudioFamily:
    if codec in _G711_CODECS:
        return "g711"
    return "aac" if codec == "aac" else "other"


def audio_track(stream: dict[str, Any]) -> AudioTrack:
    codec = str(stream.get("codec_name", "unknown"))
    return AudioTrack(
        codec=codec,
        sample_rate=optional_int(str(stream.get("sample_rate", ""))),
        channels=optional_int(str(stream.get("channels", ""))),
        family=audio_family(codec),
    )


def audio_tracks(probe: dict[str, Any]) -> tuple[AudioTrack, ...]:
    return tuple(audio_track(s) for s in probe.get("streams", []) if s.get("codec_type") == "audio")


def audio_codec_for_lane(lane: Lane) -> str:
    # En CCTV el audio nunca se "mejora": el carril clasico lo decodifica sin perdida y el visual va a AAC como hoy.
    return CLASSIC_AUDIO_CODEC if lane == "classic" else VISUAL_AUDIO_CODEC


# --- Orquestacion de la parte 2 ---


@dataclass(frozen=True)
class WorkingIngest:
    source_probe: dict[str, Any]
    working_copy: WorkingCopy
    video: VideoStream
    audio: tuple[AudioTrack, ...]
    decode: DecodeCheck
    index: FrameIndex | None
    lite: LiteAspect | None

    @property
    def warnings(self) -> tuple[str, ...]:
        checks = ((self.decode.decode_failed, UNDECODABLE_WARNING), (self.lite is not None, LITE_WARNING))
        return tuple(key for triggered, key in checks if triggered)

    def to_json(self) -> dict[str, Any]:
        return {
            "workingCopy": self.working_copy.to_json(),
            "video": self.video.to_json(),
            "audio": [track.to_json() for track in self.audio],
            "decode": self.decode.to_json(),
            "decodeFailed": self.decode.decode_failed,
            "frameIndex": None if self.index is None else self.index_json(self.index),
            "lite": None if self.lite is None else self.lite.to_json(),
            "warnings": list(self.warnings),
        }

    @staticmethod
    def index_json(index: FrameIndex) -> dict[str, Any]:
        return {"csvSha256": index.csv_sha256, **index.summary.to_json()}


def remux_frame_rate(probe: dict[str, Any], frame_rate: str | None) -> str | None:
    return validate_frame_rate(frame_rate) if frame_rate is not None else header_frame_rate(probe)


async def probe_working_video(tools: MediaTools, work: Path) -> tuple[dict[str, Any], VideoStream]:
    probe = await probe_media(tools, work)
    video = video_stream(probe)
    if video is None:
        raise NoVideoStream()
    return probe, video


async def index_if_decodable(
    tools: MediaTools, work: Path, session_dir: Path, decode: DecodeCheck
) -> FrameIndex | None:
    return None if decode.decode_failed else await build_frame_index(tools, work, session_dir)


async def ingest_working_copy(
    tools: MediaTools,
    upload: Path,
    session_dir: Path,
    container: ContainerGuess,
    frame_rate: str | None = None,
) -> WorkingIngest:
    source_probe = await probe_source(tools, upload, container)
    rate = remux_frame_rate(source_probe, frame_rate)
    working = await remux_working_copy(tools, upload, session_dir, container, rate, header_suggests_vfr(source_probe))
    work_probe, video = await probe_working_video(tools, working.path)
    decode = await check_decodable(tools, working.path)
    return WorkingIngest(
        source_probe=source_probe,
        working_copy=working,
        video=video,
        audio=audio_tracks(work_probe),
        decode=decode,
        index=await index_if_decodable(tools, working.path, session_dir, decode),
        lite=lite_aspect(video.width, video.height),
    )


def index_facts(index: FrameIndex | None) -> dict[str, Any]:
    if index is None:
        return {}
    summary = index.summary
    return {
        "frame_count": summary.frame_count,
        "measured_fps": summary.measured_fps,
        "is_vfr": summary.is_vfr,
        "probable_duplicates": summary.probable_duplicates,
    }


def source_facts(ingest: WorkingIngest) -> SourceFacts:
    return SourceFacts(
        width=ingest.video.width,
        height=ingest.video.height,
        is_lite=ingest.lite is not None,
        **index_facts(ingest.index),
    )
