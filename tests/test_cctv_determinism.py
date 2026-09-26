from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.config import Settings
from app.services import cctv_clarify_runner as runner
from app.services.cctv_chain import OPEN_GATES, STABILIZE_GATE, steps_from_request
from app.services.cctv_ingest import MediaTools, WorkingIngest, ingest_working_copy
from app.services.cctv_presets import PresetContext, preset_steps
from app.services.ffmpeg_capabilities import cached_capabilities
from app.services.ffmpeg_filters import FrameGeometry
from app.services.media_signature import sniff_file
from ffmpeg_support import needs_ffmpeg

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "ffmpeg" / "cctv_determinism_golden.json"
PRESETS = ("day", "night_ir", "analog", "low_res")
CLIP_FRAMES = 75
OSD_BOX = (8, 8, 128, 24)
X264_THREADS = 4
FFV1_SLICES = 4
MANY_THREADS = 12
STATIC_VIDEO = "testsrc2=size=352x288:rate=50,noise=alls=16:allf=t+u,interlace=scan=tff"
# Camara que tiembla: el recorte se mueve un poco en cada cuadro antes de entrelazar.
SHAKY_VIDEO = (
    "testsrc2=size=384x320:rate=50,crop=w=352:h=288:x='16+8*sin(n*0.45)':y='16+6*cos(n*0.65)',"
    "noise=alls=16:allf=t+u,interlace=scan=tff"
)
STABILIZE_RAW = {"id": "stabilize", "params": {"shakiness": 5, "smoothing": 10}}
# Cabecera IMKH de 40 bytes armada con la descripcion publica del formato:
# firma, version, codec (H.264), audio y resolucion; el resto en cero.
IMKH_HEADER = (
    b"IMKH"
    + (0x0102).to_bytes(2, "little")
    + (0x0100).to_bytes(2, "little")
    + (0x7111).to_bytes(2, "little")
    + (8000).to_bytes(4, "little")
    + (352).to_bytes(2, "little")
    + (288).to_bytes(2, "little")
).ljust(40, b"\x00")


def ffmpeg_binary() -> Path:
    return Settings().ffmpeg_binary_path


def synthetic_command(output: Path, *container: str, video: str = STATIC_VIDEO) -> list[str]:
    return [
        str(ffmpeg_binary()), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
        "-t", "3", "-threads", "1", "-c:v", "libx264", "-x264-params", "threads=1", "-flags", "+ildct+ilme",
        "-b:v", "150k", "-g", "25", "-bf", "0", *container, "-shortest", str(output),
    ]  # fmt: skip


def make_mkv_clip(folder: Path, video: str = STATIC_VIDEO) -> Path:
    clip = folder / "camera.mkv"
    subprocess.run(synthetic_command(clip, "-c:a", "pcm_alaw", video=video), check=True, capture_output=True)
    return clip


def make_imkh_clip(folder: Path) -> Path:
    program_stream = folder / "camera.mpg"
    subprocess.run(synthetic_command(program_stream, "-c:a", "mp2", "-f", "vob"), check=True, capture_output=True)
    export = folder / "export.mp4"
    export.write_bytes(IMKH_HEADER + program_stream.read_bytes())
    return export


async def ingest(folder: Path, clip: Path) -> WorkingIngest:
    settings = Settings()
    tools = MediaTools(ffmpeg=settings.ffmpeg_binary_path, ffprobe=settings.ffprobe_binary_path)
    session = folder / "cctv-session"
    session.mkdir(exist_ok=True)
    ingested = await ingest_working_copy(tools, clip, session, sniff_file(clip))
    assert ingested.index is not None and ingested.index.summary.frame_count == CLIP_FRAMES
    return ingested


def raw_plan(ingested: WorkingIngest, raw: list[dict], boxes: tuple = (OSD_BOX,)) -> runner.ClarifyPlan:
    return runner.ClarifyPlan(
        work=ingested.working_copy.path,
        steps=steps_from_request(raw, "classic", frozenset({STABILIZE_GATE})),
        geometry=FrameGeometry(ingested.video.width, ingested.video.height),
        source_pix_fmt="yuv420p",
        frames=ingested.index.frames,
        has_audio=bool(ingested.audio),
        app_version="0.0.0-test",
        osd_boxes=boxes,
    )


def preset_plan(ingested: WorkingIngest, preset: str, *extra: dict) -> runner.ClarifyPlan:
    return raw_plan(ingested, [*preset_steps(preset, "classic", PresetContext(interlaced=True)), *extra])


def vidstab_plan(ingested: WorkingIngest, preset: str | None) -> runner.ClarifyPlan:
    # Sin preset ni OSD, vidstab lee los cuadros tal como salen del decoder: el caso que cambiaba en cada corrida.
    if preset is None:
        return raw_plan(ingested, [STABILIZE_RAW], boxes=())
    return preset_plan(ingested, preset, STABILIZE_RAW)


def clarify_tools() -> runner.ClarifyTools:
    settings = Settings()
    return runner.ClarifyTools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path)


def framehash(video: Path) -> list[str]:
    command = [str(ffmpeg_binary()), "-v", "error", "-i", str(video), "-map", "0:v:0"]
    command += ["-fps_mode", "passthrough", "-f", "framehash", "-hash", "sha256", "-"]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [line for line in output.splitlines() if line and not line.startswith("#")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RunFingerprint:
    frames: tuple[str, ...]
    analysis_sha256: str
    viewing_sha256: str
    transforms_sha256: str | None = None


def transforms_sha256(result: runner.ClarifyResult) -> str | None:
    detection = result.stabilize_detection
    return None if detection is None else sha256(detection.transforms)


async def clarify_once(folder: Path, plan: runner.ClarifyPlan, filter_threads: int) -> RunFingerprint:
    folder.mkdir()
    threads = runner.ClarifyThreads(filter_threads, FFV1_SLICES, X264_THREADS)
    result = await runner.run_clarify(clarify_tools(), plan, folder, threads)
    assert result.output_frames == result.expected_frames == CLIP_FRAMES
    frames = tuple(framehash(result.analysis))
    return RunFingerprint(frames, sha256(result.analysis), sha256(result.viewing), transforms_sha256(result))


async def three_runs(folder: Path, plan: runner.ClarifyPlan) -> tuple[RunFingerprint, RunFingerprint, RunFingerprint]:
    first = await clarify_once(folder / "run-1", plan, MANY_THREADS)
    second = await clarify_once(folder / "run-2", plan, MANY_THREADS)
    single = await clarify_once(folder / "run-t1", plan, 1)
    return first, second, single


def assert_deterministic(first: RunFingerprint, second: RunFingerprint, single: RunFingerprint) -> None:
    assert len(first.frames) == CLIP_FRAMES
    assert first.frames == second.frames
    assert first.frames == single.frames
    assert first.analysis_sha256 == second.analysis_sha256
    assert first.viewing_sha256 == second.viewing_sha256


@needs_ffmpeg
@pytest.mark.parametrize("preset", PRESETS)
async def test_two_runs_and_one_thread_give_the_same_frames_and_files(tmp_path: Path, preset: str) -> None:
    ingested = await ingest(tmp_path, make_mkv_clip(tmp_path))

    first, second, single = await three_runs(tmp_path, preset_plan(ingested, preset))

    assert_deterministic(first, second, single)
    assert single.viewing_sha256 == first.viewing_sha256


@needs_ffmpeg
async def test_a_hikvision_program_stream_is_deterministic_too(tmp_path: Path) -> None:
    clip = make_imkh_clip(tmp_path)
    assert sniff_file(clip).kind == "hikvision_ps"
    ingested = await ingest(tmp_path, clip)

    first, second, single = await three_runs(tmp_path, preset_plan(ingested, "analog"))

    assert_deterministic(first, second, single)


def load_golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


async def golden_hashes(folder: Path) -> dict[str, str]:
    ingested = await ingest(folder, make_mkv_clip(folder))
    runs = {preset: await clarify_once(folder / preset, preset_plan(ingested, preset), MANY_THREADS) for preset in PRESETS}
    return {preset: run.analysis_sha256 for preset, run in runs.items()}


def record_golden(folder: Path) -> None:
    caps = cached_capabilities(ffmpeg_binary())
    golden = {
        "ffmpegVersion": caps.version,
        "ffmpegSha256": caps.binary_sha256,
        "cpuExtensions": list(caps.cpu_extensions),
        "analysisSha256": asyncio.run(golden_hashes(folder)),
    }
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")


def golden_mismatch(golden: dict) -> str | None:
    caps = cached_capabilities(ffmpeg_binary())
    if caps.binary_sha256 != golden["ffmpegSha256"]:
        return f"golden hashes are for {golden['ffmpegVersion']} ({golden['ffmpegSha256'][:12]}), not {caps.version}"
    if list(caps.cpu_extensions) != golden["cpuExtensions"]:
        return f"golden hashes are for CPU extensions {golden['cpuExtensions']}, not {list(caps.cpu_extensions)}"
    return None


@needs_ffmpeg
@pytest.mark.parametrize("preset", PRESETS)
async def test_analysis_copy_matches_the_golden_hash_of_this_build(tmp_path: Path, preset: str) -> None:
    golden = load_golden()
    reason = golden_mismatch(golden)
    if reason is not None:
        pytest.skip(reason)
    ingested = await ingest(tmp_path, make_mkv_clip(tmp_path))

    fingerprint = await clarify_once(tmp_path / "golden", preset_plan(ingested, preset), MANY_THREADS)

    assert fingerprint.analysis_sha256 == golden["analysisSha256"][preset]


# --- vidstab (P4-STABILIZE): entra al carril clasico solo si esto pasa ---


def differing_frames(first: RunFingerprint, other: RunFingerprint) -> int:
    return sum(a != b for a, b in zip(first.frames, other.frames, strict=True))


def vidstab_nondeterminism(first: RunFingerprint, second: RunFingerprint, single: RunFingerprint) -> str | None:
    repeated, threaded = differing_frames(first, second), differing_frames(first, single)
    motion_same = first.transforms_sha256 == second.transforms_sha256 == single.transforms_sha256
    files_same = first.analysis_sha256 == second.analysis_sha256 and first.viewing_sha256 == second.viewing_sha256
    if not repeated and not threaded and motion_same and files_same:
        return None
    motion = "is identical" if motion_same else "differs"
    return (
        f"vidstab is not deterministic with {cached_capabilities(ffmpeg_binary()).version}: the motion file {motion} "
        f"across runs; {repeated} of {len(first.frames)} frames differ between two identical runs "
        f"and {threaded} between {MANY_THREADS} threads and 1"
    )


@needs_ffmpeg
async def test_vidstab_detection_on_one_openmp_thread_writes_the_same_motion_file(tmp_path: Path) -> None:
    ingested = await ingest(tmp_path, make_mkv_clip(tmp_path, SHAKY_VIDEO))
    plan = vidstab_plan(ingested, None)
    digests = []
    for name, filter_threads in (("many", MANY_THREADS), ("again", MANY_THREADS), ("one", 1)):
        folder = tmp_path / name
        folder.mkdir()
        threads = runner.ClarifyThreads(filter_threads, FFV1_SLICES, X264_THREADS)
        detection = await runner.run_detect_pass(clarify_tools(), plan, folder, threads, lambda stage, fraction: None)
        digests.append(sha256(detection.transforms))

    assert len(set(digests)) == 1
    assert (tmp_path / "one" / runner.TRANSFORMS_NAME).read_bytes().startswith(b"VID.STAB 1")


@needs_ffmpeg
@pytest.mark.parametrize("preset", [None, "analog", "night_ir"])
async def test_vidstab_two_runs_and_one_thread_give_the_same_frames_and_files(tmp_path: Path, preset: str | None) -> None:
    ingested = await ingest(tmp_path, make_mkv_clip(tmp_path, SHAKY_VIDEO))

    runs = await three_runs(tmp_path, vidstab_plan(ingested, preset))

    reason = vidstab_nondeterminism(*runs)
    if reason is not None and STABILIZE_GATE not in OPEN_GATES:
        pytest.skip(f"{reason}; the stabilize step stays closed (P4-STABILIZE)")
    assert reason is None, reason
    assert_deterministic(*runs)
