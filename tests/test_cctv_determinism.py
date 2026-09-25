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
from app.services.cctv_chain import steps_from_request
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


def synthetic_command(output: Path, *container: str) -> list[str]:
    video = "testsrc2=size=352x288:rate=50,noise=alls=16:allf=t+u,interlace=scan=tff"
    return [
        str(ffmpeg_binary()), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
        "-t", "3", "-threads", "1", "-c:v", "libx264", "-x264-params", "threads=1", "-flags", "+ildct+ilme",
        "-b:v", "150k", "-g", "25", "-bf", "0", *container, "-shortest", str(output),
    ]  # fmt: skip


def make_mkv_clip(folder: Path) -> Path:
    clip = folder / "camera.mkv"
    subprocess.run(synthetic_command(clip, "-c:a", "pcm_alaw"), check=True, capture_output=True)
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


def preset_plan(ingested: WorkingIngest, preset: str) -> runner.ClarifyPlan:
    raw = preset_steps(preset, "classic", PresetContext(interlaced=True))
    return runner.ClarifyPlan(
        work=ingested.working_copy.path,
        steps=steps_from_request(raw, "classic"),
        geometry=FrameGeometry(ingested.video.width, ingested.video.height),
        source_pix_fmt="yuv420p",
        frames=ingested.index.frames,
        has_audio=bool(ingested.audio),
        app_version="0.0.0-test",
        osd_boxes=(OSD_BOX,),
    )


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


async def clarify_once(folder: Path, plan: runner.ClarifyPlan, filter_threads: int) -> RunFingerprint:
    folder.mkdir()
    threads = runner.ClarifyThreads(filter_threads, FFV1_SLICES, X264_THREADS)
    result = await runner.run_clarify(clarify_tools(), plan, folder, threads)
    assert result.output_frames == result.expected_frames == CLIP_FRAMES
    return RunFingerprint(tuple(framehash(result.analysis)), sha256(result.analysis), sha256(result.viewing))


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
