"""Riesgo latente GMFSS + reescalado en DML (P3-GPU-2, spec §7).

El stream pipeline completo (decode -> GMFSS -> upscale -> encode) corre la etapa GMFSS
y la de reescalado en DOS hilos del FramePipeline, cada uno con sus sesiones DML sobre
el mismo adaptador, mas el splat OpenCL de GMFSS. Si dos Run concurrentes tiraran el
device (887A0005), el job caeria en silencio al camino clasico (streamPipelineFallback)
o fallaria.

Dos casos: el job real por VideoUpscaler.run (en el pipeline real las etapas se solapan
poco, porque GMFSS es el cuello y el reescalado espera frames) y el caso minimo que
fuerza el solape: la etapa GMFSS y el closure de reescalado del modo existente, cada
uno en su hilo, en bucle a la vez durante un tiempo fijo.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.devices_service import DevicesService
from app.services.engines import gmfss_engine as gmfss_module
from app.services.engines.gmfss import softsplat_cl
from app.services.engines.gmfss_engine import GmfssEngine, GmfssStreamStage
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.media_tools import MediaTools
from app.services.model_registry import ModelRegistry
from app.services.video_upscaler import VideoUpscaler

pytestmark = pytest.mark.gpu

DEVICE = "dml:0"
MODEL = "realesr-animevideov3-x2"
SCALE = 2
FPS_MULTIPLIER = 2
RATE = 24
SECONDS_ENV = "UPFLOW_GMFSS_GPU_SECONDS"
REPEATS_ENV = "UPFLOW_GMFSS_GPU_REPEATS"
RESULTS_ENV = "UPFLOW_GMFSS_GPU_RESULTS"
STRESS_SECONDS_ENV = "UPFLOW_GMFSS_GPU_STRESS_SECONDS"
DEFAULT_SECONDS = 2
DEFAULT_REPEATS = 1
DEFAULT_STRESS_SECONDS = 60
STRESS_SOURCE_FRAMES = 100_000
# Sin solape el caso minimo no probaria nada: al menos una decima del tiempo de GMFSS
# en Run tiene que coincidir con Runs del reescalado.
MIN_STRESS_OVERLAP_SHARE = 0.1


@dataclass
class StageClock:
    intervals: dict[str, list[tuple[float, float]]] = field(default_factory=lambda: {"gmfss": [], "upscale": []})
    lock: threading.Lock = field(default_factory=threading.Lock)

    def timed(self, stage: str, fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                with self.lock:
                    self.intervals[stage].append((start, time.perf_counter()))

        return wrapper

    def busy(self, stage: str) -> float:
        return sum(end - start for start, end in self.intervals[stage])

    def overlap(self) -> float:
        return sum(
            max(0.0, min(g_end, u_end) - max(g_start, u_start))
            for g_start, g_end in self.intervals["gmfss"]
            for u_start, u_end in self.intervals["upscale"]
        )


@dataclass(frozen=True, slots=True)
class Stack:
    settings: Settings
    upscaler: VideoUpscaler


def require_assets(settings: Settings) -> None:
    if not settings.gmfss_available():
        pytest.skip(f"GMFSS graphs missing in {settings.gmfss_model_dir_path}")
    onnx_file = settings.builtin_onnx_path / f"{MODEL}-uint8.onnx"
    if not onnx_file.is_file():
        pytest.skip(f"{onnx_file} missing")
    if not settings.ffmpeg_binary_path.is_file():
        pytest.skip("ffmpeg missing")


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    runtime = tmp_path_factory.mktemp("gmfss-upscale-gpu")
    settings = Settings(_env_file=None, RUNTIME_DIR=str(runtime / "runtime"), ENABLE_GMFSS=True)
    require_assets(settings)
    coordinator = GpuSessionCoordinator()
    devices = DevicesService(settings)
    gmfss = GmfssEngine(settings, coordinator)
    onnx_video = OnnxVideoUpscaler(settings, ModelRegistry(settings), devices, coordinator)
    upscaler = VideoUpscaler(
        settings,
        RealEsrganNcnnEngine(settings),
        MediaTools(settings),
        gmfss_engine=gmfss,
        onnx_video_engine=onnx_video,
        model_registry=ModelRegistry(settings),
        devices=devices,
    )
    yield Stack(settings, upscaler)
    gmfss.release_device(DEVICE)
    onnx_video.release_device(DEVICE)


@pytest.fixture
def clock(stack: Stack, monkeypatch: pytest.MonkeyPatch) -> StageClock:
    clock = StageClock()
    original_runner = gmfss_module._graph_runner
    monkeypatch.setattr(
        gmfss_module, "_graph_runner", lambda sessions: clock.timed("gmfss", original_runner(sessions))
    )
    engine = stack.upscaler.onnx_video_engine
    original_builder = engine.build_frame_upscaler
    monkeypatch.setattr(
        engine,
        "build_frame_upscaler",
        lambda *args, **kwargs: clock.timed("upscale", original_builder(*args, **kwargs)),
    )
    return clock


def make_motion_clip(settings: Settings, directory: Path, size: tuple[int, int], seconds: float) -> Path:
    width, height = size
    clip = directory / f"motion_{width}x{height}.mp4"
    graph = f"testsrc2=size={width}x{height}:rate={RATE},format=yuv420p"
    command = [
        str(settings.ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i", graph,
        "-t", str(seconds), "-c:v", "libx264", "-crf", "18", "-an", str(clip),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return clip


def interp_upscale_job(clip: Path) -> VideoUpscaleJob:
    return VideoUpscaleJob(
        source_path=clip,
        original_filename=clip.name,
        model_name=MODEL,
        scale=SCALE,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        fps_multiplier=FPS_MULTIPLIER,
        interp_engine="gmfss",
        device=DEVICE,
        backend="onnx",
    )


def probe_output(settings: Settings, video: Path) -> dict:
    command = [
        str(settings.ffprobe_binary_path), "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets,avg_frame_rate", "-of", "json", str(video),
    ]  # fmt: skip
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)["streams"][0]


def record_result(result: dict) -> None:
    target = os.environ.get(RESULTS_ENV)
    if target:
        with Path(target).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")


def seconds_per_clip() -> float:
    return float(os.environ.get(SECONDS_ENV, DEFAULT_SECONDS))


def repeats() -> list[int]:
    return list(range(int(os.environ.get(REPEATS_ENV, DEFAULT_REPEATS))))


def summarize(
    size: tuple[int, int], repeat: int, job: VideoUpscaleJob, clock: StageClock, wall: float, engine: OnnxVideoUpscaler
) -> dict:
    gmfss_busy = clock.busy("gmfss")
    return {
        "size": f"{size[0]}x{size[1]}",
        "repeat": repeat,
        "wall_s": round(wall, 2),
        "gmfss_runs": len(clock.intervals["gmfss"]),
        "gmfss_busy_s": round(gmfss_busy, 2),
        "upscale_frames": len(clock.intervals["upscale"]),
        "upscale_busy_s": round(clock.busy("upscale"), 2),
        "overlap_s": round(clock.overlap(), 2),
        "overlap_share_of_gmfss": round(clock.overlap() / gmfss_busy, 3) if gmfss_busy else 0.0,
        "stream_pipeline": job.metadata.get("streamPipeline"),
        "fallback": job.metadata.get("streamPipelineFallback"),
        "encoder": job.metadata.get("videoEncoder"),
        "upscale_precision": engine.last_precision,
        "upscale_tiled": engine.last_tiled,
        "upscale_iobinding_failed": engine._iobinding_warned,
        "splat_on_opencl": not softsplat_cl._gpu_unavailable,
    }


@pytest.mark.parametrize("repeat", repeats())
@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080)], ids=["720p", "1080p"])
async def test_gmfss_and_upscale_stages_run_concurrently_on_dml_without_device_removal(
    stack: Stack, clock: StageClock, tmp_path: Path, size: tuple[int, int], repeat: int
) -> None:
    settings = stack.settings
    clip = make_motion_clip(settings, tmp_path, size, seconds_per_clip())
    job = interp_upscale_job(clip)
    source_frames = round(seconds_per_clip() * RATE)

    start = time.perf_counter()
    output = await stack.upscaler.run(job, fps_multiplier=FPS_MULTIPLIER)
    result = summarize(
        size, repeat, job, clock, time.perf_counter() - start, stack.upscaler.onnx_video_engine
    )
    record_result(result)

    assert "streamPipelineFallback" not in job.metadata, result
    assert job.metadata.get("streamPipeline") is True, result
    stream = probe_output(settings, output)
    assert (stream["width"], stream["height"]) == (size[0] * SCALE, size[1] * SCALE)
    assert int(stream["nb_read_packets"]) == source_frames * FPS_MULTIPLIER
    assert result["upscale_frames"] == source_frames * FPS_MULTIPLIER
    assert result["overlap_s"] > 0, result


def random_frames(size: tuple[int, int], count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    width, height = size
    return [rng.integers(0, 256, (1, height, width, 3), dtype=np.uint8) for _ in range(count)]


def loop_until(deadline: float, step: Callable[[int], None], errors: list[BaseException], stop: threading.Event) -> int:
    done = 0
    try:
        while time.perf_counter() < deadline and not stop.is_set():
            step(done)
            done += 1
    except BaseException as exc:  # noqa: BLE001 - el error del otro hilo tambien tiene que cortar este
        errors.append(exc)
        stop.set()
    return done


def gmfss_step(stage: GmfssStreamStage, frames: list[np.ndarray]) -> Callable[[int], None]:
    def step(index: int) -> None:
        for _ in stage.process(frames[index % len(frames)]):
            pass

    return step


def upscale_step(upscale: Callable[[np.ndarray], np.ndarray], frames: list[np.ndarray]) -> Callable[[int], None]:
    def step(index: int) -> None:
        upscale(frames[index % len(frames)])

    return step


def run_in_parallel(
    steps: dict[str, Callable[[int], None]], seconds: float
) -> tuple[dict[str, int], list[BaseException]]:
    errors: list[BaseException] = []
    stop = threading.Event()
    deadline = time.perf_counter() + seconds
    counts: dict[str, int] = {}

    def worker(name: str) -> None:
        counts[name] = loop_until(deadline, steps[name], errors, stop)

    threads = [threading.Thread(target=worker, args=(name,), name=f"p3-gpu-2-{name}") for name in steps]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return counts, errors


@pytest.mark.parametrize("size", [(1280, 720), (1920, 1080)], ids=["720p", "1080p"])
def test_forced_overlap_of_gmfss_and_upscale_runs_keeps_the_dml_device_alive(
    stack: Stack, clock: StageClock, size: tuple[int, int]
) -> None:
    gmfss = stack.upscaler.gmfss_engine
    stage = gmfss.build_stream_stage(STRESS_SOURCE_FRAMES, STRESS_SOURCE_FRAMES * FPS_MULTIPLIER, DEVICE)
    upscale = stack.upscaler.onnx_video_engine.build_frame_upscaler(MODEL, DEVICE, SCALE)
    frames = random_frames(size, 4)
    seconds = float(os.environ.get(STRESS_SECONDS_ENV, DEFAULT_STRESS_SECONDS))

    counts, errors = run_in_parallel(
        {"gmfss": gmfss_step(stage, frames), "upscale": upscale_step(upscale, frames)}, seconds
    )

    result = {
        "case": "forced-overlap",
        "size": f"{size[0]}x{size[1]}",
        "seconds": seconds,
        "gmfss_source_frames": counts.get("gmfss"),
        "gmfss_runs": len(clock.intervals["gmfss"]),
        "gmfss_busy_s": round(clock.busy("gmfss"), 2),
        "upscale_frames": counts.get("upscale"),
        "upscale_busy_s": round(clock.busy("upscale"), 2),
        "overlap_s": round(clock.overlap(), 2),
        "overlap_share_of_gmfss": round(clock.overlap() / max(clock.busy("gmfss"), 1e-9), 3),
        "errors": [repr(error) for error in errors],
    }
    record_result(result)
    assert errors == [], result
    assert result["overlap_share_of_gmfss"] >= MIN_STRESS_OVERLAP_SHARE, result
