"""Smoke DML del carril IA de CCTV (P3-GPU-1, spec §4.7 y §4.15).

Corre el CctvEnhanceRunner real (VideoUpscaler + PhotoRestoreEngine + OnnxVideoUpscaler)
sobre un clip sintetico tipo DVR: ruido de sensor, bloques de MPEG-4 y un OSD. Mide
segundos por cuadro contra la tabla derivada de §4.15, el parpadeo en zonas estaticas
contra la entrada, el rotulo, el OSD y la cancelacion. RESTORE_MODELS sigue vacio hasta
publicar (D2): la spec de drunet-deblock-color-u8 se arma con models.lock.json del
directorio de modelos (UPFLOW_RESTORE_MODELS_DIR) y los valores medidos en P0-GPU.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import datetime
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import pytest

from app.config import Settings
from app.models import CctvOptions, CctvStep, VideoUpscaleJob
from app.services.cctv_ingest import ingest_source, make_verified_copy, write_source_record
from app.services.cctv_session import cctv_job_dir
from app.services.devices_service import DevicesService
from app.services.engines.frame_restorer import FrameRestorer
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.ffmpeg_filters import escape_filter_path
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.label_band import FONT_PATH
from app.services.media_tools import MediaTools
from app.services.model_registry import ModelRegistry
from app.services.restore_models import RestoreModelSpec
from app.services.video_upscaler import VideoUpscaler

pytestmark = pytest.mark.gpu

MODELS_DIR_ENV = "UPFLOW_RESTORE_MODELS_DIR"
SECONDS_ENV = "UPFLOW_CCTV_GPU_SECONDS"
RESULTS_ENV = "UPFLOW_CCTV_GPU_RESULTS"
DEVICE = "dml:0"
MODEL_ID = "drunet-deblock-color-u8"
LOCK_NAME = "models.lock.json"
RATE = 25
DEFAULT_SECONDS = 8
STRENGTH = 40
# Rango derivado de §4.15 (s por cuadro, DRUNet fp16 de cuadro entero); el criterio acepta 2x a cada lado.
DERIVED_S_PER_FRAME = {(1920, 1080): (0.29, 0.6), (960, 1080): (0.15, 0.3)}
TOLERANCE = 2.0
OSD_MARGIN = 8
MIN_OSD_PSNR_DB = 38.0


@dataclass(frozen=True, slots=True)
class Scene:
    width: int
    height: int

    @property
    def osd_box(self) -> tuple[int, int, int, int]:
        return OSD_MARGIN, OSD_MARGIN, self.width * 3 // 8, self.height // 18

    @property
    def static_zone(self) -> tuple[int, int, int, int]:
        # Abajo a la izquierda: lejos del OSD, de la caja que se mueve (mitad superior) y de la marca del rotulo.
        return self.width // 8, self.height * 5 // 8, self.width // 4, self.height // 5


def models_dir() -> Path:
    raw = os.environ.get(MODELS_DIR_ENV)
    needed = (f"{MODEL_ID}.onnx", f"{MODEL_ID}-fp16.onnx", LOCK_NAME)
    if not raw or not all((Path(raw) / name).is_file() for name in needed):
        pytest.skip(f"{MODELS_DIR_ENV} must point at a folder with {MODEL_ID}.onnx, its fp16 file and {LOCK_NAME}")
    return Path(raw)


def video_deblock_spec(directory: Path) -> RestoreModelSpec:
    meta = json.loads((directory / LOCK_NAME).read_text(encoding="utf-8"))["models"][MODEL_ID]
    license_fields = ("license_spdx", "license_url", "copyright", "attribution", "data_lineage", "commercial_use")
    source_fields = ("source_url", "source_revision", "source_sha256")
    return RestoreModelSpec(
        id=MODEL_ID,
        name="DRUNet deblock (video, uint8)",
        bundle="core",
        filename=f"{MODEL_ID}.onnx",
        fp16_filename=f"{MODEL_ID}-fp16.onnx",
        **{name: meta[name] for name in (*license_fields, *source_fields)},
        modifications=tuple(meta["modifications"]),
        tile_min=128,
        tile_candidates=(256, 384, 512),
        tile_by_precision={"fp32": 512, "fp16": 512},
        vram_factor=34.16,
    )


@dataclass(frozen=True, slots=True)
class Stack:
    settings: Settings
    upscaler: VideoUpscaler


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    directory = models_dir()
    runtime = tmp_path_factory.mktemp("cctv-ai-gpu")
    settings = Settings(_env_file=None, RUNTIME_DIR=str(runtime / "runtime"), RESTORE_MODEL_DIR=str(directory))
    coordinator = GpuSessionCoordinator()
    devices = DevicesService(settings)
    engine = PhotoRestoreEngine(settings, coordinator, models={MODEL_ID: video_deblock_spec(directory)})
    onnx_video = OnnxVideoUpscaler(settings, ModelRegistry(settings), devices, coordinator)
    upscaler = VideoUpscaler(
        settings,
        RealEsrganNcnnEngine(settings),
        MediaTools(settings),
        onnx_video_engine=onnx_video,
        devices=devices,
        frame_restorer=FrameRestorer(engine),
    )
    yield Stack(settings, upscaler)
    engine.release_device(DEVICE)


def ffmpeg(settings: Settings) -> str:
    return str(settings.ffmpeg_binary_path)


def make_dvr_clip(settings: Settings, directory: Path, scene: Scene, seconds: float) -> Path:
    clip = directory / f"dvr_{scene.width}x{scene.height}.mp4"
    x, y, w, h = scene.osd_box
    size = f"{scene.width}x{scene.height}"
    font = escape_filter_path(FONT_PATH)
    graph = (
        f"smptehdbars=size={size}:rate={RATE},drawgrid=w=64:h=64:t=2:c=white@0.5,"
        f"noise=alls=30:allf=t,"
        f"drawtext=fontfile={font}:text=MOVING:x='mod(t*120,W-tw)':y=H/6:fontsize=60:"
        f"fontcolor=orange:box=1:boxcolor=orange,"
        f"drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill,"
        f"drawtext=fontfile={font}:text='CAM 07 %{{pts\\:hms}}':"
        f"x={x + 6}:y={y + 4}:fontsize={h - 10}:fontcolor=white,format=yuv420p"
    )
    command = [
        ffmpeg(settings), "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i", graph, "-t", str(seconds),
        "-c:v", "mpeg4", "-q:v", "20", "-g", "50", "-an", str(clip),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return clip


def enhance_job(clip: Path, scene: Scene) -> VideoUpscaleJob:
    cctv = CctvOptions(
        task="enhance",
        session_token="gpusmoke0token1",
        steps=(CctvStep("ai_deblock", {"strength": STRENGTH}),),
        osd_boxes=(scene.osd_box,),
        osd_boxes_confirmed=True,
    )
    return VideoUpscaleJob(
        source_path=clip,
        original_filename=clip.name,
        model_name="cctv-enhance",
        scale=1,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        device=DEVICE,
        video_encoder="software",
        cctv=cctv,
        metadata={"cctv": {"task": "enhance", "lane": "ai", "sourceSha256": "admitted"}},
    )


def admit(settings: Settings, job: VideoUpscaleJob, clip: Path) -> Path:
    job_dir = cctv_job_dir(settings.outputs_path, job.id)
    record = ingest_source(clip, clip.name)
    make_verified_copy(clip, record, job_dir)
    write_source_record(job_dir, record)
    return job_dir


def gray_frames(settings: Settings, video: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    command = [
        ffmpeg(settings), "-v", "error", "-i", str(video), "-vf", f"crop={w}:{h}:{x}:{y},format=gray",
        "-f", "rawvideo", "-",
    ]  # fmt: skip
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w).astype(np.float32)


def temporal_flicker(frames: np.ndarray) -> float:
    return float(np.abs(np.diff(frames, axis=0)).mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def inner(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    # El borde de 1 px de la caja se mezcla al 50 % con el procesado (osd_edge_mask).
    x, y, w, h = box
    return x + 1, y + 1, w - 2, h - 2


def stream_seconds(job_dir: Path) -> float:
    report = json.loads((job_dir / "report.json").read_text(encoding="utf-8"))
    frames = next(p for p in report["processes"] if p["label"].startswith("AI lane frames"))
    start, end = (datetime.fromisoformat(frames[key]["utc"].replace("Z", "+00:00")) for key in ("startedAt", "endedAt"))
    return (end - start).total_seconds()


def ffmpeg_children() -> list[str]:
    return [child.name() for child in psutil.Process().children(recursive=True) if "ffmpeg" in child.name().lower()]


def record_result(result: dict) -> None:
    target = os.environ.get(RESULTS_ENV)
    if target:
        with Path(target).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")


def seconds_per_run() -> float:
    return float(os.environ.get(SECONDS_ENV, DEFAULT_SECONDS))


@pytest.mark.parametrize("size", [(1920, 1080), (960, 1080)], ids=["1080p", "960x1080"])
async def test_ai_deblock_streams_on_dml_within_budget_without_flicker(
    stack: Stack, tmp_path: Path, size: tuple[int, int]
) -> None:
    scene = Scene(*size)
    settings = stack.settings
    clip = make_dvr_clip(settings, tmp_path, scene, seconds_per_run())
    job = enhance_job(clip, scene)
    job_dir = admit(settings, job, clip)

    output = await stack.upscaler.run(job)

    metadata = job.metadata["cctv"]
    restore = metadata["restore"]
    assert (restore["device"], restore["model"], restore["ioBinding"]) == (DEVICE, MODEL_ID, True)
    assert metadata["framesOut"] == metadata["framesIn"] == round(seconds_per_run() * RATE)
    assert [path.name for path in job_dir.rglob("*.png")] == []
    band = metadata["label"]["bandHeight"]
    assert (job.metadata["outputWidth"], job.metadata["outputHeight"]) == (scene.width, scene.height + band)
    source = job_dir / "01_original" / clip.name
    osd_in, osd_out = (gray_frames(settings, video, inner(scene.osd_box)) for video in (source, output))
    zone_in, zone_out = (gray_frames(settings, video, scene.static_zone) for video in (source, output))
    s_per_frame = stream_seconds(job_dir) / metadata["framesIn"]
    low, high = DERIVED_S_PER_FRAME[size]
    result = {
        "size": f"{scene.width}x{scene.height}",
        "frames": metadata["framesIn"],
        "precision": restore["precision"],
        "tile": restore["tile"],
        "s_per_frame": round(s_per_frame, 4),
        "fps": round(1 / s_per_frame, 2),
        "flicker_in": round(temporal_flicker(zone_in), 3),
        "flicker_out": round(temporal_flicker(zone_out), 3),
        "osd_psnr_db": round(psnr(osd_in, osd_out), 2),
        "duplicates_reused": metadata["duplicatesReused"],
    }
    record_result(result)
    assert low / TOLERANCE <= s_per_frame <= high * TOLERANCE, result
    assert result["osd_psnr_db"] >= MIN_OSD_PSNR_DB, result
    assert result["flicker_out"] <= result["flicker_in"], result


async def wait_for_frames(job: VideoUpscaleJob, frames: int, task: asyncio.Task) -> None:
    while (job.metadata.get("framesDone") or 0) < frames and not task.done():
        await asyncio.sleep(0.05)


async def test_cancelling_mid_stream_stops_ffmpeg_and_leaves_the_device_usable(stack: Stack, tmp_path: Path) -> None:
    scene = Scene(1920, 1080)
    settings = stack.settings
    clip = make_dvr_clip(settings, tmp_path, scene, 6)
    job = enhance_job(clip, scene)
    admit(settings, job, clip)
    task = asyncio.ensure_future(stack.upscaler.run(job))
    await wait_for_frames(job, 25, task)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert 25 <= job.metadata["framesDone"] < 150
    assert ffmpeg_children() == []
    assert not (settings.video_work_path / job.id).exists()
    retry = enhance_job(clip, scene)
    admit(settings, retry, clip)
    await stack.upscaler.run(retry)
    assert retry.metadata["cctv"]["framesOut"] == 150
    record_result({"cancel_after_frames": job.metadata["framesDone"], "retry_frames": 150})
