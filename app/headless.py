"""Modo headless de Upflow: reescalado en proceso, sin servidor.

Lo consumen la CLI (`upflow`) y el servidor MCP en modo in-process. Arma los
mismos servicios que el lifespan de app.main pero sin colas, workers ni sweeper:
un job se resuelve y corre en el hilo que llama (JobManager.run_inline), con el
mismo motor, la misma reduccion a la escala pedida y la misma metadata
`effective` que un job encolado por la API.

Contrato de salida (una sola linea JSON en la CLI con --json):
    {"ok": true, "output": "...", "width": W, "height": H, "model": "...",
     "device": "dml:0", "scale": 2, "nativeScale": 4, "resized": true,
     "tile": {"size": 0, "overlap": 10, "meaning": "..."}, "format": "webp",
     "seconds": 5.1}
Codigos de salida: 0 ok, 2 argumentos, 3 modelo no instalado, 4 dispositivo,
5 fallo de inferencia/operacion.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings, get_settings
from app.models import UpscaleJob
from app.services.compat_strategy import strategy_for
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.onnx_upscaler import OnnxUpscaler
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.health_report import build_health_report
from app.services.hf_client import HfClient
from app.services.job_manager import JobManager
from app.services.model_installer import InstallStatus, ModelInstaller
from app.services.model_preflight import preflight_upscaler
from app.services.model_registry import ModelKind, ModelRegistry, ModelStatus
from app.services.resource_probes import DxgiVramProbe, SystemRamProbe
from app.services.tile_params import validate_tile_params

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL_NOT_INSTALLED = 3
EXIT_DEVICE = 4
EXIT_FAILED = 5

ENGINE_FORMATS = ("png", "jpg", "jpeg", "webp")
# Formatos que el motor no escribe: salida PNG del motor + ffmpeg vendorizado.
FFMPEG_ENCODERS: dict[str, list[str]] = {
    "jxl": ["-c:v", "libjxl", "-distance", "1.0", "-effort", "7"],
    "avif": ["-c:v", "libaom-av1", "-crf", "18", "-cpu-used", "4", "-still-picture", "1"],
}
INSTALL_TERMINAL = (InstallStatus.installed, InstallStatus.error)
INSTALL_POLL_SECONDS = 1.0


class HeadlessError(RuntimeError):
    exit_code = EXIT_FAILED


class UsageError(HeadlessError):
    exit_code = EXIT_USAGE


class ModelNotInstalledError(HeadlessError):
    exit_code = EXIT_MODEL_NOT_INSTALLED


class DeviceError(HeadlessError):
    exit_code = EXIT_DEVICE


class InferenceError(HeadlessError):
    exit_code = EXIT_FAILED


@dataclass(slots=True)
class HeadlessContext:
    settings: Settings
    registry: ModelRegistry
    devices: DevicesService
    probes: dict[str, Any]
    ncnn_engine: RealEsrganNcnnEngine
    onnx_engine: OnnxUpscaler
    job_manager: JobManager
    hf_client: HfClient


def build_context(settings: Settings | None = None) -> HeadlessContext:
    settings = settings or get_settings()
    registry = ModelRegistry(settings)
    devices = DevicesService(settings)
    probes: dict[str, Any] = {"gpu": DxgiVramProbe(), "cpu": SystemRamProbe()}
    ncnn_engine = RealEsrganNcnnEngine(settings, vram_probe=probes["gpu"])
    onnx_engine = OnnxUpscaler(settings, registry, devices, GpuSessionCoordinator())
    job_manager = JobManager(
        settings,
        ncnn_engine,
        DeviceSemaphores(settings, resource_probes=probes),
        onnx_engine=onnx_engine,
        registry=registry,
        devices=devices,
    )
    return HeadlessContext(
        settings=settings,
        registry=registry,
        devices=devices,
        probes=probes,
        ncnn_engine=ncnn_engine,
        onnx_engine=onnx_engine,
        job_manager=job_manager,
        hf_client=HfClient(settings),
    )


# ---------------------------------------------------------------- consultas


def health(ctx: HeadlessContext) -> dict[str, Any]:
    return build_health_report(
        ctx.settings, ctx.registry, ctx.devices, ctx.ncnn_engine, ctx.onnx_engine, ctx.probes.get("gpu")
    )


def list_models(ctx: HeadlessContext) -> dict[str, Any]:
    entries = [entry for entry in ctx.registry.list() if entry.kind in (ModelKind.builtin_ncnn, ModelKind.onnx)]
    return {
        "models": [
            {
                "id": entry.id,
                "name": entry.name,
                "kind": entry.kind.value,
                "scale": entry.scale,
                "scales": _scales_for(ctx.settings, entry.id, entry.scale),
                "status": entry.status.value,
                "source": entry.source,
            }
            for entry in entries
        ]
    }


def _scales_for(settings: Settings, model_id: str, native: int | None) -> list[int]:
    option = settings.get_model_option(model_id)
    if option:
        return list(option["scales"])
    if native is None:
        return []
    return [scale for scale in settings.allowed_scale_values if scale <= native]


# ---------------------------------------------------------------- upscale


def output_format_for(output: Path, explicit: str | None) -> str:
    fmt = (explicit or output.suffix.lstrip(".")).lower()
    if fmt in ENGINE_FORMATS or fmt in FFMPEG_ENCODERS:
        return fmt
    raise UsageError(f"unsupported output format {fmt!r}; use one of {ENGINE_FORMATS + tuple(FFMPEG_ENCODERS)}")


def engine_format_for(fmt: str) -> str:
    return "png" if fmt in FFMPEG_ENCODERS else fmt


def deterministic_job_id(source: Path, **params: Any) -> str:
    digest = hashlib.sha1(source.read_bytes())
    digest.update(repr(sorted(params.items())).encode("utf-8"))
    return f"cli-{digest.hexdigest()[:16]}"


async def resolve_device(ctx: HeadlessContext, device: str | None) -> str:
    if device == AUTO_DEVICE_ID:
        raise DeviceError("device 'auto' needs the server's router; pass an explicit device (cpu, dml:0...)")
    if device is None:
        return (await asyncio.to_thread(ctx.devices.resolve_default))["id"]
    try:
        await asyncio.to_thread(ctx.devices.validate, device)
    except ValueError as exc:
        raise DeviceError(str(exc)) from exc
    return device


def ensure_model_installed(ctx: HeadlessContext, model: str) -> None:
    if model in ctx.settings.model_keys:
        if not ctx.ncnn_engine.available():
            raise ModelNotInstalledError(
                f"builtin model {model!r} needs the realesrgan-ncnn pack at {ctx.settings.engine_binary_path}"
            )
        return
    entry = ctx.registry.get(model)
    if entry is None or entry.kind != ModelKind.onnx:
        raise ModelNotInstalledError(f"model {model!r} is not installed (see `upflow models`)")
    if entry.status != ModelStatus.installed:
        raise ModelNotInstalledError(f"model {model!r} is not ready (status={entry.status.value})")


async def upscale_image(
    ctx: HeadlessContext,
    source: Path,
    output: Path,
    *,
    model: str = "realesrgan-x4plus",
    scale: int = 4,
    tile_size: int | None = None,
    tile_overlap: int | None = None,
    device: str | None = None,
    output_format: str | None = None,
) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_file():
        raise UsageError(f"input file not found: {source}")
    fmt = output_format_for(output, output_format)
    started = time.perf_counter()
    device_id = await resolve_device(ctx, device)
    ensure_model_installed(ctx, model)
    job = _build_job(ctx, source, model, scale, tile_size, tile_overlap, device_id, engine_format_for(fmt))
    try:
        produced = await ctx.job_manager.run_inline(job)
    except HeadlessError:
        raise
    except Exception as exc:  # noqa: BLE001 - cualquier fallo del motor es codigo 5
        raise InferenceError(str(exc)) from exc
    # Dimensiones ANTES de entregar: PIL no abre jxl/avif, la salida del motor si.
    size = image_size(produced)
    deliver_output(ctx.settings, produced, output, fmt)
    return describe_result(job, output, fmt, device_id, size, time.perf_counter() - started)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _build_job(
    ctx: HeadlessContext,
    source: Path,
    model: str,
    scale: int,
    tile_size: int | None,
    tile_overlap: int | None,
    device_id: str,
    engine_format: str,
) -> UpscaleJob:
    try:
        validate_tile_params(tile_size, tile_overlap)
        resolution = ctx.job_manager.resolve_model(
            model_id=model, scale=scale, output_format=engine_format, device=device_id
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    job = UpscaleJob(
        source_path=source,
        original_filename=source.name,
        model_name=resolution.engine_model_name,
        scale=resolution.scale,
        output_format=engine_format,
        model_id=resolution.model_id,
        device=device_id,
        native_scale=resolution.native_scale,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    job.id = deterministic_job_id(
        source, model=model, scale=scale, tile=tile_size, overlap=tile_overlap, device=device_id, fmt=engine_format
    )
    return job


def deliver_output(settings: Settings, produced: Path, output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt in FFMPEG_ENCODERS:
        try:
            convert_with_ffmpeg(settings, produced, output, fmt)
        finally:
            # Sin sweeper en modo headless: el PNG del motor no puede quedar huerfano.
            produced.unlink(missing_ok=True)
        return
    shutil.move(str(produced), str(output))


def convert_with_ffmpeg(settings: Settings, source_png: Path, target: Path, fmt: str) -> None:
    ffmpeg = settings.ffmpeg_binary_path
    if not ffmpeg.exists():
        raise UsageError(f"format {fmt!r} needs the vendored ffmpeg (missing at {ffmpeg})")
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_png)]
    command += FFMPEG_ENCODERS[fmt] + [str(target)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not target.exists():
        raise InferenceError(f"ffmpeg could not encode {fmt}: {completed.stderr.strip()[:400]}")


def describe_result(
    job: UpscaleJob, output: Path, fmt: str, device_id: str, size: tuple[int, int], seconds: float
) -> dict[str, Any]:
    effective = job.metadata.get("effective", {})
    width, height = size
    return {
        "ok": True,
        "output": str(output),
        "width": width,
        "height": height,
        "model": job.model_id,
        "engineModel": job.model_name,
        "engine": effective.get("engine"),
        "device": device_id,
        "scale": job.scale,
        "nativeScale": job.native_scale,
        "resized": bool(effective.get("resized", False)),
        "tile": {
            "size": effective.get("tileSize"),
            "overlap": effective.get("tileOverlap"),
            "meaning": effective.get("tileSizeMeaning"),
        },
        "format": fmt,
        "seconds": round(seconds, 2),
    }


# ---------------------------------------------------------------- modelos HF


async def preflight_model(ctx: HeadlessContext, repo_id: str) -> dict[str, Any]:
    report = await preflight_upscaler(
        hf_client=ctx.hf_client,
        devices_service=ctx.devices,
        settings=ctx.settings,
        probes=ctx.probes,
        repo_id=repo_id,
        strategy=strategy_for("image"),
    )
    payload = asdict(report)
    payload["compat"] = str(payload["compat"]) if payload["compat"] is not None else None
    return payload


async def install_model(
    ctx: HeadlessContext, repo_id: str, *, poll_seconds: float = INSTALL_POLL_SECONDS
) -> dict[str, Any]:
    installer = ModelInstaller(ctx.settings, ctx.registry, ctx.hf_client)
    await installer.start()
    try:
        try:
            install_id = await installer.install_from_hf(repo_id)
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        job = await _wait_install(installer, install_id, poll_seconds)
    finally:
        await installer.stop()
    return {
        "ok": job.status == InstallStatus.installed,
        "installId": job.id,
        "repoId": job.repo_id,
        "status": job.status.value,
        "modelId": job.model_id,
        "progressPct": job.progress_pct,
        "error": job.error,
    }


async def _wait_install(installer: ModelInstaller, install_id: str, poll_seconds: float) -> Any:
    while True:
        job = installer.status(install_id)
        if job is None:
            raise InferenceError(f"install job {install_id} vanished")
        if job.status in INSTALL_TERMINAL:
            return job
        await asyncio.sleep(poll_seconds)
