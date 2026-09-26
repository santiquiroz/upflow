"""Base del modo headless: codigos de salida, errores, contexto de servicios y ayudas de job compartidas."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings, get_settings
from app.models import UpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.onnx_upscaler import OnnxUpscaler
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.hf_client import HfClient
from app.services.job_manager import JobManager
from app.services.model_registry import ModelKind, ModelRegistry, ModelStatus
from app.services.photo_restore_job import PhotoRestoreJobRunner
from app.services.resource_probes import DxgiVramProbe, SystemRamProbe
from app.services.restore_session import RestoreSessionStore, default_detectors

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MODEL_NOT_INSTALLED = 3
EXIT_DEVICE = 4
EXIT_FAILED = 5
DEFAULT_SR_MODEL = "realesrgan-x4plus"


class HeadlessError(RuntimeError):
    exit_code = EXIT_FAILED

    def __init__(self, message: str, key: str | None = None) -> None:
        super().__init__(message)
        self.key = key


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
    restore_sessions: RestoreSessionStore
    restore_runner: PhotoRestoreJobRunner


def build_context(settings: Settings | None = None) -> HeadlessContext:
    settings = settings or get_settings()
    registry = ModelRegistry(settings)
    devices = DevicesService(settings)
    probes: dict[str, Any] = {"gpu": DxgiVramProbe(), "cpu": SystemRamProbe()}
    ncnn_engine = RealEsrganNcnnEngine(settings, vram_probe=probes["gpu"])
    coordinator = GpuSessionCoordinator()
    onnx_engine = OnnxUpscaler(settings, registry, devices, coordinator)
    restore_engine = PhotoRestoreEngine(settings, coordinator, device_health=devices)
    restore_sessions = RestoreSessionStore(settings, default_detectors(settings, restore_engine))
    restore_runner = PhotoRestoreJobRunner(settings, restore_engine, sessions=restore_sessions)
    job_manager = JobManager(
        settings,
        ncnn_engine,
        DeviceSemaphores(settings, resource_probes=probes),
        onnx_engine=onnx_engine,
        registry=registry,
        devices=devices,
        restore_runner=restore_runner,
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
        restore_sessions=restore_sessions,
        restore_runner=restore_runner,
    )


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


async def run_job_inline(ctx: HeadlessContext, job: UpscaleJob) -> Path:
    try:
        return await ctx.job_manager.run_inline(job)
    except HeadlessError:
        raise
    except Exception as exc:  # noqa: BLE001 - cualquier fallo del motor es codigo 5
        raise InferenceError(str(exc)) from exc


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def existing_file(source: Path) -> Path:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise UsageError(f"input file not found: {path}")
    return path
