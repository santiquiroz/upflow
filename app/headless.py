"""Modo headless de Upflow: reescalado, restauracion de fotos y carril clasico de CCTV en proceso, sin servidor.

Es la fachada: la restauracion vive en headless_restore.py, el modo CCTV en headless_cctv.py y los
errores, el contexto y las ayudas de job compartidas en headless_base.py; todo se usa como app.headless.<nombre>.

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

`restore_image` entrega la foto restaurada, la copia sin color (si hubo color) y
el JSON de detalles junto a la salida; vista, antes/despues y artefactos de caras
se borran (sin sweeper), asi que "recomponer caras" necesita el servidor.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.config import Settings
from app.headless_base import (  # noqa: F401 - fachada: la CLI, MCP y los tests usan app.headless.<nombre>
    DEFAULT_SR_MODEL,
    DeviceError,
    EXIT_DEVICE,
    EXIT_FAILED,
    EXIT_MODEL_NOT_INSTALLED,
    EXIT_OK,
    EXIT_USAGE,
    HeadlessContext,
    HeadlessError,
    InferenceError,
    ModelNotInstalledError,
    UsageError,
    build_context,
    deterministic_job_id,
    ensure_model_installed,
    image_size,
    resolve_device,
    run_job_inline,
)
from app.headless_cctv import (  # noqa: F401 - fachada: la CLI, MCP y los tests usan app.headless.<nombre>
    CCTV_CLARIFY_TASK,
    CCTV_ROI_TASK,
    CctvClarifyChoices,
    CctvRoiChoices,
    cctv_check_unchanged,
    cctv_clarify,
    cctv_clarify_file,
    cctv_error,
    cctv_options_of,
    cctv_probe,
    cctv_result_dir,
    cctv_roi,
    cctv_roi_file,
    cctv_roi_options_of,
    check_osd_decision,
    check_roi_request,
    checked_job_id,
    choices_with_preset_steps,
    classic_preset_steps,
    normalized_step,
    parse_sample_aspect,
    require_cctv_mode,
    require_steps_for_preset,
    roi_choices_with_steps,
    with_preset_steps,
)
from app.headless_restore import (  # noqa: F401 - fachada: la CLI, MCP y los tests usan app.headless.<nombre>
    RestoreSpec,
    analyze_photo,
    merge_restore_options,
    plan_from_analysis,
    plan_from_steps,
    preset_step_options,
    relocated_sidecar,
    restore_format_for,
    restore_image,
    validated_restore_options,
)
from app.models import UpscaleJob
from app.services.compat_strategy import strategy_for
from app.services.health_report import build_health_report
from app.services.model_installer import InstallStatus, ModelInstaller
from app.services.model_preflight import preflight_upscaler
from app.services.model_registry import ModelKind
from app.services.tile_params import validate_tile_params

ENGINE_FORMATS = ("png", "jpg", "jpeg", "webp")
# Formatos que el motor no escribe: salida PNG del motor + ffmpeg vendorizado.
FFMPEG_ENCODERS: dict[str, list[str]] = {
    "jxl": ["-c:v", "libjxl", "-distance", "1.0", "-effort", "7"],
    "avif": ["-c:v", "libaom-av1", "-crf", "18", "-cpu-used", "4", "-still-picture", "1"],
}
INSTALL_TERMINAL = (InstallStatus.installed, InstallStatus.error)
INSTALL_POLL_SECONDS = 1.0


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
    produced = await run_job_inline(ctx, job)
    # Dimensiones ANTES de entregar: PIL no abre jxl/avif, la salida del motor si.
    size = image_size(produced)
    deliver_output(ctx.settings, produced, output, fmt)
    return describe_result(job, output, fmt, device_id, size, time.perf_counter() - started)


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
