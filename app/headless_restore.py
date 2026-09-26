"""Restauracion de fotos en proceso: upflow restore y las tools MCP de restauracion sin servidor."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.api.restore_routes import analysis_response
from app.config import Settings
from app.headless_base import (
    DEFAULT_SR_MODEL,
    HeadlessContext,
    ModelNotInstalledError,
    UsageError,
    deterministic_job_id,
    ensure_model_installed,
    existing_file,
    image_size,
    resolve_device,
    run_job_inline,
)
from app.models import UpscaleJob
from app.schemas_restore import RestoreOptions
from app.services.photo_geometry import Geometry
from app.services.photo_restore_chain import steps_from_selection
from app.services.photo_restore_job import UPSCALE_AI, restore_upscale_mode, step_uses_model
from app.services.photo_restore_presets import photo_preset, resolve_preset
from app.services.restore_outputs import discard_restore_outputs, restore_output_paths
from app.services.restore_provenance import EXTENSIONS, write_sidecar
from app.services.restore_session import PREVIEW_NAME, SessionAnalysis, SessionNotFound, session_dir

@dataclass(frozen=True, slots=True)
class RestoreSpec:
    steps: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    preset: str | None = None
    scale: int = 1
    model: str = DEFAULT_SR_MODEL
    device: str | None = None
    output_format: str | None = None


@dataclass(frozen=True, slots=True)
class RestorePlan:
    steps: tuple[str, ...]
    options: dict[str, Any]
    preset: str | None


@dataclass(frozen=True, slots=True)
class RestoreSource:
    path: Path
    name: str
    token: str | None = None
    analysis: SessionAnalysis | None = None
    owned: bool = False


@dataclass(frozen=True, slots=True)
class RestoreDelivery:
    output: Path
    uncolored: Path | None
    details: Path


def restore_format_for(output: Path, explicit: str | None) -> str:
    fmt = (explicit or output.suffix.lstrip(".")).lower()
    if fmt in EXTENSIONS:
        return fmt
    raise UsageError(f"unsupported restore format {fmt!r}; use one of {tuple(EXTENSIONS)}")


def validated_restore_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        options = RestoreOptions.model_validate(dict(raw)).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise UsageError(f"invalid restore options: {validation_message(exc)}") from exc
    if "preview_crop" in options:
        raise UsageError("preview_crop (Preview this area) needs the Upflow server; restore the whole photo here")
    return options


def validation_message(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())


def preset_step_options(preset_id: str, steps: Sequence[str]) -> dict[str, dict[str, Any]]:
    try:
        preset = photo_preset(preset_id)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    return {step.step_id: dict(step.options) for step in preset.steps if step.step_id in steps}


def merge_restore_options(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        both_mappings = isinstance(current, Mapping) and isinstance(value, Mapping)
        merged[key] = {**current, **value} if both_mappings else value
    return merged


def with_preset(options: dict[str, Any], preset: str | None) -> dict[str, Any]:
    return options if preset is None else {**options, "preset": preset}


def plan_from_steps(spec: RestoreSpec) -> RestorePlan:
    base = {} if spec.preset is None else preset_step_options(spec.preset, spec.steps)
    options = merge_restore_options(base, spec.options)
    return RestorePlan(tuple(spec.steps), with_preset(options, spec.preset), spec.preset)


def plan_from_analysis(spec: RestoreSpec, analysis: Any) -> RestorePlan:
    preset = spec.preset or analysis.diagnosis.proposed_preset
    try:
        selection = resolve_preset(preset, analysis.diagnosis.facts)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    if not selection.steps:
        raise UsageError(f"the analysis found nothing for preset {preset!r} to fix; choose steps with --steps")
    options = merge_restore_options(selection.options, spec.options)
    return RestorePlan(selection.steps, with_preset(options, preset), preset)


def plan_for(spec: RestoreSpec, origin: RestoreSource) -> RestorePlan:
    return plan_from_steps(spec) if spec.steps else plan_from_analysis(spec, origin.analysis)


async def analyze_photo(ctx: HeadlessContext, source: Path) -> dict[str, Any]:
    analysis = await open_restore_session(ctx, existing_file(source))
    return analysis_payload(ctx, analysis)


def analysis_payload(ctx: HeadlessContext, analysis: SessionAnalysis) -> dict[str, Any]:
    payload = analysis_response(analysis).model_dump(by_alias=True, mode="json")
    preview = ctx.restore_sessions.file(analysis.record.token, PREVIEW_NAME)
    return {"ok": True, **payload, "previewPath": str(preview)}


async def open_restore_session(ctx: HeadlessContext, source: Path) -> SessionAnalysis:
    # La sesion se queda con una copia: la foto del usuario nunca se mueve.
    upload = ctx.settings.uploads_path / f"{uuid4().hex}-{source.name}"
    try:
        await asyncio.to_thread(copy_file, source, upload)
        return await ctx.restore_sessions.open(upload, source.name)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    finally:
        upload.unlink(missing_ok=True)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


async def restore_source(
    ctx: HeadlessContext, spec: RestoreSpec, source: Path | None, token: str | None
) -> RestoreSource:
    if (source is None) == (token is None):
        raise UsageError("pass either an input photo or an analysis token")
    if token is not None:
        return session_source(ctx, spec, token)
    path = existing_file(source)
    if spec.steps:
        return RestoreSource(path, path.name)
    return await analyzed_source(ctx, path, spec.options.get("geometry"))


def session_source(ctx: HeadlessContext, spec: RestoreSpec, token: str) -> RestoreSource:
    if not spec.steps:
        raise UsageError("a restore from an analysis token needs its steps (see proposedSteps)")
    try:
        record = ctx.restore_sessions.record(token)
        return RestoreSource(ctx.restore_sessions.original_path(token), record.original_name, token)
    except SessionNotFound as exc:
        raise UsageError("unknown restore session; analyze the photo again") from exc


async def analyzed_source(ctx: HeadlessContext, source: Path, geometry: Mapping[str, Any] | None) -> RestoreSource:
    analysis = await open_restore_session(ctx, source)
    token = analysis.record.token
    origin = RestoreSource(
        ctx.restore_sessions.original_path(token), analysis.record.original_name, token, analysis, owned=True
    )
    if not geometry:
        return origin
    try:
        return replace(origin, analysis=await session_geometry(ctx, token, geometry))
    except BaseException:
        discard_owned_session(ctx.settings, origin)
        raise


async def session_geometry(ctx: HeadlessContext, token: str, geometry: Mapping[str, Any]) -> SessionAnalysis:
    # La mascara y las caras se miden sobre la copia de trabajo: la geometria va a la sesion.
    try:
        return await ctx.restore_sessions.set_geometry(token, Geometry.from_mapping(geometry))
    except ValueError as exc:
        raise UsageError(str(exc)) from exc


def discard_owned_session(settings: Settings, origin: RestoreSource) -> None:
    if origin.owned and origin.token is not None:
        shutil.rmtree(session_dir(settings.video_work_path, origin.token), ignore_errors=True)


def ensure_steps_ready(ctx: HeadlessContext, plan: RestorePlan) -> None:
    try:
        specs = steps_from_selection(list(plan.steps))
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    for step in specs:
        check_step_ready(ctx, step.id, step_uses_model(step, plan.options))


def check_step_ready(ctx: HeadlessContext, step_id: str, uses_model: bool) -> None:
    try:
        ctx.restore_runner.check_ready(ctx.settings, step_id, uses_model=uses_model)
    except ValueError as exc:
        raise ModelNotInstalledError(str(exc)) from exc


async def prepare_restore_job(
    ctx: HeadlessContext, plan: RestorePlan, spec: RestoreSpec, origin: RestoreSource
) -> UpscaleJob:
    ensure_steps_ready(ctx, plan)
    try:
        job = await ctx.job_manager.build_job(
            source_path=origin.path,
            original_filename=origin.name,
            model_name=spec.model,
            scale=spec.scale,
            output_format=str(spec.output_format),
            device=spec.device,
            restore_steps=plan.steps,
            restore_options=plan.options,
            restore_session=origin.token,
        )
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    if restore_upscale_mode(job.restore_options, job.scale) == UPSCALE_AI:
        ensure_model_installed(ctx, spec.model)
    job.id = deterministic_job_id(
        origin.path,
        steps=job.restore_steps,
        options=json.dumps(job.restore_options, sort_keys=True),
        scale=spec.scale,
        model=spec.model,
        device=spec.device,
        fmt=spec.output_format,
    )
    return job


async def restore_image(
    ctx: HeadlessContext,
    output: Path,
    spec: RestoreSpec,
    *,
    source: Path | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    fmt = restore_format_for(output, spec.output_format)
    options = validated_restore_options(spec.options)
    started = time.perf_counter()
    device_id = await resolve_device(ctx, spec.device)
    spec = replace(spec, options=options, device=device_id, output_format=fmt)
    origin = await restore_source(ctx, spec, source, token)
    try:
        plan = plan_for(spec, origin)
        job = await prepare_restore_job(ctx, plan, spec, origin)
        await run_job_inline(ctx, job)
        delivery = deliver_restore_outputs(ctx.settings, job, output)
    finally:
        discard_owned_session(ctx.settings, origin)
    return describe_restore(job, plan, delivery, origin, time.perf_counter() - started)


def delivery_targets(output: Path, has_uncolored: bool) -> RestoreDelivery:
    uncolored = output.with_name(f"{output.stem}.uncolored{output.suffix}") if has_uncolored else None
    return RestoreDelivery(output, uncolored, output.with_name(f"{output.stem}.restore.json"))


def delivered_names(delivery: RestoreDelivery) -> dict[str, str]:
    names = {"restored": delivery.output.name}
    if delivery.uncolored is not None:
        names["uncolored"] = delivery.uncolored.name
    return names


def relocated_sidecar(sidecar: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, Any]:
    outputs = [
        {**entry, "file": names[entry["role"]]} for entry in sidecar.get("outputs", []) if entry.get("role") in names
    ]
    return {**sidecar, "outputs": outputs}


def deliver_restore_outputs(settings: Settings, job: UpscaleJob, output: Path) -> RestoreDelivery:
    paths = restore_output_paths(settings.outputs_path, job.id, job.output_format)
    targets = delivery_targets(output, paths.uncolored.is_file())
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        sidecar = json.loads(paths.sidecar.read_text(encoding="utf-8"))
        shutil.move(str(paths.final), str(targets.output))
        if targets.uncolored is not None:
            shutil.move(str(paths.uncolored), str(targets.uncolored))
        write_sidecar(targets.details, relocated_sidecar(sidecar, delivered_names(targets)))
    finally:
        # Sin sweeper en modo headless: vista, antes/despues y artefactos de caras no quedan huerfanos.
        discard_restore_outputs(paths)
    return targets


def describe_restore(
    job: UpscaleJob, plan: RestorePlan, delivery: RestoreDelivery, origin: RestoreSource, seconds: float
) -> dict[str, Any]:
    summary = job.metadata.get("restore") or {}
    width, height = image_size(delivery.output)
    return {
        "ok": True,
        "output": str(delivery.output),
        "uncolored": None if delivery.uncolored is None else str(delivery.uncolored),
        "details": str(delivery.details),
        "width": width,
        "height": height,
        "steps": list(job.restore_steps),
        "preset": plan.preset,
        "options": dict(job.restore_options),
        "scale": job.scale,
        "upscale": summary.get("upscale"),
        "device": job.device,
        "format": job.output_format,
        "faces": summary.get("faces", []),
        "compositeReasons": summary.get("compositeReasons", []),
        "badge": summary.get("badge", False),
        "digitalSourceType": summary.get("digitalSourceType"),
        "warnings": summary.get("warnings", []),
        "cpuFallback": summary.get("cpuFallback", []),
        "recomposeAvailable": False,
        "token": None if origin.owned else origin.token,
        "seconds": round(seconds, 2),
    }
