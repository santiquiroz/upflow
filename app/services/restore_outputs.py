from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.services.image_io import LoadedImage, MetadataPrivacy, save_restored
from app.services.photo_restore_pipeline import PostResult, PreResult, restore_metadata
from app.services.restore_provenance import (
    EXTENSIONS,
    InputInfo,
    ModelCatalog,
    OutputFile,
    SidecarContext,
    UpscaleInfo,
    badge_applies,
    before_after_image,
    build_sidecar,
    download_names,
    encode_jpeg,
    facts_from_records,
    is_composite,
    output_bit_depth,
    sha256_file,
    with_badge,
    write_sidecar,
    xmp_fields,
)
from app.services.xmp_packet import build_xmp_packet

VIEW_MAX_SIDE = 8192
PREVIEW_MAX_SIDE = 2048
ARTIFACT_DIR_SUFFIX = ".restore"


@dataclass(frozen=True, slots=True)
class RestoreOutputPaths:
    final: Path
    uncolored: Path
    preview: Path
    view: Path
    before_after: Path
    sidecar: Path
    artifact_dir: Path

    def files(self) -> tuple[Path, ...]:
        return (self.final, self.uncolored, self.preview, self.view, self.before_after, self.sidecar)


@dataclass(frozen=True, slots=True)
class OutputContext:
    original_name: str
    source_path: Path
    fmt: str
    upscale: UpscaleInfo
    app_version: str
    catalog: ModelCatalog
    options: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)


def restore_output_paths(outputs_dir: Path, job_id: str, fmt: str) -> RestoreOutputPaths:
    ext = EXTENSIONS.get(fmt.lower(), fmt.lower())
    return RestoreOutputPaths(
        final=outputs_dir / f"{job_id}.{ext}",
        uncolored=outputs_dir / f"{job_id}.uncolored.{ext}",
        preview=outputs_dir / f"{job_id}.preview.jpg",
        view=outputs_dir / f"{job_id}.view.jpg",
        before_after=outputs_dir / f"{job_id}.beforeafter.jpg",
        sidecar=outputs_dir / f"{job_id}.restore.json",
        artifact_dir=outputs_dir / f"{job_id}{ARTIFACT_DIR_SUFFIX}",
    )


def discard_restore_outputs(paths: RestoreOutputPaths) -> None:
    for path in paths.files():
        path.unlink(missing_ok=True)
    shutil.rmtree(paths.artifact_dir, ignore_errors=True)


def saved_bit_depth(fmt: str, input_bit_depth: int, upscale: UpscaleInfo) -> int:
    depth = output_bit_depth(input_bit_depth, upscale)
    return depth if EXTENSIONS.get(fmt.lower()) == "png" else min(depth, 8)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def fit_long_side(pixels: np.ndarray, max_side: int) -> np.ndarray:
    height, width = pixels.shape[:2]
    if max(height, width) <= max_side:
        return pixels
    factor = max_side / max(height, width)
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return np.asarray(Image.fromarray(pixels, mode="RGB").resize(size, Image.Resampling.LANCZOS))


def write_jpeg(pixels: np.ndarray, path: Path, icc: bytes | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_jpeg(pixels, icc))


def save_preview_output(post: PostResult, paths: RestoreOutputPaths, icc: bytes | None) -> Path:
    if post.preview is None:
        raise ValueError("The preview area produced no image")
    write_jpeg(to_uint8(post.preview), paths.preview, icc)
    return paths.preview


def save_full_outputs(
    loaded: LoadedImage, pre: PreResult, post: PostResult, paths: RestoreOutputPaths, context: OutputContext
) -> dict[str, Any]:
    facts = facts_from_records((*pre.records, *post.records), context.upscale)
    badge = badge_applies(facts, _wants_badge(context))
    input_info = _input_info(loaded, context)
    draft = _sidecar(input_info, pre, post, context, (), None)
    xmp = build_xmp_packet(xmp_fields(draft, _photo_date(context)))
    final = with_badge(post.image) if badge else post.image
    privacy = _save_image(final, paths.final, loaded, context, xmp)
    outputs = [OutputFile("restored", paths.final)]
    if post.uncolored is not None:
        _save_image(post.uncolored, paths.uncolored, loaded, context, xmp)
        outputs.append(OutputFile("uncolored", paths.uncolored))
    outputs.extend(_save_views(final, paths, loaded.icc))
    # El antes/despues se comparte fuera de Upflow: lleva la insignia en todo composite (§3.6).
    before_after = before_after_image(pre.request.image, post.image, badge=is_composite(facts))
    write_jpeg(before_after, paths.before_after, loaded.icc)
    outputs.append(OutputFile("beforeafter", paths.before_after))
    sidecar = _sidecar(input_info, pre, post, context, tuple(outputs), privacy)
    write_sidecar(paths.sidecar, sidecar)
    return restore_summary(pre, post, sidecar, context, badge=badge, view_full=_is_full_view(final))


def restore_summary(
    pre: PreResult,
    post: PostResult,
    sidecar: Mapping[str, Any],
    context: OutputContext,
    *,
    badge: bool,
    view_full: bool,
) -> dict[str, Any]:
    names = download_names(context.original_name, context.fmt, colorized=post.uncolored is not None)
    return {
        **restore_metadata(pre, post),
        "upscale": context.upscale.to_metadata(),
        "digitalSourceType": sidecar["digitalSourceType"],
        "compositeReasons": list(sidecar["compositeReasons"]),
        "badge": badge,
        "downloadNames": asdict(names),
        "artifacts": _artifact_names(post),
        "viewFullResolution": view_full,
    }


def _save_image(
    rgb: np.ndarray, path: Path, loaded: LoadedImage, context: OutputContext, xmp: str
) -> MetadataPrivacy:
    depth = saved_bit_depth(context.fmt, loaded.bit_depth, context.upscale)
    keep_gps = bool(context.options.get("keep_gps", False))
    return save_restored(rgb, path, context.fmt, depth, loaded.icc, loaded.exif, xmp, keep_gps=keep_gps)


def _save_views(final: np.ndarray, paths: RestoreOutputPaths, icc: bytes | None) -> list[OutputFile]:
    pixels = to_uint8(final)
    write_jpeg(fit_long_side(pixels, VIEW_MAX_SIDE), paths.view, icc)
    write_jpeg(fit_long_side(pixels, PREVIEW_MAX_SIDE), paths.preview, icc)
    return [OutputFile("view", paths.view), OutputFile("preview", paths.preview)]


def _sidecar(
    input_info: InputInfo,
    pre: PreResult,
    post: PostResult,
    context: OutputContext,
    outputs: tuple[OutputFile, ...],
    privacy: MetadataPrivacy | None,
) -> dict[str, object]:
    sidecar_context = SidecarContext(
        input=input_info,
        outputs=outputs,
        output_bit_depth=saved_bit_depth(context.fmt, input_info.bit_depth, context.upscale),
        app_version=context.app_version,
        upscale=context.upscale,
        geometry=dict(context.options.get("geometry") or {}),
        privacy=privacy,
        photo_date=_photo_date(context),
        environment=context.environment,
    )
    return build_sidecar(pre, post, sidecar_context, context.catalog)


def _input_info(loaded: LoadedImage, context: OutputContext) -> InputInfo:
    return InputInfo(
        name=context.original_name,
        sha256=sha256_file(context.source_path),
        size_bytes=context.source_path.stat().st_size,
        bit_depth=loaded.bit_depth,
        has_icc=loaded.icc is not None,
        orientation_applied=loaded.orientation_applied,
    )


def _artifact_names(post: PostResult) -> list[str]:
    names = ["preview", "view", "beforeafter", "sidecar"]
    return [*names, "uncolored"] if post.uncolored is not None else names


def _is_full_view(image: np.ndarray) -> bool:
    return max(image.shape[:2]) <= VIEW_MAX_SIDE


def _wants_badge(context: OutputContext) -> bool:
    return bool(context.options.get("badge", True))


def _photo_date(context: OutputContext) -> str | None:
    value = context.options.get("photo_date")
    return None if value is None else str(value)
