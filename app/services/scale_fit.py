from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings

# El binario realesrgan-ncnn-vulkan NO reescala a una escala distinta de la del
# modelo: con `-s 2` sobre un modelo x4 arma cada tile copiando el cuarto superior
# izquierdo del tile ampliado x4 (medido 2026-09-02: dimensiones correctas, PSNR
# 13 dB contra el x4 real, rejilla de tiles a la vista). Los motores corren SIEMPRE
# a la escala nativa del modelo; la escala pedida se resuelve aca con Lanczos.
_TRAILING_SCALE = re.compile(r"-x([234])$")
DEFAULT_NATIVE_SCALE = 4
INTERMEDIATE_SUFFIX = ".native.png"
SAVE_OPTIONS: dict[str, dict[str, Any]] = {
    "jpg": {"quality": 95},
    "jpeg": {"quality": 95},
    "webp": {"quality": 95},
}


def native_scale_for_engine_model(engine_model_name: str) -> int:
    match = _TRAILING_SCALE.search(engine_model_name)
    return int(match.group(1)) if match else DEFAULT_NATIVE_SCALE


def effective_scale(requested: int, native: int | None) -> int:
    if native is None:
        return requested
    return min(requested, native)


def requested_scale_of(job: Any) -> int | None:
    return getattr(job, "scale", None)


def native_scale_of(job: Any) -> int | None:
    return getattr(job, "native_scale", None) or requested_scale_of(job)


def needs_resize(job: Any) -> bool:
    requested, native = requested_scale_of(job), native_scale_of(job)
    return requested is not None and native is not None and requested != native


def final_output_path(settings: Settings, job: Any) -> Path:
    return settings.outputs_path / f"{job.id}.{job.output_format.lower()}"


def engine_output_path(settings: Settings, job: Any) -> Path:
    # Intermedio PNG (sin perdida) cuando falta reducir: el formato final se
    # codifica una sola vez, en fit_output_to_scale.
    if needs_resize(job):
        return settings.outputs_path / f"{job.id}{INTERMEDIATE_SUFFIX}"
    return final_output_path(settings, job)


def scaled_size(native_size: tuple[int, int], requested: int, native: int) -> tuple[int, int]:
    width, height = native_size
    return (round(width * requested / native), round(height * requested / native))


def fit_output_to_scale(native_path: Path, job: Any, settings: Settings) -> Path:
    if not needs_resize(job):
        return native_path
    target = final_output_path(settings, job)
    with Image.open(native_path) as image:
        size = scaled_size(image.size, job.scale, native_scale_of(job) or job.scale)
        resized = image.resize(size, Image.LANCZOS)
    save_image(resized, target)
    native_path.unlink(missing_ok=True)
    job.metadata["effective"] = {**job.metadata.get("effective", {}), "resized": True, "resizeFilter": "lanczos"}
    return target


def save_image(image: Image.Image, target: Path) -> None:
    fmt = target.suffix.lstrip(".").lower()
    if fmt in ("jpg", "jpeg") and image.mode != "RGB":
        image = image.convert("RGB")
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, **SAVE_OPTIONS.get(fmt, {}))
