from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.image_io import MetadataPrivacy
from app.services.photo_restore_chain import STEP_PLAIN_NAMES
from app.services.photo_restore_pipeline import PostResult, PreResult, StepRecord, restore_metadata
from app.services.restore_models import RESTORE_BUNDLES, RESTORE_MODELS, VENDORED_MODELS, RestoreModelSpec
from app.services.xmp_packet import DIGITAL_SOURCE_COMPOSITE, DIGITAL_SOURCE_ENHANCED, XmpFields, build_xmp_packet

SIDECAR_SCHEMA_VERSION = 1
# Un relleno de mas del 1% de la foto ya es contenido inventado a la vista (§3.6).
FILL_COMPOSITE_COVERAGE = 0.01
FP16_EFFECTIVE_BITS = 11
AI_UPSCALE_BITS = 8
BADGE_TEXT = "Restored with AI"
BADGE_FONT_FRACTION = 0.025
BADGE_MIN_FONT_PX = 9
BADGE_MAX_FONT_PX = 64
BADGE_MAX_WIDTH_FRACTION = 0.5
BADGE_MIN_IMAGE_PX = 48
BADGE_BACKGROUND = (0, 0, 0, 150)
BADGE_TEXT_COLOR = (255, 255, 255, 235)
BEFORE_AFTER_MAX_HEIGHT = 2048
BEFORE_AFTER_DIVIDER_PX = 4
JPEG_QUALITY = 90
STEM_MAX_CHARS = 100
STEM_FALLBACK = "photo"
EXTENSIONS = {"png": "png", "jpg": "jpg", "jpeg": "jpg", "webp": "webp"}
HASH_CHUNK_BYTES = 1 << 20
COMPOSITE_REASONS = ("faces", "colorize", "largeFill", "fillOverFace", "generativeUpscale")
UPSCALE_PLAIN_NAMES = {"classic": "classic upscaling (Lanczos)", "ai": "AI upscaling"}
STEM_UNSAFE = re.compile(r"[^\w .()\-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ProvenanceFacts:
    faces_restored: int = 0
    colorized: bool = False
    fill_coverage: float = 0.0
    fill_touches_faces: bool = False
    generative_upscale: bool = False


@dataclass(frozen=True, slots=True)
class UpscaleInfo:
    mode: str = "none"
    scale: float = 1.0
    model: str | None = None
    generative: bool = False
    backend: str | None = None
    precision: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "scale": self.scale,
            "model": self.model,
            "generative": self.generative,
            "backend": self.backend,
            "precision": self.precision,
        }


@dataclass(frozen=True, slots=True)
class InputInfo:
    name: str
    sha256: str
    size_bytes: int
    bit_depth: int
    has_icc: bool
    orientation_applied: int

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "bitDepth": self.bit_depth,
            "hasIcc": self.has_icc,
            "orientationApplied": self.orientation_applied,
        }


@dataclass(frozen=True, slots=True)
class OutputFile:
    role: str
    path: Path


@dataclass(frozen=True, slots=True)
class DownloadNames:
    restored: str
    uncolored: str
    before_after: str
    sidecar: str


@dataclass(frozen=True, slots=True)
class SidecarContext:
    input: InputInfo
    outputs: tuple[OutputFile, ...]
    output_bit_depth: int
    app_version: str
    upscale: UpscaleInfo = UpscaleInfo()
    commit: str | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)
    privacy: MetadataPrivacy | None = None
    photo_date: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)


ModelFileResolver = Callable[[str, str], "Path | None"]
MODEL_HASH_CACHE_SIZE = 64


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    specs: Mapping[str, RestoreModelSpec]
    file_hashes: Mapping[tuple[str, str], str]
    resolve_file: ModelFileResolver | None = None

    def license_of(self, model_id: str) -> dict[str, object] | None:
        spec = self.specs.get(model_id)
        return None if spec is None else model_license(spec)

    def expected_sha256_of(self, model_id: str, precision: str) -> str | None:
        return self.file_hashes.get((model_id, precision))

    def sha256_of(self, model_id: str, precision: str) -> str | None:
        # Con resolvedor se hashea el archivo que se cargo de verdad; el catalogo es solo lo esperado.
        if self.resolve_file is None:
            return self.expected_sha256_of(model_id, precision)
        path = self.resolve_file(model_id, precision)
        return None if path is None or not path.is_file() else cached_file_sha256(path)


def default_model_catalog(resolve_file: ModelFileResolver | None = None) -> ModelCatalog:
    specs = {**{key: model.spec for key, model in VENDORED_MODELS.items()}, **RESTORE_MODELS}
    # Los vendorizados se usan sin cambios: el sha256 esperado del archivo es el de la fuente.
    vendored = {(key, "fp32"): model.spec.source_sha256 for key, model in VENDORED_MODELS.items()}
    published = {(a.model_id, a.precision): a.sha256 for bundle in RESTORE_BUNDLES.values() for a in bundle.artifacts}
    return ModelCatalog(specs, {**vendored, **published}, resolve_file)


def cached_file_sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_of_version(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=MODEL_HASH_CACHE_SIZE)
def _sha256_of_version(path: str, mtime_ns: int, size: int) -> str:
    # mtime y tamaño son parte de la clave: un archivo reemplazado se vuelve a hashear.
    return sha256_file(Path(path))


def model_license(spec: RestoreModelSpec) -> dict[str, object]:
    return {
        "spdx": spec.license_spdx,
        "url": spec.license_url,
        "copyright": spec.copyright,
        "attribution": spec.attribution,
        "commercialUse": spec.commercial_use,
        "dataLineage": spec.data_lineage,
    }


def composite_reasons(facts: ProvenanceFacts) -> tuple[str, ...]:
    flags = {
        "faces": facts.faces_restored > 0,
        "colorize": facts.colorized,
        "largeFill": facts.fill_coverage > FILL_COMPOSITE_COVERAGE,
        "fillOverFace": facts.fill_touches_faces,
        "generativeUpscale": facts.generative_upscale,
    }
    return tuple(reason for reason in COMPOSITE_REASONS if flags[reason])


def is_composite(facts: ProvenanceFacts) -> bool:
    return bool(composite_reasons(facts))


def digital_source_type(facts: ProvenanceFacts) -> str:
    return DIGITAL_SOURCE_COMPOSITE if is_composite(facts) else DIGITAL_SOURCE_ENHANCED


def badge_applies(facts: ProvenanceFacts, requested: bool = True) -> bool:
    # Los casos fieles nunca la llevan; en los composite va por defecto y el usuario la puede quitar.
    return requested and is_composite(facts)


def facts_from_records(records: Sequence[StepRecord], upscale: UpscaleInfo) -> ProvenanceFacts:
    by_id = {record.step_id: record for record in records}
    repair = by_id.get("repair")
    faces = by_id.get("faces")
    return ProvenanceFacts(
        faces_restored=0 if faces is None else len(faces.details.get("restored", ())),
        colorized="colorize" in by_id,
        fill_coverage=0.0 if repair is None else float(repair.details.get("finalCoverage", 0.0)),
        fill_touches_faces=False if repair is None else bool(repair.details.get("touchesFaces", False)),
        generative_upscale=upscale.mode == "ai" and upscale.generative,
    )


def facts_from_sidecar(sidecar: Mapping[str, Any]) -> ProvenanceFacts:
    damage = sidecar.get("damage") or {}
    upscale = sidecar.get("upscale") or {}
    faces = sidecar.get("faces") or []
    return ProvenanceFacts(
        faces_restored=sum(1 for face in faces if face.get("restored") and face.get("enabled")),
        colorized=sidecar.get("colorize") is not None,
        fill_coverage=float(damage.get("finalCoverage", 0.0)),
        fill_touches_faces=bool(damage.get("touchesFaces", False)),
        generative_upscale=upscale.get("mode") == "ai" and bool(upscale.get("generative")),
    )


def output_bit_depth(input_bit_depth: int, upscale: UpscaleInfo) -> int:
    # El motor SR trabaja en 8 bits: un 16 bits reescalado con IA ya no tiene 16 bits reales.
    return AI_UPSCALE_BITS if upscale.mode == "ai" else input_bit_depth


def effective_bits(bit_depth: int, records: Sequence[StepRecord]) -> int:
    fp16 = any(model.precision == "fp16" for record in records for model in _models_of(record))
    return min(bit_depth, FP16_EFFECTIVE_BITS) if fp16 else bit_depth


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outputs_metadata(outputs: Sequence[OutputFile]) -> list[dict[str, str]]:
    return [{"role": out.role, "file": out.path.name, "sha256": sha256_file(out.path)} for out in outputs]


def sidecar_step(record: StepRecord, catalog: ModelCatalog) -> dict[str, object]:
    entry = record.to_metadata()
    entry["model"] = None if record.model is None else _model_entry(record.model.to_metadata(), catalog)
    entry["auxiliaryModels"] = [_model_entry(model.to_metadata(), catalog) for model in record.aux_models]
    return entry


def build_sidecar(
    pre: PreResult, post: PostResult, context: SidecarContext, catalog: ModelCatalog
) -> dict[str, object]:
    records = (*pre.records, *post.records)
    facts = facts_from_records(records, context.upscale)
    summary = restore_metadata(pre, post)
    return {
        "schemaVersion": SIDECAR_SCHEMA_VERSION,
        "upflow": {"version": context.app_version, "commit": context.commit},
        "input": context.input.to_metadata(),
        "geometry": dict(context.geometry),
        "privacy": privacy_metadata(context.privacy, context.photo_date),
        "steps": [sidecar_step(record, catalog) for record in records],
        "damage": summary["damage"],
        "faces": summary["faces"],
        "upscale": context.upscale.to_metadata(),
        "colorize": summary["colorize"],
        "outputs": outputs_metadata(context.outputs),
        "effectiveBits": effective_bits(context.output_bit_depth, records),
        "aiApplied": any(used_a_model(record) for record in records) or context.upscale.mode == "ai",
        "inventsDetail": any(record.invents_detail for record in records) or facts.generative_upscale,
        "digitalSourceType": digital_source_type(facts),
        "compositeReasons": list(composite_reasons(facts)),
        "environment": dict(context.environment),
        "cpuFallback": summary["cpuFallback"],
        "calibration": summary["calibration"],
        "warnings": summary["warnings"],
    }


def used_a_model(record: StepRecord) -> bool:
    # Un detector (p. ej. BOPBTL) que decide que pixeles se rellenan tambien es IA, aunque el relleno sea Telea.
    return record.model is not None or bool(record.aux_models)


def uncolored_sidecar(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    # La copia sin color no lleva la colorizacion: su XMP y su insignia salen de estos hechos.
    steps = [step for step in sidecar.get("steps") or [] if step.get("id") != "colorize"]
    base = {**sidecar, "steps": steps, "colorize": None}
    facts = facts_from_sidecar(base)
    return {**base, "digitalSourceType": digital_source_type(facts), "compositeReasons": list(composite_reasons(facts))}


def uncolored_marks(sidecar: Mapping[str, Any], badge_requested: bool, photo_date: str | None) -> tuple[str, bool]:
    view = uncolored_sidecar(sidecar)
    xmp = build_xmp_packet(xmp_fields(view, photo_date))
    return xmp, badge_applies(facts_from_sidecar(view), badge_requested)


def recomposed_sidecar(
    sidecar: Mapping[str, Any],
    face_states: Mapping[int, tuple[bool, float]],
    outputs: Sequence[OutputFile],
    recomposed_at: str,
) -> dict[str, object]:
    faces = [_recomposed_face(face, face_states, recomposed_at) for face in sidecar.get("faces") or []]
    updated = {**sidecar, "faces": faces, "outputs": outputs_metadata(outputs)}
    facts = facts_from_sidecar(updated)
    reasons = list(composite_reasons(facts))
    return {**updated, "digitalSourceType": digital_source_type(facts), "compositeReasons": reasons}


def privacy_metadata(privacy: MetadataPrivacy | None, photo_date: str | None) -> dict[str, object]:
    return {
        "gpsRemoved": bool(privacy and privacy.gps_removed),
        "dateTimeOriginalMovedToDigitized": bool(privacy and privacy.date_time_original_moved),
        "approximatePhotoDate": photo_date,
    }


def xmp_fields(sidecar: Mapping[str, Any], photo_date: str | None = None) -> XmpFields:
    return XmpFields(
        digital_source_type=str(sidecar["digitalSourceType"]),
        description=restoration_description(sidecar),
        creator_tool=f"Upflow {sidecar['upflow']['version']}",
        date_created=photo_date,
    )


def restoration_description(sidecar: Mapping[str, Any]) -> str:
    steps = [STEP_PLAIN_NAMES.get(step["id"], step["id"]) for step in sidecar.get("steps") or []]
    upscale = UPSCALE_PLAIN_NAMES.get((sidecar.get("upscale") or {}).get("mode", "none"))
    names = [*steps, *([upscale] if upscale else [])]
    return "Restored with Upflow: " + (", ".join(names) if names else "no changes") + "."


def write_sidecar(path: Path, sidecar: Mapping[str, Any]) -> None:
    payload = json.dumps(sidecar, indent=2, ensure_ascii=False).encode("utf-8")
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def sanitize_stem(name: str) -> str:
    stem = Path(name.replace("\\", "/").rsplit("/", 1)[-1]).stem
    cleaned = re.sub(r"_+", "_", STEM_UNSAFE.sub("_", stem)).strip(" ._")
    return cleaned[:STEM_MAX_CHARS].rstrip(" .") or STEM_FALLBACK


def download_names(original_name: str, fmt: str, colorized: bool) -> DownloadNames:
    stem = sanitize_stem(original_name)
    extension = EXTENSIONS.get(fmt.lower().lstrip("."))
    if extension is None:
        raise ValueError(f"Unknown output format {fmt!r}")
    role = "colorized" if colorized else "restored"
    return DownloadNames(
        restored=f"{stem}_{role}.{extension}",
        uncolored=f"{stem}_uncolored.{extension}",
        before_after=f"{stem}_before-after.jpg",
        sidecar=f"{stem}_restore.json",
    )


def with_badge(rgb: np.ndarray) -> np.ndarray:
    marked = np.array(rgb, dtype=np.float32, copy=True)
    height, width = marked.shape[:2]
    if min(height, width) < BADGE_MIN_IMAGE_PX:
        return marked
    overlay = badge_overlay(height, width)
    margin = _badge_margin(height, width)
    x0, y0 = width - overlay.shape[1] - margin, height - overlay.shape[0] - margin
    region = marked[y0 : y0 + overlay.shape[0], x0 : x0 + overlay.shape[1]]
    alpha = overlay[:, :, 3:4]
    region += alpha * (overlay[:, :, :3] - region)
    return marked


def badge_overlay(height: int, width: int) -> np.ndarray:
    font = ImageFont.load_default(size=_badge_font_px(height, width))
    left, top, right, bottom = font.getbbox(BADGE_TEXT)
    pad = max(2, (bottom - top) // 3)
    canvas = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, canvas.width - 1, canvas.height - 1), radius=pad, fill=BADGE_BACKGROUND)
    draw.text((pad - left, pad - top), BADGE_TEXT, font=font, fill=BADGE_TEXT_COLOR)
    return np.asarray(canvas, dtype=np.float32) / np.float32(255.0)


def before_after_image(before: np.ndarray, after: np.ndarray, *, badge: bool) -> np.ndarray:
    height = min(after.shape[0], BEFORE_AFTER_MAX_HEIGHT)
    right = _fit_height(after, height)
    right = with_badge(right) if badge else right
    divider = np.ones((height, BEFORE_AFTER_DIVIDER_PX, 3), dtype=np.float32)
    joined = np.concatenate([_fit_height(before, height), divider, right], axis=1)
    return np.round(np.clip(joined, 0.0, 1.0) * 255.0).astype(np.uint8)


def encode_jpeg(pixels: np.ndarray, icc: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    options = {"quality": JPEG_QUALITY} if icc is None else {"quality": JPEG_QUALITY, "icc_profile": icc}
    Image.fromarray(pixels, mode="RGB").save(buffer, format="JPEG", **options)
    return buffer.getvalue()


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image.astype(np.float32, copy=False)
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    resized = Image.fromarray(np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


def _badge_font_px(height: int, width: int) -> int:
    size = int(np.clip(round(min(height, width) * BADGE_FONT_FRACTION), BADGE_MIN_FONT_PX, BADGE_MAX_FONT_PX))
    text_width = ImageFont.load_default(size=size).getlength(BADGE_TEXT)
    fit = size * BADGE_MAX_WIDTH_FRACTION * width / max(text_width, 1.0)
    return max(BADGE_MIN_FONT_PX, min(size, int(fit)))


def _badge_margin(height: int, width: int) -> int:
    return max(2, _badge_font_px(height, width) // 2)


def _models_of(record: StepRecord) -> tuple[Any, ...]:
    return tuple(model for model in (record.model, *record.aux_models) if model is not None)


def _model_entry(model: dict[str, object], catalog: ModelCatalog) -> dict[str, object]:
    model_id, precision = str(model["id"]), str(model["precision"])
    return {
        **model,
        "sha256": catalog.sha256_of(model_id, precision),
        "expectedSha256": catalog.expected_sha256_of(model_id, precision),
        "license": catalog.license_of(model_id),
    }


def _recomposed_face(face: Mapping[str, Any], states: Mapping[int, tuple[bool, float]], at: str) -> dict[str, object]:
    state = states.get(int(face["index"]))
    if state is None:
        return dict(face)
    enabled, blend = state
    return {**face, "enabled": enabled, "blend": blend, "recomposedAt": at}

