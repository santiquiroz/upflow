from __future__ import annotations

import contextlib
import json
import math
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.engines.drunet_restore import CPU_DEVICE, CPU_PRECISION, CpuFallback, reference_device
from app.services.engines.photo_restore_engine import PhotoRestoreEngine, ReadyInfer, is_gpu_device
from app.services.engines.tiled_restore_runner import (
    TDR_BUDGET_REASON,
    CalibrationCache,
    CalibrationKey,
    CalibrationSpec,
    Clock,
    RestoreCancelled,
    TileCalibration,
    TileProgress,
    calibrate_tile,
)
from app.services.face_geometry import (
    FACE_SIZE,
    align_face,
    align_matrix,
    inverse_paste_matrix,
    paste_box,
    template_mask,
    warp_to_box,
)
from app.services.photo_dsp import neutral_gray, wavelet_color_transfer
from app.services.photo_restore_presets import FACE_MODEL_DEFAULT, ToneKind

FACE_MODEL_ID = FACE_MODEL_DEFAULT
FACE_CHANNELS = 3
MONOCHROME_TONES = frozenset({"mono", "toned"})
ARTIFACT_DIR_SUFFIX = ".restore"
BEFORE_FACES_NAME = "before-faces.png"
FACES_MANIFEST_NAME = "faces.json"
MANIFEST_SCHEMA_VERSION = 1
LOW_DISK_WARNING = "recompose_disabled_low_disk"
WRITE_FAILED_WARNING = "recompose_disabled_write_failed"
PNG16_PEAK = 65535.0
PNG16_BYTES_PER_PIXEL = 2 * FACE_CHANNELS
CROPS_PER_FACE = 2
# Un PNG casi incompresible (grano, ruido) pesa un poco mas que los pixeles crudos.
PNG_OVERHEAD = 1.05
SCALE_TOLERANCE_PX = 1

DiskProbe = Callable[[Path], int | None]


@dataclass(frozen=True, slots=True)
class FaceTarget:
    index: int
    landmarks: tuple[tuple[float, float], ...]
    blend: float
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class FaceRestoreContext:
    engine: PhotoRestoreEngine
    calibrations: CalibrationCache
    budget_ms: float
    clock: Clock = time.perf_counter
    cancel_event: threading.Event | None = None
    on_progress: TileProgress | None = None


@dataclass(frozen=True, slots=True)
class FaceModel:
    ready: ReadyInfer
    cpu_fallback: CpuFallback | None = None


@dataclass(frozen=True, slots=True)
class RestoredFace:
    index: int
    matrix: np.ndarray
    blend: float
    aligned: np.ndarray
    restored: np.ndarray


@dataclass(frozen=True, slots=True)
class FacePatch:
    crop: np.ndarray
    matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class FaceRestoreResult:
    image: np.ndarray
    faces: tuple[RestoredFace, ...]
    scale: float
    model_id: str | None = None
    device: str | None = None
    precision: str | None = None
    cpu_fallback: CpuFallback | None = None


@dataclass(frozen=True, slots=True)
class RecomposeArtifacts:
    directory: Path | None
    warning: str | None = None

    @property
    def available(self) -> bool:
        return self.directory is not None


def restore_faces(
    context: FaceRestoreContext,
    source: np.ndarray,
    base: np.ndarray,
    targets: Sequence[FaceTarget],
    *,
    tone_kind: ToneKind,
    device: str,
    model_id: str = FACE_MODEL_ID,
) -> FaceRestoreResult:
    _require_rgb_float(source, "Source")
    _require_rgb_float(base, "Output base")
    scale = output_scale(source.shape, base.shape)
    selected = [face for face in targets if face.enabled]
    for face in selected:
        _require_blend(face.blend)
    if not selected:
        return FaceRestoreResult(image=base, faces=(), scale=scale)
    monochrome = is_monochrome(tone_kind)
    _raise_if_cancelled(context.cancel_event)
    sample = face_model_input(_aligned(source, selected[0]), monochrome)
    model = prepare_face_model(context, model_id, device, sample)
    faces = _restore_each(context, model, source, selected, monochrome)
    image = paste_faces(base, [blended_patch(face) for face in faces], scale)
    return FaceRestoreResult(
        image=image,
        faces=faces,
        scale=scale,
        model_id=model_id,
        device=model.ready.device,
        precision=model.ready.precision,
        cpu_fallback=model.cpu_fallback,
    )


def is_monochrome(tone_kind: ToneKind) -> bool:
    return tone_kind in MONOCHROME_TONES


def face_model_input(aligned: np.ndarray, monochrome: bool) -> np.ndarray:
    # Los modelos de caras se entrenaron con color natural: una dominante sesga la piel reconstruida.
    return neutral_gray(aligned) if monochrome else aligned


def match_face_tone(restored: np.ndarray, aligned: np.ndarray, monochrome: bool) -> np.ndarray:
    luminance = neutral_gray(restored) if monochrome else restored
    return wavelet_color_transfer(luminance, aligned)


def blend_face(restored: np.ndarray, aligned: np.ndarray, alpha: float) -> np.ndarray:
    _require_blend(alpha)
    if alpha == 1.0:
        return restored
    return (aligned + np.float32(alpha) * (restored - aligned)).astype(np.float32, copy=False)


def blended_patch(face: RestoredFace) -> FacePatch:
    return FacePatch(blend_face(face.restored, face.aligned, face.blend), face.matrix)


def output_scale(source_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> float:
    scale = output_shape[0] / source_shape[0]
    if abs(source_shape[1] * scale - output_shape[1]) > SCALE_TOLERANCE_PX:
        raise ValueError(f"Output {output_shape[:2]} is not a uniform rescale of {source_shape[:2]}")
    return scale


def paste_faces(base: np.ndarray, patches: Sequence[FacePatch], scale: float) -> np.ndarray:
    canvas = np.array(base, dtype=np.float32, copy=True)
    mask = template_mask()
    for patch in patches:
        _paste_into(canvas, patch, mask, scale)
    return canvas


def prepare_face_model(context: FaceRestoreContext, model_id: str, device: str, sample: np.ndarray) -> FaceModel:
    ready = context.engine.ready_infer(
        model_id,
        device,
        sample,
        reference_device=reference_device(context.calibrations, model_id, device),
    )
    if not is_gpu_device(device):
        return FaceModel(ready)
    calibration = _calibration(context, ready)
    if calibration.tile is not None:
        return FaceModel(ready)
    return _cpu_face_model(context, model_id, calibration)


def fixed_face_calibration_spec() -> CalibrationSpec:
    return CalibrationSpec(tile_min=FACE_SIZE, fixed_shape=True, channels=FACE_CHANNELS)


def restore_artifact_dir(outputs_path: Path, job_id: str) -> Path:
    return outputs_path / f"{job_id}{ARTIFACT_DIR_SUFFIX}"


def recompose_artifact_bytes(base_shape: tuple[int, ...], face_count: int) -> int:
    pixels = base_shape[0] * base_shape[1] + face_count * CROPS_PER_FACE * FACE_SIZE * FACE_SIZE
    return math.ceil(pixels * PNG16_BYTES_PER_PIXEL * PNG_OVERHEAD)


def disk_free_bytes(path: Path) -> int | None:
    # El directorio del job todavia no existe: se mide en el primer ancestro que si.
    probe = next((candidate for candidate in (path, *path.parents) if candidate.exists()), None)
    if probe is None:
        return None
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def save_recompose_artifacts(
    directory: Path,
    before_faces: np.ndarray,
    result: FaceRestoreResult,
    *,
    free_bytes: DiskProbe = disk_free_bytes,
) -> RecomposeArtifacts:
    if not _has_room(directory, recompose_artifact_bytes(before_faces.shape, len(result.faces)), free_bytes):
        return RecomposeArtifacts(None, LOW_DISK_WARNING)
    try:
        _write_artifacts(directory, before_faces, result)
    except OSError:
        _remove_artifacts(directory, result)
        return RecomposeArtifacts(None, WRITE_FAILED_WARNING)
    return RecomposeArtifacts(directory)


def faces_manifest(before_faces: np.ndarray, result: FaceRestoreResult) -> dict[str, object]:
    height, width = before_faces.shape[:2]
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "scale": result.scale,
        "size": {"width": width, "height": height},
        "beforeFaces": BEFORE_FACES_NAME,
        "faces": [_face_entry(face) for face in result.faces],
    }


def face_crop_names(index: int) -> tuple[str, str]:
    return f"face-{index}-aligned.png", f"face-{index}-restored.png"


def artifact_names(result: FaceRestoreResult) -> tuple[str, ...]:
    crops = (name for face in result.faces for name in face_crop_names(face.index))
    return (BEFORE_FACES_NAME, *crops, FACES_MANIFEST_NAME)


def write_unit_png16(path: Path, rgb: np.ndarray) -> None:
    pixels = np.round(np.clip(rgb, 0.0, 1.0) * PNG16_PEAK).astype(np.uint16)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not encode {path.name} as PNG")
    # imencode + write_bytes: cv2.imwrite no abre rutas con caracteres fuera de ASCII en Windows.
    path.write_bytes(encoded.tobytes())


def read_unit_png16(path: Path) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.uint16:
        raise ValueError(f"{path.name} is not a 16-bit PNG")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(PNG16_PEAK)


def _restore_each(
    context: FaceRestoreContext,
    model: FaceModel,
    source: np.ndarray,
    selected: Sequence[FaceTarget],
    monochrome: bool,
) -> tuple[RestoredFace, ...]:
    faces = []
    for done, target in enumerate(selected, start=1):
        _raise_if_cancelled(context.cancel_event)
        faces.append(_restore_one(model, source, target, monochrome))
        if context.on_progress is not None:
            context.on_progress(done, len(selected))
    return tuple(faces)


def _restore_one(model: FaceModel, source: np.ndarray, target: FaceTarget, monochrome: bool) -> RestoredFace:
    matrix = align_matrix(np.asarray(target.landmarks))
    aligned = align_face(source, matrix)
    raw = model.ready.infer(face_model_input(aligned, monochrome))
    restored = match_face_tone(raw, aligned, monochrome)
    return RestoredFace(target.index, matrix, target.blend, aligned, restored)


def _aligned(source: np.ndarray, target: FaceTarget) -> np.ndarray:
    return align_face(source, align_matrix(np.asarray(target.landmarks)))


def _paste_into(canvas: np.ndarray, patch: FacePatch, mask: np.ndarray, scale: float) -> None:
    inverse = inverse_paste_matrix(patch.matrix, scale)
    box = paste_box(inverse, canvas.shape[:2])
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    face = warp_to_box(patch.crop, inverse, box)
    weight = warp_to_box(mask, inverse, box)[:, :, np.newaxis]
    region = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = region + weight * (face - region)


def _calibration(context: FaceRestoreContext, ready: ReadyInfer) -> TileCalibration:
    key = CalibrationKey(ready.model_id, ready.device, ready.precision)
    return context.calibrations.get_or_calibrate(
        key,
        lambda: calibrate_tile(
            ready.infer,
            fixed_face_calibration_spec(),
            precision=ready.precision,
            budget_ms=context.budget_ms,
            clock=context.clock,
        ),
    )


def _cpu_face_model(context: FaceRestoreContext, model_id: str, calibration: TileCalibration) -> FaceModel:
    context.engine.begin_phase(CPU_DEVICE)
    infer = context.engine.tile_infer(model_id, CPU_DEVICE, CPU_PRECISION)
    reason = calibration.cpu_fallback_reason or TDR_BUDGET_REASON
    return FaceModel(ReadyInfer(model_id, CPU_DEVICE, CPU_PRECISION, infer), CpuFallback(model_id, reason))


def _has_room(directory: Path, needed: int, free_bytes: DiskProbe) -> bool:
    free = free_bytes(directory)
    return free is None or free >= needed


def _write_artifacts(directory: Path, before_faces: np.ndarray, result: FaceRestoreResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_unit_png16(directory / BEFORE_FACES_NAME, before_faces)
    for face in result.faces:
        aligned_name, restored_name = face_crop_names(face.index)
        write_unit_png16(directory / aligned_name, face.aligned)
        write_unit_png16(directory / restored_name, face.restored)
    manifest = json.dumps(faces_manifest(before_faces, result), indent=2)
    (directory / FACES_MANIFEST_NAME).write_text(manifest, encoding="utf-8")


def _remove_artifacts(directory: Path, result: FaceRestoreResult) -> None:
    # Solo lo propio: la colorizacion guarda ab_512.npy en el mismo directorio.
    for name in artifact_names(result):
        with contextlib.suppress(OSError):
            (directory / name).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        directory.rmdir()


def _face_entry(face: RestoredFace) -> dict[str, object]:
    aligned_name, restored_name = face_crop_names(face.index)
    return {
        "index": face.index,
        "blend": face.blend,
        "matrix": face.matrix.tolist(),
        "aligned": aligned_name,
        "restored": restored_name,
    }


def _require_rgb_float(image: np.ndarray, name: str) -> None:
    if image.ndim != 3 or image.shape[2] != FACE_CHANNELS or image.dtype != np.float32:
        raise ValueError(f"{name} must be a float32 RGB image, got {image.dtype} {image.shape}")


def _require_blend(alpha: float) -> None:
    if not (math.isfinite(alpha) and 0.0 <= alpha <= 1.0):
        raise ValueError(f"Face blend must be within [0, 1], got {alpha}")


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Restoration cancelled")
