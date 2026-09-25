from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.services.engines.face_detect import FaceDetection, landmarked_face_detector
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.scratch_detect import probability_to_u8, scratch_detector
from app.services.image_io import LoadedImage, load_image_for_restore
from app.services.photo_diagnosis import (
    DAMAGE_THRESHOLD,
    PORTRAIT_BLEND,
    DamageDetector,
    DetectedFace,
    PhotoDiagnosis,
    diagnose_photo,
)
from app.services.photo_geometry import Geometry
from app.services.photo_restore_pipeline import FaceSelection, PixelLimits, check_pixel_limits
from app.services.photo_restore_runners import detected_selections
from app.services.restore_outputs import fit_long_side, to_uint8
from app.services.restore_provenance import encode_jpeg, sha256_file

SESSION_PREFIX = "restore-"
SESSION_SCHEMA_VERSION = 1
SESSION_NAME = "session.json"
ORIGINAL_STEM = "original"
PREVIEW_NAME = "preview.jpg"
DAMAGE_PROB_NAME = "damage_prob.png"
DAMAGE_MASK_NAME = "damage_mask.png"
FACES_NAME = "faces.json"
DIAGNOSIS_NAME = "diagnosis.json"
PREVIEW_MAX_SIDE = 2048
FACE_THUMB_SIDE = 256
FACE_THUMB_MARGIN = 0.3
MASK_MODES = frozenset({"1", "L"})
MASK_ON = 255
JPEG_FORMATS = frozenset({"JPEG", "MPO"})
TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")
FACE_THUMB_PATTERN = re.compile(r"face-(?P<index>\d{1,3})\.jpg")
PUBLIC_FILES = frozenset({PREVIEW_NAME, DAMAGE_PROB_NAME})

LandmarkDetector = Callable[[np.ndarray], Sequence[FaceDetection]]
ImageLoader = Callable[[Path], LoadedImage]


class SessionNotFound(LookupError):
    pass


class InvalidMask(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AnalysisDetectors:
    damage: DamageDetector | None = None
    faces: LandmarkDetector | None = None


DetectorsFactory = Callable[[], AnalysisDetectors]


@dataclass(frozen=True, slots=True)
class SessionRecord:
    token: str
    original_name: str
    original_file: str
    sha256: str
    size_bytes: int
    bit_depth: int
    has_icc: bool
    jpeg_origin: bool
    width: int
    height: int
    geometry: Geometry = field(default_factory=Geometry)
    owner_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schemaVersion": SESSION_SCHEMA_VERSION,
            "token": self.token,
            "originalName": self.original_name,
            "originalFile": self.original_file,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "bitDepth": self.bit_depth,
            "hasIcc": self.has_icc,
            "jpegOrigin": self.jpeg_origin,
            "width": self.width,
            "height": self.height,
            "geometry": self.geometry.to_dict(),
            "ownerId": self.owner_id,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> SessionRecord:
        return cls(
            token=str(raw["token"]),
            original_name=str(raw["originalName"]),
            original_file=_plain_name(str(raw["originalFile"])),
            sha256=str(raw["sha256"]),
            size_bytes=int(raw["sizeBytes"]),
            bit_depth=int(raw["bitDepth"]),
            has_icc=bool(raw["hasIcc"]),
            jpeg_origin=bool(raw["jpegOrigin"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            geometry=Geometry.from_mapping(raw.get("geometry")),
            owner_id=raw.get("ownerId"),
        )


@dataclass(frozen=True, slots=True)
class WorkingAnalysis:
    diagnosis: PhotoDiagnosis
    probability: np.ndarray | None
    faces: tuple[FaceSelection, ...]


@dataclass(frozen=True, slots=True)
class SessionAnalysis:
    record: SessionRecord
    diagnosis: PhotoDiagnosis
    faces: tuple[FaceSelection, ...]
    damage_over_faces: bool
    has_damage_map: bool


@dataclass(frozen=True, slots=True)
class SessionInputs:
    original: Path
    geometry: Geometry
    damage_probability: np.ndarray | None
    user_mask: np.ndarray | None
    faces: tuple[FaceSelection, ...]


def is_valid_token(token: str) -> bool:
    return TOKEN_PATTERN.fullmatch(token) is not None


def session_dir(root: Path, token: str) -> Path:
    if not is_valid_token(token):
        raise SessionNotFound("Unknown restore session")
    return root / f"{SESSION_PREFIX}{token}"


def default_detectors(settings: Settings, engine: PhotoRestoreEngine) -> DetectorsFactory:
    # Los packs se pueden bajar con la app andando: se miran en cada analisis. Ambos detectores
    # corren siempre en CPU (§3.4.2, §3.4.7), asi el analisis nunca compite por la GPU.
    def build() -> AnalysisDetectors:
        return AnalysisDetectors(
            damage=scratch_detector(engine) if settings.restore_core_installed else None,
            faces=landmarked_face_detector(engine) if settings.restore_faces_installed else None,
        )

    return build


def analyze_working_copy(rgb: np.ndarray, detectors: AnalysisDetectors, *, jpeg_origin: bool) -> WorkingAnalysis:
    probability = None if detectors.damage is None else detectors.damage(rgb)
    detections = None if detectors.faces is None else tuple(detectors.faces(rgb))
    diagnosis = diagnose_photo(
        rgb,
        jpeg_origin=jpeg_origin,
        damage_detector=None if probability is None else (lambda _image: probability),
        face_detector=None if detections is None else _replayed_faces(rgb.shape[:2], detections),
    )
    faces = () if not detections else detected_selections(rgb, detections, PORTRAIT_BLEND)
    return WorkingAnalysis(diagnosis, probability, faces)


def _replayed_faces(
    shape: tuple[int, int], detections: Sequence[FaceDetection]
) -> Callable[[np.ndarray], list[DetectedFace]]:
    # diagnose_photo detecta sobre una copia chica y reescala: se le devuelven en esa escala
    # las caras ya detectadas en la copia de trabajo, para no correr el detector dos veces.
    def replay(small: np.ndarray) -> list[DetectedFace]:
        factor = small.shape[0] / shape[0]
        return [face.scaled(factor).to_detected_face() for face in detections]

    return replay


def damage_over_faces(probability: np.ndarray | None, faces: Sequence[FaceSelection]) -> bool:
    if probability is None:
        return False
    return any(_damage_in_box(probability, face.box) for face in faces if face.box is not None)


def _damage_in_box(probability: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    height, width = probability.shape[:2]
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
    return bool((probability[y0:y1, x0:x1] > DAMAGE_THRESHOLD).any())


def preview_jpeg(rgb: np.ndarray, icc: bytes | None) -> bytes:
    return encode_jpeg(fit_long_side(to_uint8(rgb), PREVIEW_MAX_SIDE), icc)


def face_thumbnail(rgb: np.ndarray, box: tuple[float, float, float, float], icc: bytes | None) -> bytes:
    height, width = rgb.shape[:2]
    margin = FACE_THUMB_MARGIN * max(box[2] - box[0], box[3] - box[1])
    x0, y0 = max(0, int(box[0] - margin)), max(0, int(box[1] - margin))
    x1, y1 = min(width, int(np.ceil(box[2] + margin))), min(height, int(np.ceil(box[3] + margin)))
    crop = rgb[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    return encode_jpeg(fit_long_side(to_uint8(crop), FACE_THUMB_SIDE), icc)


def mask_from_png(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if image.format != "PNG" or image.mode not in MASK_MODES:
                raise InvalidMask("The damage mask must be a black and white PNG")
            pixels = np.asarray(image.convert("L"))
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidMask("The damage mask must be a black and white PNG") from exc
    if pixels.shape != shape:
        raise InvalidMask(
            f"The damage mask is {pixels.shape[1]}x{pixels.shape[0]}; the photo is {shape[1]}x{shape[0]}"
        )
    if not np.isin(pixels, (0, MASK_ON)).all():
        raise InvalidMask("The damage mask must only contain black (keep) and white (repair) pixels")
    return pixels == MASK_ON


def mask_to_png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.where(mask, MASK_ON, 0).astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def faces_to_json(faces: Sequence[FaceSelection]) -> list[dict[str, Any]]:
    return [
        {
            "index": face.index,
            "landmarks": [list(point) for point in face.landmarks],
            "box": None if face.box is None else list(face.box),
            "score": face.score,
            "eyePx": face.eye_px,
            "sharpness": face.sharpness,
            "enabled": face.enabled,
            "blend": face.blend,
        }
        for face in faces
    ]


def faces_from_json(raw: Sequence[dict[str, Any]]) -> tuple[FaceSelection, ...]:
    return tuple(
        FaceSelection(
            index=int(entry["index"]),
            landmarks=tuple((float(x), float(y)) for x, y in entry["landmarks"]),
            blend=float(entry["blend"]),
            enabled=bool(entry["enabled"]),
            box=None if entry.get("box") is None else tuple(float(value) for value in entry["box"]),
            score=entry.get("score"),
            eye_px=entry.get("eyePx"),
            sharpness=entry.get("sharpness"),
        )
        for entry in raw
    )


def diagnosis_to_json(diagnosis: PhotoDiagnosis) -> dict[str, Any]:
    return {
        "findings": [_finding_json(finding) for finding in diagnosis.findings],
        "proposedPreset": diagnosis.proposed_preset,
        "proposedSteps": list(diagnosis.proposed_steps),
        "suggestedPresets": list(diagnosis.suggested_presets),
        "toneKind": diagnosis.facts.tone_kind,
        "eta": {"gpuSeconds": diagnosis.estimate.gpu_seconds, "cpuSeconds": diagnosis.estimate.cpu_seconds},
        "damage": None
        if diagnosis.damage is None
        else {"coverage": diagnosis.damage.coverage, "largeAreas": diagnosis.damage.large_areas},
    }


def _finding_json(finding: Any) -> dict[str, Any]:
    return {
        "key": finding.key,
        "value": finding.value,
        "reasonKey": finding.reason_key,
        "params": dict(finding.params),
        "proposes": [
            {"stepId": proposal.step_id, "options": dict(proposal.options), "enabled": proposal.enabled}
            for proposal in finding.proposes
        ],
        "missingPack": finding.missing_pack,
        "message": finding.message,
    }


def _plain_name(name: str) -> str:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise SessionNotFound("The restore session is damaged")
    return name


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _image_format(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            return image.format
    except Image.DecompressionBombError as exc:
        raise ValueError("Uploaded image exceeds the maximum allowed dimensions") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image") from exc


class RestoreSessionStore:
    def __init__(
        self,
        settings: Settings,
        detectors: DetectorsFactory,
        *,
        load_image: ImageLoader = load_image_for_restore,
    ) -> None:
        self._root = settings.video_work_path
        self._limits = PixelLimits.from_settings(settings)
        self._detectors = detectors
        self._load_image = load_image
        self._semaphore = asyncio.Semaphore(settings.restore_analysis_concurrency)

    async def open(self, upload: Path, original_name: str, owner_id: str | None = None) -> SessionAnalysis:
        token = uuid4().hex
        directory = session_dir(self._root, token)
        try:
            async with self._semaphore:
                return await asyncio.to_thread(self._open, token, directory, upload, original_name, owner_id)
        except Exception:
            await asyncio.to_thread(shutil.rmtree, directory, True)
            raise

    async def set_geometry(self, token: str, geometry: Geometry) -> SessionAnalysis:
        record = self.record(token)
        async with self._semaphore:
            return await asyncio.to_thread(self._analyze, replace(record, geometry=geometry))

    def record(self, token: str) -> SessionRecord:
        path = session_dir(self._root, token) / SESSION_NAME
        if not path.is_file():
            raise SessionNotFound("Unknown restore session")
        return SessionRecord.from_json(json.loads(path.read_text(encoding="utf-8")))

    def geometry_of(self, token: str) -> dict[str, Any]:
        return self.record(token).geometry.to_dict()

    def original_path(self, token: str) -> Path:
        return session_dir(self._root, token) / self.record(token).original_file

    def file(self, token: str, name: str) -> Path:
        directory = session_dir(self._root, token)
        if name not in PUBLIC_FILES and FACE_THUMB_PATTERN.fullmatch(name) is None:
            raise SessionNotFound(f"Unknown session file {name!r}")
        path = directory / name
        if not path.is_file():
            raise SessionNotFound(f"The session has no {name}")
        return path

    def save_mask(self, token: str, data: bytes) -> float:
        record = self.record(token)
        mask = mask_from_png(data, (record.height, record.width))
        (session_dir(self._root, token) / DAMAGE_MASK_NAME).write_bytes(mask_to_png(mask))
        return float(mask.mean())

    def job_inputs(self, token: str) -> SessionInputs:
        record = self.record(token)
        directory = session_dir(self._root, token)
        faces_path = directory / FACES_NAME
        faces = faces_from_json(json.loads(faces_path.read_text(encoding="utf-8"))) if faces_path.is_file() else ()
        return SessionInputs(
            original=directory / record.original_file,
            geometry=record.geometry,
            damage_probability=_read_probability(directory / DAMAGE_PROB_NAME),
            user_mask=_read_mask(directory / DAMAGE_MASK_NAME),
            faces=faces,
        )

    def _open(
        self, token: str, directory: Path, upload: Path, original_name: str, owner_id: str | None
    ) -> SessionAnalysis:
        directory.mkdir(parents=True, exist_ok=False)
        original = directory / f"{ORIGINAL_STEM}{Path(original_name).suffix.lower()}"
        shutil.move(str(upload), str(original))
        image_format = _image_format(original)
        record = SessionRecord(
            token=token,
            original_name=original_name,
            original_file=original.name,
            sha256=sha256_file(original),
            size_bytes=original.stat().st_size,
            bit_depth=8,
            has_icc=False,
            jpeg_origin=image_format in JPEG_FORMATS,
            width=0,
            height=0,
            owner_id=owner_id,
        )
        return self._analyze(record)

    def _analyze(self, record: SessionRecord) -> SessionAnalysis:
        directory = session_dir(self._root, record.token)
        loaded = self._load_checked(directory / record.original_file)
        working = record.geometry.apply(loaded.rgb)
        analysis = analyze_working_copy(working, self._detectors(), jpeg_origin=record.jpeg_origin)
        updated = replace(
            record,
            bit_depth=loaded.bit_depth,
            has_icc=loaded.icc is not None,
            width=working.shape[1],
            height=working.shape[0],
        )
        self._write_analysis(directory, updated, working, loaded.icc, analysis)
        return SessionAnalysis(
            record=updated,
            diagnosis=analysis.diagnosis,
            faces=analysis.faces,
            damage_over_faces=damage_over_faces(analysis.probability, analysis.faces),
            has_damage_map=analysis.probability is not None,
        )

    def _load_checked(self, path: Path) -> LoadedImage:
        with Image.open(path) as image:
            width, height = image.size
        check_pixel_limits(height, width, 1.0, self._limits)
        return self._load_image(path)

    def _write_analysis(
        self,
        directory: Path,
        record: SessionRecord,
        working: np.ndarray,
        icc: bytes | None,
        analysis: WorkingAnalysis,
    ) -> None:
        _remove_derived_files(directory)
        (directory / PREVIEW_NAME).write_bytes(preview_jpeg(working, icc))
        if analysis.probability is not None:
            Image.fromarray(probability_to_u8(analysis.probability), mode="L").save(directory / DAMAGE_PROB_NAME)
        for face in analysis.faces:
            if face.box is not None:
                (directory / f"face-{face.index}.jpg").write_bytes(face_thumbnail(working, face.box, icc))
        _write_json(directory / FACES_NAME, faces_to_json(analysis.faces))
        _write_json(directory / DIAGNOSIS_NAME, diagnosis_to_json(analysis.diagnosis))
        # session.json se escribe al final: una sesion sin el es una sesion que no termino.
        _write_json(directory / SESSION_NAME, record.to_json())


def _remove_derived_files(directory: Path) -> None:
    # Otra geometria invalida la mascara pintada y las miniaturas: se borran antes de rehacerlas.
    for name in (DAMAGE_PROB_NAME, DAMAGE_MASK_NAME, FACES_NAME, DIAGNOSIS_NAME, PREVIEW_NAME):
        (directory / name).unlink(missing_ok=True)
    for thumbnail in directory.glob("face-*.jpg"):
        thumbnail.unlink(missing_ok=True)


def _read_probability(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / np.float32(255.0)


def _read_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) == MASK_ON
