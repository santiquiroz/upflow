from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import cv2
import numpy as np
from scipy import ndimage

from app.services.missing_pack import missing_pack_message
from app.services.photo_capture import CaptureSigns, capture_signs
from app.services.photo_dsp import (
    FIX_FADED_DEFAULT_STRENGTH,
    LUMA_WEIGHTS,
    PeriodicPeaks,
    find_periodic_peaks,
    neutral_axis_band_means,
    screen_period,
)
from app.services.photo_restore_chain import CORE_PACK, FACES_PACK
from app.services.photo_restore_presets import (
    DAMAGE_MIN_COVERAGE,
    FACE_MODEL_DEFAULT,
    FADED_CAST_MIN_DE,
    KEEP_GRAIN_DEFAULT,
    NOISE_MIN_SIGMA,
    REPAIR_SENSITIVITY_MEDIUM,
    PhotoFacts,
    ToneKind,
    proposed_preset,
    resolve_preset,
    suggested_presets,
)

PatternKind = Literal["halftone", "texture"]

# Umbrales de §3.2, todos [propuesta]: se recalibran en P4-BENCH.
ANALYSIS_MAX_PIXELS = 1_000_000
MONO_MAX_CHROMA_P95 = 12.0
TONED_MIN_MEAN_CHROMA = 3.0
TONED_MAX_HUE_DISPERSION = 0.35
HUE_MIN_CHROMA = 1.0
HAND_TINT_MAX_FRACTION = 0.25
NATIVE_CROP_SIDE = 1024
MAD_TO_SIGMA = 0.6745
DENOISE_FULL_STRENGTH_SIGMA = 25.0 / 255.0
DENOISE_MIN_STRENGTH = 0.1
JPEG_BLOCK = 8
BLOCKING_MIN_RATIO = 1.3
BLOCK_STEP_CAP = 8.0
BLOCK_MIN_STEP = 0.5
HALFTONE_HARMONIC_RATIO = 1.3
HALFTONE_MIN_HARMONICS = 2
SHARPNESS_ANALYSIS_SIDE = 1024
BLUR_MAX_SHARPNESS = 0.01
FLAT_MAX_VARIANCE = 1e-6
DAMAGE_THRESHOLD = 0.4
LARGE_HOLE_PX = 24.0
FACE_ANALYSIS_SIDE = 1280
FACE_MIN_SCORE = 0.9
PORTRAIT_BLEND = 0.6
DEBLOCK_DEFAULT_STRENGTH = 0.5
DESCREEN_DEFAULT_STRENGTH = 1.0
RATIO_EPSILON = 1e-6

CAST_HUE_START_DEG = -30.0
# Limite superior de cada sector del angulo de tono en Lab, empezando en -30 grados.
CAST_SECTORS: tuple[tuple[float, str], ...] = (
    (65.0, "reddish"),
    (115.0, "yellow"),
    (235.0, "cyan/green"),
    (290.0, "blue"),
    (330.0, "magenta"),
)

_TONE_REASONS: Mapping[str, str] = MappingProxyType(
    {
        "mono": "restore.diag.mono",
        "toned": "restore.diag.sepia",
        "hand_tinted": "restore.diag.handTinted",
        "color": "restore.diag.color",
    }
)


@dataclass(frozen=True, slots=True)
class StepProposal:
    step_id: str
    options: Mapping[str, Any] = field(default_factory=dict)
    # False = sugerido: la UI lo muestra pero el usuario lo enciende (colorize, faces).
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class Finding:
    key: str
    value: float | int | str | None
    reason_key: str
    proposes: tuple[StepProposal, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    missing_pack: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class ToneAnalysis:
    kind: ToneKind
    chroma_p95: float
    tinted_fraction: float


@dataclass(frozen=True, slots=True)
class Blocking:
    ratio: float
    # Exceso medio de la frontera sobre el interior, en niveles de 8 bits.
    step: float

    @property
    def visible(self) -> bool:
        # Un JPEG de calidad 90+ deja escalones de una fraccion de nivel que nadie ve.
        return self.ratio > BLOCKING_MIN_RATIO and self.step > BLOCK_MIN_STEP


@dataclass(frozen=True, slots=True)
class PatternAnalysis:
    kind: PatternKind | None
    period: float | None
    peaks: PeriodicPeaks


@dataclass(frozen=True, slots=True)
class ColorCast:
    delta_e: float
    hue_deg: float
    name: str


@dataclass(frozen=True, slots=True)
class DamageStats:
    coverage: float
    large_areas: int


@dataclass(frozen=True, slots=True)
class DetectedFace:
    box: tuple[float, float, float, float]
    score: float
    eye_px: float

    def scaled(self, factor: float) -> DetectedFace:
        box = tuple(value * factor for value in self.box)
        return DetectedFace(box=box, score=self.score, eye_px=self.eye_px * factor)


@dataclass(frozen=True, slots=True)
class StepCost:
    gpu_seconds_per_mpx: float
    cpu_seconds_per_mpx: float
    gpu_seconds_per_item: float = 0.0
    cpu_seconds_per_item: float = 0.0


@dataclass(frozen=True, slots=True)
class TimeEstimate:
    gpu_seconds: float
    cpu_seconds: float


@dataclass(frozen=True, slots=True)
class PhotoDiagnosis:
    findings: tuple[Finding, ...]
    facts: PhotoFacts
    proposed_preset: str
    proposed_steps: tuple[str, ...]
    suggested_presets: tuple[str, ...]
    estimate: TimeEstimate
    pattern: PatternAnalysis
    faces: tuple[DetectedFace, ...]
    damage: DamageStats | None
    step_estimates: Mapping[str, TimeEstimate]


@dataclass(frozen=True, slots=True)
class _Measurements:
    tone: ToneAnalysis
    noise_sigma: float
    blocking: Blocking
    jpeg_origin: bool
    pattern: PatternAnalysis
    sharpness: float
    cast: ColorCast | None
    damage: DamageStats | None
    faces: tuple[DetectedFace, ...] | None
    capture: CaptureSigns | None = None


DamageDetector = Callable[[np.ndarray], np.ndarray]
FaceDetector = Callable[[np.ndarray], Sequence[DetectedFace]]

# [estimacion] CPU medida en port-restore-onnx con onnxruntime CPU (7900X3D, 2026-09-25):
# DRUNet 1,3 s por tile de 512 (~5 s/Mpx mas solape), detector BOPBTL 0,29 s a 256x320,
# GFPGAN 0,68 s por cara. GPU sin medir hasta P0-GPU. Los pasos DSP corren en CPU igual.
DEFAULT_STEP_COSTS: Mapping[str, StepCost] = MappingProxyType(
    {
        "descreen": StepCost(0.5, 0.5),
        "repair": StepCost(0.15, 1.5, gpu_seconds_per_item=0.3, cpu_seconds_per_item=0.3),
        "deblock": StepCost(0.8, 6.0),
        "denoise": StepCost(0.8, 6.0),
        "tone": StepCost(0.3, 0.3),
        "faces": StepCost(0.0, 0.0, gpu_seconds_per_item=0.15, cpu_seconds_per_item=0.9),
        "colorize": StepCost(0.2, 0.2, gpu_seconds_per_item=0.5, cpu_seconds_per_item=4.0),
    }
)


def diagnose_photo(
    rgb: np.ndarray,
    *,
    jpeg_origin: bool = False,
    damage_detector: DamageDetector | None = None,
    face_detector: FaceDetector | None = None,
    costs: Mapping[str, StepCost] = DEFAULT_STEP_COSTS,
) -> PhotoDiagnosis:
    _require_rgb(rgb)
    measures = _measure(rgb, jpeg_origin, damage_detector, face_detector)
    facts = _facts(measures)
    preset = proposed_preset(facts)
    steps = resolve_preset(preset, facts).steps
    faces = measures.faces or ()
    items = {"faces": len(faces)}
    return PhotoDiagnosis(
        findings=_findings(measures),
        facts=facts,
        proposed_preset=preset,
        proposed_steps=steps,
        suggested_presets=suggested_presets(facts),
        estimate=estimate_restore_time(megapixels(rgb), steps, costs, items=items),
        pattern=measures.pattern,
        faces=faces,
        damage=measures.damage,
        step_estimates=step_time_estimates(megapixels(rgb), costs, items),
    )


def _measure(
    rgb: np.ndarray,
    jpeg_origin: bool,
    damage_detector: DamageDetector | None,
    face_detector: FaceDetector | None,
) -> _Measurements:
    tone = classify_tone(rgb)
    return _Measurements(
        tone=tone,
        noise_sigma=estimate_noise_sigma(rgb),
        blocking=measure_blocking(rgb),
        jpeg_origin=jpeg_origin,
        pattern=analyze_pattern(rgb),
        sharpness=measure_sharpness(rgb),
        cast=measure_color_cast(rgb) if tone.kind == "color" else None,
        damage=damage_stats(damage_detector(rgb), rgb.shape[:2]) if damage_detector else None,
        faces=detect_faces(rgb, face_detector) if face_detector else None,
        capture=capture_signs(rgb),
    )


def _facts(measures: _Measurements) -> PhotoFacts:
    return PhotoFacts(
        damage_coverage=measures.damage.coverage if measures.damage else 0.0,
        jpeg_blocking=_is_blocky(measures),
        noise_sigma=measures.noise_sigma,
        halftone=measures.pattern.kind == "halftone",
        tone_kind=measures.tone.kind,
        color_cast_de=measures.cast.delta_e if measures.cast else 0.0,
        max_eye_px=measures.faces[0].eye_px if measures.faces else 0.0,
    )


def _findings(measures: _Measurements) -> tuple[Finding, ...]:
    candidates = (
        _tone_finding(measures.tone),
        _pattern_finding(measures.pattern),
        _damage_finding(measures.damage),
        _blocking_finding(measures),
        _noise_finding(measures.noise_sigma),
        _blur_finding(measures.sharpness),
        _faded_finding(measures.cast),
        _faces_finding(measures.faces),
        _capture_finding(measures.capture),
    )
    return tuple(item for item in candidates if item is not None)


def _require_rgb(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Photo diagnosis needs an RGB image with 3 channels, got shape {rgb.shape}")


def megapixels(rgb: np.ndarray) -> float:
    return rgb.shape[0] * rgb.shape[1] / 1_000_000


def classify_tone(rgb: np.ndarray) -> ToneAnalysis:
    ab = _to_lab(_analysis_sample(rgb))[:, 1:]
    # Croma desde el tono base: un sepia fuerte sigue siendo un solo tono (research-color-danos §2.4).
    relative = _chroma(ab - np.median(ab, axis=0))
    tinted = relative > MONO_MAX_CHROMA_P95
    p95 = float(np.percentile(relative, 95))
    kind = _tone_kind(ab, tinted, p95)
    return ToneAnalysis(kind=kind, chroma_p95=p95, tinted_fraction=float(tinted.mean()))


def _tone_kind(ab: np.ndarray, tinted: np.ndarray, p95: float) -> ToneKind:
    if p95 < MONO_MAX_CHROMA_P95:
        return "toned" if _is_toned(ab) else "mono"
    if tinted.mean() <= HAND_TINT_MAX_FRACTION and _is_neutral_or_toned(ab[~tinted]):
        return "hand_tinted"
    return "color"


def _is_toned(ab: np.ndarray) -> bool:
    return _chroma(ab).mean() > TONED_MIN_MEAN_CHROMA and _hue_dispersion(ab) < TONED_MAX_HUE_DISPERSION


def _is_neutral_or_toned(ab: np.ndarray) -> bool:
    return _chroma(ab).mean() <= TONED_MIN_MEAN_CHROMA or _hue_dispersion(ab) < TONED_MAX_HUE_DISPERSION


def _hue_dispersion(ab: np.ndarray) -> float:
    hued = ab[_chroma(ab) > HUE_MIN_CHROMA]
    if len(hued) == 0:
        return 1.0
    angles = np.arctan2(hued[:, 1], hued[:, 0])
    return float(1.0 - np.hypot(np.cos(angles).mean(), np.sin(angles).mean()))


def _chroma(ab: np.ndarray) -> np.ndarray:
    return np.hypot(ab[:, 0], ab[:, 1])


def _analysis_sample(rgb: np.ndarray) -> np.ndarray:
    pixels = rgb.reshape(-1, 3)
    stride = max(1, int(np.ceil(len(pixels) / ANALYSIS_MAX_PIXELS)))
    return pixels[::stride]


def _to_lab(pixels: np.ndarray) -> np.ndarray:
    column = np.ascontiguousarray(pixels.reshape(-1, 1, 3), dtype=np.float32)
    return cv2.cvtColor(np.clip(column, 0.0, 1.0), cv2.COLOR_RGB2Lab).reshape(-1, 3)


def estimate_noise_sigma(rgb: np.ndarray) -> float:
    # A resolucion nativa: reducir la foto promedia el ruido y lo esconde.
    luma = _luma(_native_crop(rgb))
    rows, cols = (size // 2 * 2 for size in luma.shape)
    if rows == 0 or cols == 0:
        return 0.0
    even = luma[:rows, :cols]
    diagonal = (even[0::2, 0::2] - even[0::2, 1::2] - even[1::2, 0::2] + even[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(diagonal)) / MAD_TO_SIGMA)


def measure_blocking(rgb: np.ndarray) -> Blocking:
    luma = _luma(_native_crop(rgb)) * 255.0
    horizontal = _grid_blocking(np.abs(np.diff(luma, axis=1)))
    vertical = _grid_blocking(np.abs(np.diff(luma, axis=0)).T)
    return Blocking(ratio=(horizontal.ratio + vertical.ratio) / 2.0, step=(horizontal.step + vertical.step) / 2.0)


def _grid_blocking(differences: np.ndarray) -> Blocking:
    usable = differences.shape[1] // JPEG_BLOCK * JPEG_BLOCK
    if usable == 0:
        return Blocking(ratio=1.0, step=0.0)
    # Un escalon de bloque mide pocos niveles; sin tope, los bordes reales de la foto pesan mas que la grilla.
    capped = np.minimum(differences[:, :usable], BLOCK_STEP_CAP)
    by_offset = capped.reshape(differences.shape[0], -1, JPEG_BLOCK).mean(axis=(0, 1))
    # La fase de la grilla se busca: un recorte o un giro de la sesion la corre.
    boundary = float(by_offset.max())
    interior = float(np.delete(by_offset, int(np.argmax(by_offset))).mean())
    return Blocking(ratio=boundary / max(interior, RATIO_EPSILON), step=boundary - interior)


def _is_blocky(measures: _Measurements) -> bool:
    return measures.jpeg_origin and measures.blocking.visible


def analyze_pattern(rgb: np.ndarray) -> PatternAnalysis:
    peaks = find_periodic_peaks(rgb)
    return PatternAnalysis(kind=_pattern_kind(peaks), period=screen_period(peaks), peaks=peaks)


def _pattern_kind(peaks: PeriodicPeaks) -> PatternKind | None:
    if len(peaks.frequencies) == 0:
        return None
    radii = np.hypot(peaks.frequencies[:, 0], peaks.frequencies[:, 1])
    # Un punto binario reparte energia en armonicos; una textura de papel es casi sinusoidal.
    harmonics = int(np.sum(radii >= radii.min() * HALFTONE_HARMONIC_RATIO))
    return "halftone" if harmonics >= HALFTONE_MIN_HARMONICS else "texture"


def measure_sharpness(rgb: np.ndarray) -> float:
    luma = _luma(_fit_within(rgb, SHARPNESS_ANALYSIS_SIDE)[0])
    contrast = float(luma.var())
    if contrast <= FLAT_MAX_VARIANCE:
        return float("inf")
    return float(cv2.Laplacian(luma, cv2.CV_32F).var()) / contrast


def measure_color_cast(rgb: np.ndarray) -> ColorCast:
    ab = _to_lab(neutral_axis_band_means(rgb))[:, 1:]
    hue = float(np.degrees(np.arctan2(ab[:, 1].mean(), ab[:, 0].mean())))
    return ColorCast(delta_e=float(_chroma(ab).mean()), hue_deg=hue, name=cast_name(hue))


def cast_name(hue_deg: float) -> str:
    wrapped = (hue_deg - CAST_HUE_START_DEG) % 360.0 + CAST_HUE_START_DEG
    return next(name for upper, name in CAST_SECTORS if wrapped < upper)


def damage_stats(probability: np.ndarray, native_shape: tuple[int, int]) -> DamageStats:
    mask = probability > DAMAGE_THRESHOLD
    if not mask.any():
        return DamageStats(coverage=0.0, large_areas=0)
    return DamageStats(coverage=float(mask.mean()), large_areas=_count_large_areas(mask, native_shape))


def _count_large_areas(mask: np.ndarray, native_shape: tuple[int, int]) -> int:
    to_native = float(np.sqrt(native_shape[0] * native_shape[1] / mask.size))
    padded = np.pad(mask, 1).astype(np.uint8)
    inside = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    labels, count = ndimage.label(mask)
    # Ancho de cada componente = el doble de su distancia maxima al borde (§3.4.2).
    widths = 2.0 * np.asarray(ndimage.maximum(inside, labels, np.arange(1, count + 1))) * to_native
    return int(np.sum(widths > LARGE_HOLE_PX))


def detect_faces(rgb: np.ndarray, detector: FaceDetector) -> tuple[DetectedFace, ...]:
    small, factor = _fit_within(rgb, FACE_ANALYSIS_SIDE)
    found = [face.scaled(1.0 / factor) for face in detector(small) if face.score >= FACE_MIN_SCORE]
    return tuple(sorted(found, key=lambda face: face.eye_px, reverse=True))


def estimate_restore_time(
    megapixel_count: float,
    steps: Sequence[str],
    costs: Mapping[str, StepCost] = DEFAULT_STEP_COSTS,
    items: Mapping[str, int] | None = None,
) -> TimeEstimate:
    missing = [step for step in steps if step not in costs]
    if missing:
        raise ValueError(f"No time cost for restore steps: {', '.join(missing)}")
    counts = items or {}
    step_costs = [(costs[step], counts.get(step, 1)) for step in steps]
    return TimeEstimate(
        gpu_seconds=sum(_gpu_seconds(cost, megapixel_count, count) for cost, count in step_costs),
        cpu_seconds=sum(_cpu_seconds(cost, megapixel_count, count) for cost, count in step_costs),
    )


def step_time_estimates(
    megapixel_count: float,
    costs: Mapping[str, StepCost] = DEFAULT_STEP_COSTS,
    items: Mapping[str, int] | None = None,
) -> dict[str, TimeEstimate]:
    return {step: estimate_restore_time(megapixel_count, (step,), costs, items) for step in costs}


def _gpu_seconds(cost: StepCost, megapixel_count: float, count: int) -> float:
    return cost.gpu_seconds_per_mpx * megapixel_count + cost.gpu_seconds_per_item * count


def _cpu_seconds(cost: StepCost, megapixel_count: float, count: int) -> float:
    return cost.cpu_seconds_per_mpx * megapixel_count + cost.cpu_seconds_per_item * count


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb, dtype=np.float32) @ LUMA_WEIGHTS


def _native_crop(rgb: np.ndarray) -> np.ndarray:
    top, left = (max(0, (size - NATIVE_CROP_SIDE) // 2) for size in rgb.shape[:2])
    return rgb[top : top + NATIVE_CROP_SIDE, left : left + NATIVE_CROP_SIDE]


def _fit_within(rgb: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    factor = min(1.0, side / max(rgb.shape[:2]))
    if factor == 1.0:
        return rgb, 1.0
    size = (max(1, round(rgb.shape[1] * factor)), max(1, round(rgb.shape[0] * factor)))
    resized = cv2.resize(np.ascontiguousarray(rgb, dtype=np.float32), size, interpolation=cv2.INTER_AREA)
    return resized, factor


def _tone_finding(tone: ToneAnalysis) -> Finding:
    return Finding("tone", tone.kind, _TONE_REASONS[tone.kind], proposes=_tone_proposals(tone.kind))


def _tone_proposals(kind: ToneKind) -> tuple[StepProposal, ...]:
    if kind not in ("mono", "toned"):
        return ()
    keep = StepProposal("tone", {"keep_tone": True, "neutral_gray": False, "fix_faded": False})
    return keep, StepProposal("colorize", enabled=False)


def _pattern_finding(pattern: PatternAnalysis) -> Finding | None:
    if pattern.kind == "halftone":
        options = {"mode": "halftone", "period": pattern.period, "strength": DESCREEN_DEFAULT_STRENGTH}
        return Finding("pattern", pattern.kind, "restore.diag.halftone", (StepProposal("descreen", options),))
    if pattern.kind == "texture":
        options = {"mode": "texture", "strength": DESCREEN_DEFAULT_STRENGTH}
        return Finding("pattern", pattern.kind, "restore.diag.texture", (StepProposal("descreen", options),))
    return None


def _damage_finding(damage: DamageStats | None) -> Finding | None:
    if damage is None:
        return _pack_missing_finding("damage", CORE_PACK)
    if damage.coverage <= DAMAGE_MIN_COVERAGE:
        return None
    repair = StepProposal("repair", {"engine": "fast", "sensitivity": REPAIR_SENSITIVITY_MEDIUM, "grow_px": 1})
    params = {"pct": round(damage.coverage * 100.0, 1), "largeAreas": damage.large_areas}
    return Finding("damage", damage.coverage, "restore.diag.damage", (repair,), params)


def _blocking_finding(measures: _Measurements) -> Finding | None:
    if not _is_blocky(measures):
        return None
    deblock = StepProposal("deblock", {"strength": DEBLOCK_DEFAULT_STRENGTH})
    params = {"ratio": round(measures.blocking.ratio, 2), "step": round(measures.blocking.step, 2)}
    return Finding("jpeg_blocking", measures.blocking.ratio, "restore.diag.jpeg", (deblock,), params)


def _noise_finding(sigma: float) -> Finding | None:
    if sigma <= NOISE_MIN_SIGMA:
        return None
    denoise = StepProposal("denoise", {"strength": denoise_strength(sigma), "keep_grain": KEEP_GRAIN_DEFAULT})
    return Finding("noise", sigma, "restore.diag.noise", (denoise,))


def denoise_strength(sigma: float) -> float:
    return round(float(np.clip(sigma / DENOISE_FULL_STRENGTH_SIGMA, DENOISE_MIN_STRENGTH, 1.0)), 2)


def _blur_finding(sharpness: float) -> Finding | None:
    if sharpness >= BLUR_MAX_SHARPNESS:
        return None
    # Solo informativo: el deblur es P4 (§3.2).
    return Finding("blur", sharpness, "restore.diag.blur")


def _faded_finding(cast: ColorCast | None) -> Finding | None:
    if cast is None or cast.delta_e <= FADED_CAST_MIN_DE:
        return None
    options = {"strength": FIX_FADED_DEFAULT_STRENGTH, "keep_tone": True, "neutral_gray": False, "fix_faded": True}
    return Finding("faded", cast.delta_e, "restore.diag.faded", (StepProposal("tone", options),), {"cast": cast.name})


def _faces_finding(faces: tuple[DetectedFace, ...] | None) -> Finding | None:
    if faces is None:
        return _pack_missing_finding("faces", FACES_PACK)
    if not faces:
        return None
    restore = StepProposal("faces", {"model": FACE_MODEL_DEFAULT, "blend": PORTRAIT_BLEND}, enabled=False)
    return Finding("faces", len(faces), "restore.diag.faces", (restore,), {"count": len(faces)})


# Solo informa: el consejo (reescanear a 600 dpi o fotografiar en angulo) no es un paso.
def _capture_finding(signs: CaptureSigns | None) -> Finding | None:
    if signs is None or not signs.phone_capture:
        return None
    params = {"glare": int(signs.glare), "perspective": int(signs.perspective)}
    return Finding("capture", "phone", "restore.diag.phoneCapture", params=params)


def _pack_missing_finding(key: str, pack: str) -> Finding:
    return Finding(
        key,
        None,
        "restore.diag.packMissing",
        params={"pack": pack},
        missing_pack=pack,
        message=missing_pack_message(pack),
    )
