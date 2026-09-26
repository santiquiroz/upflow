"""Validacion DML de los modelos de restauracion de fotos (spec §6.1: P0-GPU y P1-GPU-1).

Por modelo y precision: sesion con create_session(..., prefer_native=False), nodos
que ORT manda a CPU (perfilado), calibracion TDR del runner, mediana de 5 llamadas
al tile elegido, canario contra DML fp32 y contra CPU fp32, NaN/Inf, vram_factor y
si el grafo necesita ORT_DISABLE_ALL. Escribe la tabla markdown que commitea P0-GPU.

Usa la GPU: en serie, con la GPU libre y sin jobs activos en el server :8090.

    $env:UPFLOW_GPU_TESTS="1"
    python scripts/spike_restore_dml.py --models-dir C:/personal/port-restore-onnx/dist
    python scripts/spike_restore_dml.py --installed --expect-report docs/superpowers/specs/<fecha>-restore-dml-validation.md
        (el informe nuevo queda al lado, como <fecha>-restore-dml-validation-rerun.md)
    python scripts/spike_restore_dml.py --models-dir <dir> --device cpu    # ensayo sin GPU
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for _path in (REPO, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import restore_dml_probes as probes  # noqa: E402
# Reexportados: los tests y P1-GPU-1 usan el informe a través de este módulo.
from restore_dml_report import (  # noqa: E402, F401
    DML_DISABLE_GRAPH_FUSION,
    MODE_DEFAULT,
    MODE_DISABLE_ALL,
    MODE_NO_FUSION,
    RESULT_COLUMNS,
    SESSION_MODES,
    SPEC_COLUMNS,
    ReportContext,
    SpecSuggestion,
    median_issue,
    precision_issues,
    render_report,
    report_differences,
    report_rows,
    suggest_specs,
    verdict,
)

from app.services import ep_registry  # noqa: E402
from app.services.dml_device import try_parse_dml_device_id  # noqa: E402
from app.services.engines.onnx_video_upscaler import is_device_removed_error  # noqa: E402
from app.services.engines.frame_model_runner import OrtValueFactory, dml_ortvalue  # noqa: E402
from app.services.engines.restore_canary import (  # noqa: E402
    DELTA_E,
    CanaryRule,
    canary_rule_for,
    canary_score,
    has_non_finite,
)
from app.services.engines.tiled_restore_runner import (  # noqa: E402
    CalibrationSpec,
    TileCalibration,
    TileInfer,
    calibrate_tile,
    session_tile_infer,
)
from app.services.onnx_cpu_fallback_probe import build_synthetic_inputs, hot_cpu_ops  # noqa: E402
from app.services.restore_models import (  # noqa: E402
    RESTORE_BUNDLES,
    RESTORE_MODELS,
    RestoreBundle,
    RestoreModelSpec,
    bundle_installed,
)

GPU_TESTS_ENV = "UPFLOW_GPU_TESTS"
CPU_DEVICE = "cpu"
DML_PROVIDER = ep_registry.DML_PROVIDER
PRECISION_ORDER = ("fp32", "fp16")
MEDIAN_CALLS = 5
DEFAULT_TILE_MIN = 128
DEFAULT_TILE_CANDIDATES = (256, 384, 512)
LOCK_NAME = "models.lock.json"
FP16_SUFFIXES = ("-fp16", "_fp16")
FLOAT_INPUT = "tensor(float)"
NCHW_RANK = 4
REPORT_DIR = REPO / "docs" / "superpowers" / "specs"
MIB = 1024 * 1024
# El 4.o canal de DRUNet es un mapa de nivel (ruido o calidad JPEG), no imagen.
CONDITION_CHANNEL_VALUE = 0.1
CANARY_SEED = 20260925
# Rec. 709: DDColor recibe el RGB de Lab(L, 0, 0), o sea una imagen gris.
GRAY_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

CPU_BY_DESIGN = {
    "bopbtl-scratch-detector": "detector de daño: CPU por diseño (§3.4.2)",
    "retinaface-r34": "detector de caras: CPU por diseño (§3.4.7)",
}

Clock = Callable[[], float]
OpenSession = Callable[[Path, str, str], Any]
CpuNodes = Callable[[Path, str], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ModelFile:
    precision: str
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ModelTarget:
    model_id: str
    files: tuple[ModelFile, ...]

    def file(self, precision: str) -> ModelFile | None:
        return next((item for item in self.files if item.precision == precision), None)


@dataclass(frozen=True, slots=True)
class Skipped:
    model_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class GraphContract:
    channels: int
    fixed_side: int | None


@dataclass(frozen=True, slots=True)
class PrecisionResult:
    model_id: str
    precision: str
    file: ModelFile
    providers: tuple[str, ...] = ()
    cpu_nodes: tuple[str, ...] = ()
    calibration: TileCalibration | None = None
    median_ms: float | None = None
    vs_fp32: float | None = None
    vs_cpu: float | None = None
    non_finite: bool = False
    vram_factor: float | None = None
    session_mode: str = MODE_DEFAULT
    error: str | None = None
    issues: tuple[str, ...] = ()
    output: np.ndarray | None = field(default=None, repr=False, compare=False)
    # Solo el grafo uint8 de video: cuadros enteros medidos y si IOBinding siguió activo.
    frames: tuple[probes.FrameTiming, ...] = ()
    io_binding: bool | None = None


@dataclass(frozen=True, slots=True)
class Harness:
    device: str
    budget_ms: float
    open_session: OpenSession
    cpu_nodes: CpuNodes
    vram_usage_mb: Callable[[], int | None]
    clock: Clock = time.perf_counter
    tile_min: int = DEFAULT_TILE_MIN
    tile_candidates: tuple[int, ...] = DEFAULT_TILE_CANDIDATES
    catalog: Mapping[str, RestoreModelSpec] = field(default_factory=lambda: RESTORE_MODELS)
    ortvalue_factory: OrtValueFactory = dml_ortvalue


@dataclass(frozen=True, slots=True)
class VideoReference:
    spec: CalibrationSpec
    canary: np.ndarray
    output: np.ndarray


def canary_tile(side: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(CANARY_SEED)
    grid = np.linspace(0.0, 1.0, side, dtype=np.float32)
    planes = [_canary_plane(grid, index, rng) for index in range(min(channels, 3))]
    planes += [np.full((side, side), CONDITION_CHANNEL_VALUE, dtype=np.float32)] * max(channels - 3, 0)
    return np.stack(planes, axis=-1)


def canary_for(model_id: str, side: int, channels: int) -> np.ndarray:
    tile = canary_tile(side, channels)
    if canary_rule_for(model_id).metric != DELTA_E:
        return tile
    gray = tile[:, :, :3] @ GRAY_WEIGHTS
    return np.ascontiguousarray(np.repeat(gray[:, :, np.newaxis], channels, axis=-1), dtype=np.float32)


def _canary_plane(grid: np.ndarray, index: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.meshgrid(grid, grid, indexing="ij")
    smooth = 0.5 + 0.35 * np.sin(2 * np.pi * (index + 1) * xs) * np.cos(2 * np.pi * (index + 2) * ys)
    texture = rng.normal(0.0, 0.03, size=smooth.shape)
    return np.clip(smooth + texture, 0.0, 1.0).astype(np.float32)


# --- contrato del grafo y calibracion -------------------------------------------


def graph_contract(inputs: Sequence[Any]) -> GraphContract | str:
    first = inputs[0]
    if first.type != FLOAT_INPUT:
        return f"contrato no soportado: entrada {first.type}"
    if len(inputs) > 1:
        return "contrato no soportado: más de una entrada"
    shape = list(first.shape)
    if len(shape) != NCHW_RANK or not isinstance(shape[1], int):
        return f"contrato no soportado: entrada {shape} no es NCHW"
    return _contract_from_nchw(shape)


def _contract_from_nchw(shape: list) -> GraphContract | str:
    height, width = shape[2], shape[3]
    if not (isinstance(height, int) and isinstance(width, int)):
        return GraphContract(channels=shape[1], fixed_side=None)
    if height != width:
        return f"contrato no soportado: forma fija no cuadrada {height}x{width}"
    return GraphContract(channels=shape[1], fixed_side=height)


def calibration_spec_for(harness: Harness, model_id: str, contract: GraphContract) -> CalibrationSpec:
    if contract.fixed_side is not None:
        return CalibrationSpec(tile_min=contract.fixed_side, fixed_shape=True, channels=contract.channels)
    spec = harness.catalog.get(model_id)
    tile_min = spec.tile_min if spec else harness.tile_min
    candidates = spec.tile_candidates if spec else harness.tile_candidates
    return CalibrationSpec(tile_min=tile_min, tile_candidates=candidates, channels=contract.channels)


def median_call_ms(infer: TileInfer, tile: int | None, channels: int, clock: Clock) -> float | None:
    if tile is None:
        return None
    source = np.full((tile, tile, channels), 0.5, dtype=np.float32)
    return statistics.median(_timed_call_ms(infer, source, clock) for _ in range(MEDIAN_CALLS))


def _timed_call_ms(infer: TileInfer, source: np.ndarray, clock: Clock) -> float:
    started = clock()
    infer(source)
    return (clock() - started) * 1000.0


def vram_factor(before_mb: int | None, after_mb: int | None, file_bytes: int) -> float | None:
    if before_mb is None or after_mb is None or file_bytes <= 0:
        return None
    return round(max(after_mb - before_mb, 0) * MIB / file_bytes, 2)


# --- medicion ----------------------------------------------------------------


def validate_targets(harness: Harness, targets: Iterable[ModelTarget]) -> tuple[list[PrecisionResult], list[Skipped]]:
    results: list[PrecisionResult] = []
    skipped: list[Skipped] = []
    for target in targets:
        outcome = validate_target(harness, target)
        if isinstance(outcome, Skipped):
            skipped.append(outcome)
        else:
            results.extend(outcome)
    return results, skipped


def validate_target(harness: Harness, target: ModelTarget) -> list[PrecisionResult] | Skipped:
    if target.model_id in CPU_BY_DESIGN:
        return Skipped(target.model_id, CPU_BY_DESIGN[target.model_id])
    try:
        prepared = _prepare_reference(harness, target)
    except Exception as exc:  # noqa: BLE001 -- un grafo que ni carga en CPU queda anotado, el resto sigue
        if is_device_removed_error(exc):
            raise
        return Skipped(target.model_id, f"la referencia en CPU falla: {_one_line(exc)}")
    if isinstance(prepared, str):
        return Skipped(target.model_id, prepared)
    if isinstance(prepared, VideoReference):
        return _measure_video(harness, target, prepared)
    return _measure_precisions(harness, target, *prepared)


def reference_file(target: ModelTarget) -> ModelFile:
    # DDColor se publica solo con pesos fp16: su referencia es ese mismo grafo en CPU.
    return target.file("fp32") or target.file("fp16")


def _prepare_reference(
    harness: Harness, target: ModelTarget
) -> tuple[CalibrationSpec, np.ndarray, np.ndarray] | VideoReference | str:
    session = harness.open_session(reference_file(target).path, CPU_DEVICE, MODE_DEFAULT)
    if probes.is_video_contract(session.get_inputs()):
        return _video_reference(harness, target.model_id, session)
    contract = graph_contract(session.get_inputs())
    if isinstance(contract, str):
        return contract
    spec = calibration_spec_for(harness, target.model_id, contract)
    canary = canary_for(target.model_id, spec.tile_min, contract.channels)
    return spec, canary, session_tile_infer(session)(canary)


def _video_reference(harness: Harness, model_id: str, session: Any) -> VideoReference:
    spec = calibration_spec_for(harness, model_id, GraphContract(channels=3, fixed_side=None))
    canary = probes.u8_canary(canary_tile(spec.tile_min, 3))
    inference = probes.frame_inference(session, CPU_DEVICE, harness.ortvalue_factory)
    return VideoReference(spec, canary, probes.canary_output(inference, canary))


def _measure_precisions(
    harness: Harness, target: ModelTarget, spec: CalibrationSpec, canary: np.ndarray, reference: np.ndarray
) -> list[PrecisionResult]:
    rule = canary_rule_for(target.model_id)
    results: list[PrecisionResult] = []
    dml_fp32: np.ndarray | None = None
    for model_file in _files_to_measure(harness, target):
        measured = measure_with_fallback(harness, target.model_id, model_file, spec, canary)
        scored = score_result(measured, rule, reference, dml_fp32)
        if model_file.precision == "fp32" and not scored.non_finite:
            dml_fp32 = scored.output
        issues = precision_issues(scored, rule, harness.device, harness.budget_ms)
        results.append(replace(scored, issues=issues, output=None))
    return results


def _measure_video(harness: Harness, target: ModelTarget, reference: VideoReference) -> list[PrecisionResult]:
    rule = canary_rule_for(target.model_id)
    results: list[PrecisionResult] = []
    dml_fp32: np.ndarray | None = None
    for model_file in _files_to_measure(harness, target):
        measured = measure_with_fallback(
            harness, target.model_id, model_file, reference.spec, reference.canary, measure=measure_video_precision
        )
        scored = score_result(measured, rule, reference.output, dml_fp32)
        if model_file.precision == "fp32" and not scored.non_finite:
            dml_fp32 = scored.output
        issues = precision_issues(scored, rule, harness.device, harness.budget_ms)
        results.append(replace(scored, issues=issues, output=None))
    return results


def _files_to_measure(harness: Harness, target: ModelTarget) -> list[ModelFile]:
    files = [model_file for p in PRECISION_ORDER if (model_file := target.file(p)) is not None]
    # En CPU siempre corre un solo grafo: el fp32, o el unico que haya.
    return files[:1] if harness.device == CPU_DEVICE else files


MeasurePrecision = Callable[["Harness", str, ModelFile, CalibrationSpec, np.ndarray, str], PrecisionResult]


class SilentCpuFallback(RuntimeError):
    pass


def open_on_device(harness: Harness, path: Path, mode: str) -> Any:
    session = harness.open_session(path, harness.device, mode)
    providers = tuple(session.get_providers())
    # ORT cae a CPU sin lanzar si el EP no inicializa el grafo (p. ej. la fusion de DML): eso no es validar en DML.
    if harness.device != CPU_DEVICE and providers[:1] != (DML_PROVIDER,):
        raise SilentCpuFallback(f"ORT abrió la sesión en {providers[0] if providers else 'ningún provider'}")
    return session


def measure_with_fallback(
    harness: Harness,
    model_id: str,
    model_file: ModelFile,
    spec: CalibrationSpec,
    canary: np.ndarray,
    measure: MeasurePrecision | None = None,
) -> PrecisionResult:
    measure = measure or measure_precision
    errors: list[Exception] = []
    for mode in SESSION_MODES:
        try:
            return measure(harness, model_id, model_file, spec, canary, mode)
        except Exception as exc:  # noqa: BLE001 -- el arnes anota el error del grafo y sigue con el resto
            if is_device_removed_error(exc):
                raise
            errors.append(exc)
    return PrecisionResult(model_id, model_file.precision, model_file, error=_one_line(errors[0]))


def measure_precision(
    harness: Harness, model_id: str, model_file: ModelFile, spec: CalibrationSpec, canary: np.ndarray, mode: str
) -> PrecisionResult:
    before = harness.vram_usage_mb()
    session = open_on_device(harness, model_file.path, mode)
    providers = tuple(session.get_providers())
    infer = session_tile_infer(session)
    calibration = calibrate_tile(
        infer, spec, precision=model_file.precision, budget_ms=harness.budget_ms, clock=harness.clock
    )
    median = median_call_ms(infer, calibration.tile, spec.channels, harness.clock)
    output = infer(canary)
    after = harness.vram_usage_mb()
    del infer, session
    return PrecisionResult(
        model_id,
        model_file.precision,
        model_file,
        providers=providers,
        cpu_nodes=harness.cpu_nodes(model_file.path, mode),
        calibration=calibration,
        median_ms=median,
        non_finite=has_non_finite(output),
        vram_factor=vram_factor(before, after, model_file.size),
        session_mode=mode,
        output=output,
    )


def measure_video_precision(
    harness: Harness, model_id: str, model_file: ModelFile, spec: CalibrationSpec, canary: np.ndarray, mode: str
) -> PrecisionResult:
    before = harness.vram_usage_mb()
    session = open_on_device(harness, model_file.path, mode)
    providers = tuple(session.get_providers())
    inference = probes.frame_inference(session, harness.device, harness.ortvalue_factory)
    planned_binding = inference.uses_iobinding
    calibration = probes.calibrate_video(inference, spec, model_file.precision, harness.budget_ms, harness.clock)
    median = probes.median_tile_ms(inference, calibration.tile, harness.clock)
    frames = tuple(
        probes.frame_timing(inference, calibration, shape, harness.budget_ms, harness.clock)
        for shape in probes.VIDEO_FRAME_SHAPES
    )
    output = probes.canary_output(inference, canary)
    after = harness.vram_usage_mb()
    # None = no aplica (CPU o una sesión sin io_binding); False = se intentó y cayó a run común.
    io_binding = inference.uses_iobinding if planned_binding else None
    del inference, session
    return PrecisionResult(
        model_id,
        model_file.precision,
        model_file,
        providers=providers,
        cpu_nodes=harness.cpu_nodes(model_file.path, mode),
        calibration=calibration,
        median_ms=median,
        non_finite=has_non_finite(output),
        vram_factor=vram_factor(before, after, model_file.size),
        session_mode=mode,
        output=output,
        frames=frames,
        io_binding=io_binding,
    )


def score_result(
    result: PrecisionResult, rule: CanaryRule, reference: np.ndarray, dml_fp32: np.ndarray | None
) -> PrecisionResult:
    if result.output is None or result.non_finite:
        return result
    vs_cpu = canary_score(rule, reference, result.output)
    compare_fp32 = result.precision == "fp16" and dml_fp32 is not None
    vs_fp32 = canary_score(rule, dml_fp32, result.output) if compare_fp32 else None
    return replace(result, vs_cpu=vs_cpu, vs_fp32=vs_fp32)


def _one_line(exc: Exception) -> str:
    return f"{type(exc).__name__}: {' '.join(str(exc).split())}"


# --- descubrimiento de modelos ------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_file(path: Path, precision: str) -> ModelFile:
    return ModelFile(precision=precision, path=path, sha256=sha256_file(path), size=path.stat().st_size)


def _targets(entries: Iterable[tuple[str, str, Path]]) -> list[ModelTarget]:
    grouped: dict[str, dict[str, Path]] = {}
    for model_id, precision, path in entries:
        grouped.setdefault(model_id, {})[precision] = path
    return [
        ModelTarget(model_id, tuple(_model_file(paths[p], p) for p in PRECISION_ORDER if p in paths))
        for model_id, paths in grouped.items()
    ]


def discover_models_dir(models_dir: Path) -> list[ModelTarget]:
    lock = models_dir / LOCK_NAME
    if lock.is_file():
        return _targets(_lock_entries(models_dir, json.loads(lock.read_text(encoding="utf-8"))))
    return _targets(_named_entries(models_dir))


def _lock_entries(models_dir: Path, lock: Mapping[str, Any]) -> list[tuple[str, str, Path]]:
    return [(asset["model"], asset["precision"], models_dir / asset["file"]) for asset in lock["assets"]]


def _named_entries(models_dir: Path) -> list[tuple[str, str, Path]]:
    return [(*_id_and_precision(path.stem), path) for path in sorted(models_dir.glob("*.onnx"))]


def _id_and_precision(stem: str) -> tuple[str, str]:
    suffix = next((suffix for suffix in FP16_SUFFIXES if stem.endswith(suffix)), None)
    return (stem[: -len(suffix)], "fp16") if suffix else (stem, "fp32")


def installed_targets(
    model_dir: Path, models: Mapping[str, RestoreModelSpec], bundles: Mapping[str, RestoreBundle]
) -> list[ModelTarget]:
    installed = [bundle for bundle in bundles.values() if bundle_installed(model_dir, bundle)]
    return _targets(
        (spec.id, precision, model_dir / filename)
        for spec in models.values()
        if any(bundle.name == spec.bundle for bundle in installed)
        for precision, filename in spec.files_by_precision().items()
    )


def select_targets(targets: list[ModelTarget], only: Sequence[str]) -> list[ModelTarget]:
    return [target for target in targets if not only or target.model_id in only]


def probe_detectors(harness: Harness, targets: Iterable[ModelTarget]) -> list[probes.DetectorProbe]:
    if harness.device == CPU_DEVICE:
        return []
    return [probe_detector_target(harness, t) for t in targets if t.model_id in probes.DETECTOR_SHAPES]


def probe_detector_target(harness: Harness, target: ModelTarget) -> probes.DetectorProbe:
    model_file = reference_file(target)
    base = probes.DetectorProbe(
        target.model_id, model_file.path.name, model_file.sha256, probes.DETECTOR_SHAPES[target.model_id]
    )
    try:
        # El perfilado va antes de abrir la sesion medida: con RetinaFace op11, destruir una segunda
        # sesion DML del mismo grafo tira abajo la que sigue viva (access violation, medido en P0-GPU).
        cpu_nodes = harness.cpu_nodes(model_file.path, MODE_DEFAULT)
        reference = harness.open_session(model_file.path, CPU_DEVICE, MODE_DEFAULT)
        session = harness.open_session(model_file.path, harness.device, MODE_DEFAULT)
        return replace(base, **probes.probe_detector(target.model_id, reference, session, cpu_nodes, harness.clock))
    except Exception as exc:  # noqa: BLE001 -- dato informativo: el error queda en la fila
        if is_device_removed_error(exc):
            raise
        return replace(base, error=_one_line(exc))


# --- ORT real y CLI -----------------------------------------------------------


def restore_session_options(device: str, mode: str, profile: bool = False) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_profiling = profile
    if device != CPU_DEVICE:
        options.intra_op_num_threads = 1
    if mode == MODE_DISABLE_ALL:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if mode == MODE_NO_FUSION:
        options.add_session_config_entry(DML_DISABLE_GRAPH_FUSION, "1")
    return options


def open_restore_session(path: Path, device: str, settings: Any, mode: str, profile: bool = False) -> Any:
    return ep_registry.create_session(
        str(path),
        device,
        settings,
        sess_options_factory=lambda: restore_session_options(device, mode, profile),
        prefer_native=False,
    )


def profile_cpu_nodes(open_profiled: Callable[[], Any], device: str) -> tuple[str, ...]:
    if device == CPU_DEVICE:
        return ()
    session = open_profiled()
    session.run(None, build_synthetic_inputs(session.get_inputs()))
    profile = Path(session.end_profiling())
    try:
        return tuple(hot_cpu_ops(json.loads(profile.read_text(encoding="utf-8")), DML_PROVIDER))
    finally:
        profile.unlink(missing_ok=True)


def process_vram_mb(device: str) -> int | None:
    from app.services.devices_service import _query_adapter_vram_info_mb

    index = try_parse_dml_device_id(device)
    info = None if index is None else _query_adapter_vram_info_mb(index)
    return None if info is None else info[1]


def build_harness(args: argparse.Namespace, settings: Any, budget_ms: float) -> Harness:
    device = args.device
    return Harness(
        device=device,
        budget_ms=budget_ms,
        open_session=lambda path, on_device, mode: open_restore_session(path, on_device, settings, mode),
        cpu_nodes=lambda path, mode: profile_cpu_nodes(
            lambda: open_restore_session(path, device, settings, mode, profile=True), device
        ),
        vram_usage_mb=lambda: process_vram_mb(device),
        tile_min=args.tile_min,
        tile_candidates=args.tile_candidates,
    )


def gpu_allowed(device: str, environ: Mapping[str, str]) -> bool:
    return device == CPU_DEVICE or environ.get(GPU_TESTS_ENV) == "1"


def _tile_list(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--models-dir", type=Path, help="ONNX locales (lee models.lock.json si existe)")
    source.add_argument("--installed", action="store_true", help="bundles instalados en RESTORE_MODEL_DIR")
    parser.add_argument("--device", default="dml:0")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expect-report", type=Path, default=None)
    parser.add_argument("--models", type=lambda text: tuple(text.split(",")), default=())
    parser.add_argument("--budget-ms", type=float, default=None)
    parser.add_argument("--tile-min", type=int, default=DEFAULT_TILE_MIN)
    parser.add_argument("--tile-candidates", type=_tile_list, default=DEFAULT_TILE_CANDIDATES)
    parser.add_argument(
        "--probe-cpu-models", action="store_true", help="mide también en el device los detectores que corren en CPU"
    )
    return parser.parse_args(argv)


def _source_targets(args: argparse.Namespace, settings: Any) -> tuple[str, list[ModelTarget]]:
    if args.installed:
        model_dir = settings.restore_model_dir_path
        return str(model_dir), installed_targets(model_dir, RESTORE_MODELS, RESTORE_BUNDLES)
    return str(args.models_dir), discover_models_dir(args.models_dir)


def report_path_for(args: argparse.Namespace, today: str) -> Path:
    if args.report is not None:
        return args.report
    if args.expect_report is not None:
        # Nunca se pisa el informe contra el que se compara (misma fecha = mismo nombre).
        return args.expect_report.with_name(f"{args.expect_report.stem}-rerun.md")
    return REPORT_DIR / f"{today}-restore-dml-validation.md"


def _ort_version() -> str:
    import onnxruntime as ort

    return ort.__version__


def _tolerate_console_encoding() -> None:
    # La consola cp1252 de Windows no tiene "Δ" ni "∞": sin esto el resumen termina en traceback.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _tolerate_console_encoding()
    args = parse_args(argv)
    if not gpu_allowed(args.device, os.environ):
        print(f"{args.device} usa la GPU: correr con {GPU_TESTS_ENV}=1, en serie y con la GPU libre.", file=sys.stderr)
        return 2
    from app.config import get_settings

    settings = get_settings()
    budget_ms = args.budget_ms or float(settings.restore_call_budget_ms)
    expected = args.expect_report.read_text(encoding="utf-8") if args.expect_report else None
    source, targets = _source_targets(args, settings)
    harness = build_harness(args, settings, budget_ms)
    selected = select_targets(targets, args.models)
    results, skipped = validate_targets(harness, selected)
    detectors = probe_detectors(harness, selected) if args.probe_cpu_models else []
    today = dt.date.today().isoformat()
    context = ReportContext(today, args.device, source, _ort_version(), budget_ms)
    report = render_report(context, results, skipped, detectors)
    report_path = report_path_for(args, today)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _print_summary(report_path, results, skipped)
    return _compare_with_expected(expected, report)


def _print_summary(report_path: Path, results: Sequence[PrecisionResult], skipped: Sequence[Skipped]) -> None:
    for result in results:
        print(f"{result.model_id} {result.precision}: {verdict(result)}")
    for item in skipped:
        print(f"{item.model_id}: sin validar ({item.reason})")
    print(f"Informe: {report_path}")


def _compare_with_expected(expected: str | None, report: str) -> int:
    if expected is None:
        return 0
    differences = report_differences(expected, report)
    for line in differences:
        print(f"diferencia: {line}", file=sys.stderr)
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
