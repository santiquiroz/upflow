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
import math
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.services import ep_registry  # noqa: E402
from app.services.dml_device import try_parse_dml_device_id  # noqa: E402
from app.services.engines.onnx_video_upscaler import is_device_removed_error  # noqa: E402
from app.services.engines.restore_canary import (  # noqa: E402
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
NOT_AVAILABLE = "n/d"
# El 4.o canal de DRUNet es un mapa de nivel (ruido o calidad JPEG), no imagen.
CONDITION_CHANNEL_VALUE = 0.1
CANARY_SEED = 20260925

CPU_BY_DESIGN = {
    "bopbtl-scratch-detector": "detector de daño: CPU por diseño (§3.4.2)",
    "retinaface-r34": "detector de caras: CPU por diseño (§3.4.7)",
}

RESULT_COLUMNS = (
    "Modelo", "Precisión", "Archivo", "SHA-256", "Providers", "Nodos en CPU", "Tile", "ms/Mpx", "Mediana ms",
    "vs DML fp32", "vs CPU fp32", "NaN/Inf", "vram_factor", "ORT_DISABLE_ALL", "Veredicto",
)
SPEC_COLUMNS = ("Modelo", "tile_by_precision", "fp16_filename", "ort_disable_all", "vram_factor")
SKIPPED_COLUMNS = ("Modelo", "Motivo")

Clock = Callable[[], float]
OpenSession = Callable[[Path, str, bool], Any]
CpuNodes = Callable[[Path, bool], tuple[str, ...]]
RowKey = tuple[str, str]


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
    ort_disable_all: bool = False
    error: str | None = None
    issues: tuple[str, ...] = ()
    output: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class SpecSuggestion:
    model_id: str
    tile_by_precision: Mapping[str, int]
    fp16_filename: str | None
    ort_disable_all: bool
    vram_factor: float | None


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


@dataclass(frozen=True, slots=True)
class ReportContext:
    date: str
    device: str
    source: str
    ort_version: str
    budget_ms: float


def canary_tile(side: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(CANARY_SEED)
    grid = np.linspace(0.0, 1.0, side, dtype=np.float32)
    planes = [_canary_plane(grid, index, rng) for index in range(min(channels, 3))]
    planes += [np.full((side, side), CONDITION_CHANNEL_VALUE, dtype=np.float32)] * max(channels - 3, 0)
    return np.stack(planes, axis=-1)


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
    return _measure_precisions(harness, target, *prepared)


def reference_file(target: ModelTarget) -> ModelFile:
    # DDColor se publica solo con pesos fp16: su referencia es ese mismo grafo en CPU.
    return target.file("fp32") or target.file("fp16")


def _prepare_reference(harness: Harness, target: ModelTarget) -> tuple[CalibrationSpec, np.ndarray, np.ndarray] | str:
    session = harness.open_session(reference_file(target).path, CPU_DEVICE, False)
    contract = graph_contract(session.get_inputs())
    if isinstance(contract, str):
        return contract
    spec = calibration_spec_for(harness, target.model_id, contract)
    canary = canary_tile(spec.tile_min, contract.channels)
    return spec, canary, session_tile_infer(session)(canary)


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


def _files_to_measure(harness: Harness, target: ModelTarget) -> list[ModelFile]:
    files = [model_file for p in PRECISION_ORDER if (model_file := target.file(p)) is not None]
    # En CPU siempre corre un solo grafo: el fp32, o el unico que haya.
    return files[:1] if harness.device == CPU_DEVICE else files


def measure_with_fallback(
    harness: Harness, model_id: str, model_file: ModelFile, spec: CalibrationSpec, canary: np.ndarray
) -> PrecisionResult:
    errors: list[Exception] = []
    for disable_all in (False, True):
        try:
            return measure_precision(harness, model_id, model_file, spec, canary, disable_all)
        except Exception as exc:  # noqa: BLE001 -- el arnes anota el error del grafo y sigue con el resto
            if is_device_removed_error(exc):
                raise
            errors.append(exc)
    return PrecisionResult(model_id, model_file.precision, model_file, error=_one_line(errors[0]))


def measure_precision(
    harness: Harness, model_id: str, model_file: ModelFile, spec: CalibrationSpec, canary: np.ndarray, disable_all: bool
) -> PrecisionResult:
    before = harness.vram_usage_mb()
    session = harness.open_session(model_file.path, harness.device, disable_all)
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
        cpu_nodes=harness.cpu_nodes(model_file.path, disable_all),
        calibration=calibration,
        median_ms=median,
        non_finite=has_non_finite(output),
        vram_factor=vram_factor(before, after, model_file.size),
        ort_disable_all=disable_all,
        output=output,
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


# --- veredicto -----------------------------------------------------------------


def precision_issues(result: PrecisionResult, rule: CanaryRule, device: str, budget_ms: float) -> tuple[str, ...]:
    if result.error is not None:
        return (f"error: {result.error}",)
    checks = (
        provider_issue(result.providers, device),
        cpu_node_issue(result.cpu_nodes),
        tile_issue(result.calibration),
        median_issue(result.median_ms, budget_ms),
        "NaN/Inf en la salida" if result.non_finite else None,
        canary_issue("DML fp32", result.vs_fp32, rule),
        canary_issue("CPU fp32", result.vs_cpu, rule),
    )
    return tuple(issue for issue in checks if issue is not None)


def provider_issue(providers: tuple[str, ...], device: str) -> str | None:
    if device == CPU_DEVICE or providers[:1] == (DML_PROVIDER,):
        return None
    return f"la sesión corre en {providers[0] if providers else 'un provider desconocido'}"


def cpu_node_issue(cpu_nodes: tuple[str, ...]) -> str | None:
    if not cpu_nodes:
        return None
    return f"{len(cpu_nodes)} nodos en CPU ({', '.join(sorted(set(cpu_nodes)))})"


def tile_issue(calibration: TileCalibration | None) -> str | None:
    if calibration is None or not calibration.runs_on_cpu:
        return None
    return f"corre en CPU ({calibration.cpu_fallback_reason})"


def median_issue(median_ms: float | None, budget_ms: float) -> str | None:
    target_ms = budget_ms / 2
    if median_ms is None or median_ms < target_ms:
        return None
    return f"mediana {median_ms:.0f} ms > {target_ms:.0f} ms"


def canary_issue(label: str, score: float | None, rule: CanaryRule) -> str | None:
    if score is None or rule.passes(score):
        return None
    return f"canario vs {label} {rule.describe(score)} (umbral {rule.describe(rule.threshold)})"


def verdict(result: PrecisionResult) -> str:
    if not result.issues:
        return "ok"
    prefix = "fp16 descartado" if result.precision == "fp16" else "falla"
    return f"{prefix}: {'; '.join(result.issues)}"


def suggest_specs(results: Sequence[PrecisionResult]) -> list[SpecSuggestion]:
    return [_suggest_spec(model_id, rows) for model_id, rows in _group_by_model(results).items()]


def _suggest_spec(model_id: str, rows: list[PrecisionResult]) -> SpecSuggestion:
    passed = [row for row in rows if not row.issues]
    factors = [row.vram_factor for row in rows if row.vram_factor is not None]
    return SpecSuggestion(
        model_id=model_id,
        tile_by_precision={row.precision: row.calibration.tile for row in passed},
        fp16_filename=next((row.file.path.name for row in passed if row.precision == "fp16"), None),
        ort_disable_all=any(row.ort_disable_all for row in rows),
        vram_factor=max(factors) if factors else None,
    )


def _group_by_model(results: Sequence[PrecisionResult]) -> dict[str, list[PrecisionResult]]:
    groups: dict[str, list[PrecisionResult]] = {}
    for result in results:
        groups.setdefault(result.model_id, []).append(result)
    return groups


# --- informe -------------------------------------------------------------------


def _number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return NOT_AVAILABLE
    if math.isinf(value):
        return "∞"
    return f"{value:.{digits}f}"


def _score_cell(score: float | None, rule: CanaryRule) -> str:
    return NOT_AVAILABLE if score is None else rule.describe(score)


def _cpu_nodes_cell(cpu_nodes: tuple[str, ...]) -> str:
    return str(len(cpu_nodes)) if not cpu_nodes else f"{len(cpu_nodes)}: {', '.join(sorted(set(cpu_nodes)))}"


def result_row(result: PrecisionResult) -> tuple[str, ...]:
    rule = canary_rule_for(result.model_id)
    calibration = result.calibration
    return (
        result.model_id,
        result.precision,
        result.file.path.name,
        result.file.sha256,
        ", ".join(result.providers) or NOT_AVAILABLE,
        _cpu_nodes_cell(result.cpu_nodes),
        _tile_cell(calibration),
        _number(calibration.ms_per_mpx if calibration else None, 0),
        _number(result.median_ms, 0),
        _score_cell(result.vs_fp32, rule),
        _score_cell(result.vs_cpu, rule),
        "sí" if result.non_finite else "no",
        _number(result.vram_factor, 2),
        "sí" if result.ort_disable_all else "no",
        verdict(result),
    )


def _tile_cell(calibration: TileCalibration | None) -> str:
    if calibration is None:
        return NOT_AVAILABLE
    return "CPU" if calibration.tile is None else str(calibration.tile)


def spec_row(suggestion: SpecSuggestion) -> tuple[str, ...]:
    return (
        suggestion.model_id,
        repr(dict(suggestion.tile_by_precision)),
        repr(suggestion.fp16_filename),
        repr(suggestion.ort_disable_all),
        _number(suggestion.vram_factor, 2),
    )


def markdown_table(columns: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [_markdown_row(columns), "|" + "|".join("---" for _ in columns) + "|"]
    lines += [_markdown_row(row) for row in rows]
    return "\n".join(lines)


def _markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


def render_report(context: ReportContext, results: Sequence[PrecisionResult], skipped: Sequence[Skipped]) -> str:
    sections = [
        f"# Validación DML de los modelos de restauración — {context.date}",
        _report_preamble(context),
        "## Resultados",
        markdown_table(RESULT_COLUMNS, (result_row(result) for result in results)),
        "## Valores para RestoreModelSpec",
        markdown_table(SPEC_COLUMNS, (spec_row(item) for item in suggest_specs(results))),
    ]
    if skipped:
        sections += ["## Sin validar", markdown_table(SKIPPED_COLUMNS, ((s.model_id, s.reason) for s in skipped))]
    return "\n\n".join(sections) + "\n"


def _report_preamble(context: ReportContext) -> str:
    return (
        f"- Device: `{context.device}` · onnxruntime {context.ort_version} · presupuesto por llamada "
        f"{context.budget_ms:.0f} ms (la mediana tiene que quedar < {context.budget_ms / 2:.0f} ms)\n"
        f"- Modelos: `{context.source}`\n"
        "- Sesiones con `create_session(..., prefer_native=False)`; nodos en CPU por perfilado de ORT; calibración "
        "TDR por precisión y canario contra DML fp32 y CPU fp32 (spec §3.5 y §6.1).\n"
        "- Generado por `scripts/spike_restore_dml.py`."
    )


def report_rows(text: str) -> dict[RowKey, dict[str, str]]:
    lines = text.splitlines()
    header = _markdown_row(RESULT_COLUMNS)
    start = lines.index(header) + 2
    rows: dict[RowKey, dict[str, str]] = {}
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = dict(zip(RESULT_COLUMNS, _split_row(line), strict=True))
        rows[(cells["Modelo"], cells["Precisión"])] = cells
    return rows


def _split_row(line: str) -> list[str]:
    placeholder = "\x00"
    cells = line.replace("\\|", placeholder).strip().strip("|").split("|")
    return [cell.strip().replace(placeholder, "|") for cell in cells]


def report_differences(expected_text: str, actual_text: str) -> list[str]:
    expected = report_rows(expected_text)
    actual = report_rows(actual_text)
    problems: list[str] = []
    for key, row in expected.items():
        problems += _row_differences(key, row, actual.get(key))
    extra = sorted(actual.keys() - expected.keys())
    problems += [f"{model} {precision}: no estaba en el informe esperado" for model, precision in extra]
    return problems


def _row_differences(key: RowKey, expected: dict[str, str], actual: dict[str, str] | None) -> list[str]:
    label = f"{key[0]} {key[1]}"
    if actual is None:
        return [f"{label}: falta en la corrida nueva"]
    problems = []
    if actual["SHA-256"] != expected["SHA-256"]:
        problems.append(f"{label}: SHA-256 {expected['SHA-256']} -> {actual['SHA-256']}")
    if (actual["Veredicto"] == "ok") != (expected["Veredicto"] == "ok"):
        problems.append(f"{label}: veredicto {expected['Veredicto']} -> {actual['Veredicto']}")
    return problems


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


# --- ORT real y CLI -----------------------------------------------------------


def restore_session_options(device: str, disable_all: bool, profile: bool = False) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_profiling = profile
    if device != CPU_DEVICE:
        options.intra_op_num_threads = 1
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def open_restore_session(path: Path, device: str, settings: Any, disable_all: bool, profile: bool = False) -> Any:
    return ep_registry.create_session(
        str(path),
        device,
        settings,
        sess_options_factory=lambda: restore_session_options(device, disable_all, profile),
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
        open_session=lambda path, on_device, disable_all: open_restore_session(path, on_device, settings, disable_all),
        cpu_nodes=lambda path, disable_all: profile_cpu_nodes(
            lambda: open_restore_session(path, device, settings, disable_all, profile=True), device
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
    results, skipped = validate_targets(build_harness(args, settings, budget_ms), select_targets(targets, args.models))
    today = dt.date.today().isoformat()
    report = render_report(ReportContext(today, args.device, source, _ort_version(), budget_ms), results, skipped)
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
