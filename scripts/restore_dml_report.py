"""Veredicto e informe markdown de spike_restore_dml.py: problemas por fila, valores sugeridos para
RestoreModelSpec, tablas y comparación contra un informe esperado (P1-GPU-1)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import restore_dml_probes as probes

from app.services import ep_registry
from app.services.engines.restore_canary import CanaryRule, canary_rule_for
from app.services.engines.tiled_restore_runner import TileCalibration

CPU_DEVICE = "cpu"
DML_PROVIDER = ep_registry.DML_PROVIDER
NOT_AVAILABLE = "n/d"
RESULT_COLUMNS = (
    "Modelo", "Precisión", "Archivo", "SHA-256", "Providers", "Nodos en CPU", "Tile", "ms/Mpx", "Mediana ms",
    "vs DML fp32", "vs CPU fp32", "NaN/Inf", "vram_factor", "Sesión", "Veredicto",
)
SPEC_COLUMNS = ("Modelo", "tile_by_precision", "fp16_filename", "ort_disable_all", "dml_graph_fusion", "vram_factor")
# Se prueban en orden: si ORT no abre el grafo en el device (o lo abre en CPU sin avisar), el siguiente.
MODE_DEFAULT = "por defecto"
MODE_DISABLE_ALL = "ORT_DISABLE_ALL"
MODE_NO_FUSION = "sin fusión DML"
SESSION_MODES = (MODE_DEFAULT, MODE_DISABLE_ALL, MODE_NO_FUSION)
DML_DISABLE_GRAPH_FUSION = "ep.dml.disable_graph_fusion"
SKIPPED_COLUMNS = ("Modelo", "Motivo")

RowKey = tuple[str, str]
# Las filas vienen de spike_restore_dml.py (PrecisionResult y Skipped); aca solo se leen sus campos.
PrecisionResult = Any
Skipped = Any


@dataclass(frozen=True, slots=True)
class SpecSuggestion:
    model_id: str
    tile_by_precision: Mapping[str, int]
    fp16_filename: str | None
    ort_disable_all: bool
    dml_graph_fusion: bool
    vram_factor: float | None


@dataclass(frozen=True, slots=True)
class ReportContext:
    date: str
    device: str
    source: str
    ort_version: str
    budget_ms: float


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
        io_binding_issue(result.io_binding, device),
    )
    frames = probes.frame_issues(result.frames, budget_ms)
    return tuple(issue for issue in checks if issue is not None) + frames


def io_binding_issue(io_binding: bool | None, device: str) -> str | None:
    if io_binding is None or io_binding or device == CPU_DEVICE:
        return None
    return "IOBinding cayó a run común"


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
        # Solo cuentan las precisiones que pasaron: el modo de una fila descartada no se usa nunca.
        ort_disable_all=any(row.session_mode == MODE_DISABLE_ALL for row in passed),
        dml_graph_fusion=not any(row.session_mode == MODE_NO_FUSION for row in passed),
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
        result.session_mode,
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
        repr(suggestion.dml_graph_fusion),
        _number(suggestion.vram_factor, 2),
    )


def markdown_table(columns: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [_markdown_row(columns), "|" + "|".join("---" for _ in columns) + "|"]
    lines += [_markdown_row(row) for row in rows]
    return "\n".join(lines)


def _markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


def render_report(
    context: ReportContext,
    results: Sequence[PrecisionResult],
    skipped: Sequence[Skipped],
    detectors: Sequence[probes.DetectorProbe] = (),
) -> str:
    sections = [
        f"# Validación DML de los modelos de restauración — {context.date}",
        _report_preamble(context),
        "## Resultados",
        markdown_table(RESULT_COLUMNS, (result_row(result) for result in results)),
        "## Valores para RestoreModelSpec",
        markdown_table(SPEC_COLUMNS, (spec_row(item) for item in suggest_specs(results))),
    ]
    sections += _frames_section(results)
    if skipped:
        sections += ["## Sin validar", markdown_table(SKIPPED_COLUMNS, ((s.model_id, s.reason) for s in skipped))]
    if detectors:
        sections += [
            "## Detectores en el device (informativo: en la app corren en CPU)",
            markdown_table(probes.DETECTOR_COLUMNS, (probes.detector_row(item) for item in detectors)),
        ]
    return "\n\n".join(sections) + "\n"


def _frames_section(results: Sequence[PrecisionResult]) -> list[str]:
    rows = [
        row
        for result in results
        for row in probes.frame_rows(result.model_id, result.precision, result.frames, bool(result.io_binding))
    ]
    if not rows:
        return []
    return [
        "## Cuadros enteros del grafo de video (carril IA de CCTV)",
        "Misma regla que `tdr_start_tile`: cuadro entero si la extrapolación desde el tile mínimo da "
        "≤ presupuesto/2; si no, el tile calibrado (el cuadro entero no se llama).",
        markdown_table(probes.FRAME_COLUMNS, rows),
    ]


def _report_preamble(context: ReportContext) -> str:
    return (
        f"- Device: `{context.device}` · onnxruntime {context.ort_version} · presupuesto por llamada "
        f"{context.budget_ms:.0f} ms (la mediana tiene que quedar < {context.budget_ms / 2:.0f} ms)\n"
        f"- Modelos: `{context.source}`\n"
        "- Sesiones con `create_session(..., prefer_native=False)`; si ORT no abre el grafo en el device o lo abre "
        "en CPU sin avisar, se reintenta con `ORT_DISABLE_ALL` y después sin la fusión de grafo de DML (columna Sesión).\n"
        "- Nodos en CPU por perfilado de ORT; calibración "
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
