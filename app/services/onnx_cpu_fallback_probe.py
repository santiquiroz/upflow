from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.services.backend_registry import BUILTIN_ONNX_MODELS
from app.services.devices_service import DevicesService

# ---------------------------------------------------------------------------
# Fase 0.1: detects ONNX Runtime ops that silently fall back to the CPU EP on
# a DirectML (or other GPU) session. `disable_cpu_ep_fallback` would make
# session CREATION raise on the first such op instead of enumerating them, so
# this uses ONNX Runtime's own profiling API instead: run once with a
# synthetic input, read the profiling JSON `end_profiling()` returns, and
# read the `provider` field ORT stamps on every executed node -- an official,
# structured mechanism (not string-scraping the general log stream).
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, Any] = {
    "tensor(uint8)": np.uint8,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}
_DEFAULT_DYNAMIC_DIM = 64


@dataclass(frozen=True, slots=True)
class CpuFallbackReport:
    model_id: str
    device_id: str
    hot_ops: tuple[str, ...]
    clean: bool


def _resolve_dim(dim: object) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return _DEFAULT_DYNAMIC_DIM


def build_synthetic_inputs(input_nodes: list[Any]) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for node in input_nodes:
        shape = [_resolve_dim(dim) for dim in node.shape]
        dtype = _DTYPE_MAP.get(node.type, np.float32)
        feeds[node.name] = np.zeros(shape, dtype=dtype)
    return feeds


def hot_cpu_ops(profile_events: list[dict], device_provider: str) -> list[str]:
    hot: list[str] = []
    for event in profile_events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args") or {}
        provider = args.get("provider")
        if provider is None or provider == device_provider:
            continue
        hot.append(args.get("op_name", event.get("name", "unknown")))
    return hot


def probe_cpu_fallback(
    model_path: str, model_id: str, device_id: str, device_ep: str, providers: list[Any]
) -> CpuFallbackReport:
    # model_id is the catalog id (e.g. "realesrgan-x4plus") stored on the
    # report -- it must be passed in explicitly rather than derived from
    # model_path's filename stem, because every builtin model's vendored
    # filename carries a "-uint8" suffix the catalog id does not
    # (BUILTIN_ONNX_MODELS["realesrgan-x4plus"].filename ==
    # "realesrgan-x4plus-uint8.onnx"); deriving it from the path would make
    # this report's model_id diverge from the catalog's model_id for every
    # builtin model except apollo. device_id is likewise the user-facing id
    # ("cpu"/"dml:0") stored on the report for cache/catalog keying; device_ep
    # is the raw ORT provider string ("CPUExecutionProvider"/
    # "DmlExecutionProvider") that profiling stamps on each node -- the two
    # are NOT interchangeable for GPU devices, see
    # OnnxCpuFallbackProbe._resolve for where device_ep actually comes from.
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session = ort.InferenceSession(model_path, session_options, providers=providers)
    feeds = build_synthetic_inputs(session.get_inputs())
    output_names = [output.name for output in session.get_outputs()]
    session.run(output_names, feeds)
    profile_path = session.end_profiling()
    try:
        events = json.loads(Path(profile_path).read_text())
    finally:
        Path(profile_path).unlink(missing_ok=True)
    hot_ops = tuple(hot_cpu_ops(events, device_ep))
    return CpuFallbackReport(model_id=model_id, device_id=device_id, hot_ops=hot_ops, clean=not hot_ops)


class OnnxCpuFallbackProbe:
    """Diagnostic-only: probe_cpu_fallback runs a real ORT session, so this
    is never called from a job's hot path -- only manually from the
    Optimization Center diagnostics panel, one (model, device) pair at a
    time."""

    def __init__(self, settings: Settings, devices: DevicesService) -> None:
        self.settings = settings
        self.devices = devices
        self._cache: dict[tuple[str, str], CpuFallbackReport] = {}
        self._lock = asyncio.Lock()

    def catalog(self) -> list[tuple[str, str]]:
        # Builtin Real-ESRGAN ONNX exports + Apollo -- both have a single,
        # fixed-role graph with a known input contract. AudioSR (multi-graph
        # DDIM loop) and GMFSS (4 graphs, ORT_DISABLE_ALL-gated) need their
        # own model-specific harness to probe meaningfully -- deferred, see
        # the plan's "Deferred" section.
        device_ids = [device["id"] for device in self.devices.list_devices()]
        pairs: list[tuple[str, str]] = []
        for model_id in BUILTIN_ONNX_MODELS:
            for device_id in device_ids:
                pairs.append((model_id, device_id))
        if self.settings.apollo_restore_model_path.exists():
            for device_id in device_ids:
                pairs.append(("apollo", device_id))
        return pairs

    def cached(self, model_id: str, device_id: str) -> CpuFallbackReport | None:
        return self._cache.get((model_id, device_id))

    async def scan(self, model_id: str, device_id: str) -> CpuFallbackReport:
        model_path, providers, device_ep = self._resolve(model_id, device_id)
        async with self._lock:
            report = await asyncio.to_thread(
                probe_cpu_fallback, model_path, model_id, device_id, device_ep, providers
            )
            self._cache[(model_id, device_id)] = report
            return report

    def _resolve(self, model_id: str, device_id: str) -> tuple[str, list[Any], str]:
        from app.services.engines.onnx_upscaler import _build_providers

        providers = _build_providers(device_id)
        # _build_providers returns a plain provider-name string for "cpu" but
        # a (name, options) tuple for "dml:N" -- device_ep must always be the
        # bare provider name string to compare against profiling's `provider`
        # field, so unwrap the tuple case.
        first = providers[0]
        device_ep = first[0] if isinstance(first, tuple) else first
        if model_id == "apollo":
            return str(self.settings.apollo_restore_model_path), providers, device_ep
        model = BUILTIN_ONNX_MODELS[model_id]
        return str(self.settings.builtin_onnx_path / model.filename), providers, device_ep
