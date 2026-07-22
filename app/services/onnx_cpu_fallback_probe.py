from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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
