from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from app.config import Settings
from app.services.devices_service import CPU_DEVICE, DevicesService
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.onnx_cpu_fallback_probe import (
    CpuFallbackReport,
    OnnxCpuFallbackProbe,
    build_synthetic_inputs,
    hot_cpu_ops,
    probe_cpu_fallback,
)


class _FakeInputNode:
    def __init__(self, name: str, shape: list, type_: str) -> None:
        self.name = name
        self.shape = shape
        self.type = type_


def test_build_synthetic_inputs_uses_fixed_dims_as_is() -> None:
    nodes = [_FakeInputNode("audio", [1, 1, 100], "tensor(float)")]

    feeds = build_synthetic_inputs(nodes)

    assert feeds["audio"].shape == (1, 1, 100)
    assert feeds["audio"].dtype == np.float32


def test_build_synthetic_inputs_replaces_dynamic_dims_with_default() -> None:
    nodes = [_FakeInputNode("frame", [1, "height", "width", 3], "tensor(uint8)")]

    feeds = build_synthetic_inputs(nodes)

    assert feeds["frame"].shape == (1, 64, 64, 3)
    assert feeds["frame"].dtype == np.uint8


def test_build_synthetic_inputs_handles_multiple_inputs() -> None:
    nodes = [
        _FakeInputNode("a", [1, 8], "tensor(float16)"),
        _FakeInputNode("b", [1, 2], "tensor(int64)"),
    ]

    feeds = build_synthetic_inputs(nodes)

    assert set(feeds) == {"a", "b"}
    assert feeds["a"].dtype == np.float16
    assert feeds["b"].dtype == np.int64


def test_hot_cpu_ops_filters_nodes_on_other_provider() -> None:
    events = [
        {"cat": "Node", "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"}},
        {"cat": "Node", "args": {"op_name": "Relu", "provider": "DmlExecutionProvider"}},
        {"cat": "Session", "args": {}},
    ]

    hot = hot_cpu_ops(events, device_provider="DmlExecutionProvider")

    assert hot == ["Conv"]


def test_hot_cpu_ops_returns_empty_when_all_on_device() -> None:
    events = [{"cat": "Node", "args": {"op_name": "Relu", "provider": "DmlExecutionProvider"}}]

    hot = hot_cpu_ops(events, device_provider="DmlExecutionProvider")

    assert hot == []


def test_hot_cpu_ops_ignores_events_without_provider_arg() -> None:
    events = [{"cat": "Node", "args": {"op_name": "Reshape"}}]

    hot = hot_cpu_ops(events, device_provider="DmlExecutionProvider")

    assert hot == []


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def _write_trivial_relu_model(path: Path) -> None:
    # A single-node graph (Relu) is enough to exercise real ORT profiling
    # end-to-end on the CPU EP without any GPU or vendored model file.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "trivial", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))


def test_probe_cpu_fallback_reports_clean_when_all_on_target_ep(tmp_path: Path) -> None:
    model_path = tmp_path / "trivial.onnx"
    _write_trivial_relu_model(model_path)

    # device_ep is the raw ORT provider string profiling stamps on each node
    # ("CPUExecutionProvider"/"DmlExecutionProvider") -- NOT the user-facing
    # device_id ("cpu"/"dml:0"), which is a separate argument (see the
    # OnnxCpuFallbackProbe._resolve fix below for where these come from).
    # model_id is likewise passed in explicitly ("trivial-model", deliberately
    # NOT equal to the "trivial" filename stem) to prove it is not derived
    # from model_path.
    report = probe_cpu_fallback(
        str(model_path), "trivial-model", "cpu", "CPUExecutionProvider", providers=["CPUExecutionProvider"]
    )

    assert report.clean is True
    assert report.hot_ops == ()
    assert report.model_id == "trivial-model"
    assert report.device_id == "cpu"


def test_probe_cpu_fallback_uses_passed_model_id_not_path_stem(tmp_path: Path) -> None:
    # Regression: every BUILTIN_ONNX_MODELS filename carries a "-uint8" suffix
    # the catalog model_id does not (e.g. "realesrgan-x4plus" ->
    # "realesrgan-x4plus-uint8.onnx"). Deriving model_id from
    # Path(model_path).stem would report "realesrgan-x4plus-uint8" here,
    # diverging from the catalog id the API's outer response field uses.
    model_path = tmp_path / "realesrgan-x4plus-uint8.onnx"
    _write_trivial_relu_model(model_path)

    report = probe_cpu_fallback(
        str(model_path), "realesrgan-x4plus", "cpu", "CPUExecutionProvider", providers=["CPUExecutionProvider"]
    )

    assert report.model_id == "realesrgan-x4plus"


def test_onnx_cpu_fallback_probe_catalog_includes_builtin_models(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    devices = DevicesService(settings)
    probe = OnnxCpuFallbackProbe(settings, devices, GpuSessionCoordinator())

    catalog = probe.catalog()

    # DevicesService always includes CPU_DEVICE (id "cpu") even with no GPU
    # present, so every builtin model is guaranteed to appear at least once.
    assert ("realesrgan-x4plus", CPU_DEVICE["id"]) in catalog


def test_onnx_cpu_fallback_probe_scan_caches_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(tmp_path)
    devices = DevicesService(settings)
    probe = OnnxCpuFallbackProbe(settings, devices, GpuSessionCoordinator())
    calls = {"n": 0}

    def fake_probe_cpu_fallback(
        model_path: str, model_id: str, device_id: str, device_ep: str, providers: list[str]
    ) -> CpuFallbackReport:
        calls["n"] += 1
        return CpuFallbackReport(model_id, device_id, (), True)

    import app.services.onnx_cpu_fallback_probe as mod

    monkeypatch.setattr(mod, "probe_cpu_fallback", fake_probe_cpu_fallback)

    # "realesrgan-x4plus" is a real BUILTIN_ONNX_MODELS key so _resolve()
    # succeeds before the (mocked) probe ever runs; scan() never touches the
    # model file itself since probe_cpu_fallback is monkeypatched out.
    first = asyncio.run(probe.scan("realesrgan-x4plus", "cpu"))
    assert probe.cached("realesrgan-x4plus", "cpu") == first
    assert calls["n"] == 1


def test_onnx_cpu_fallback_probe_scan_raises_when_model_file_missing(tmp_path: Path) -> None:
    # Regression: catalog() lists every BUILTIN_ONNX_MODELS entry
    # unconditionally, regardless of whether the vendored .onnx file has
    # actually been downloaded/placed -- so scan() must raise a well-known
    # exception type (caught by the route as HTTP 400) instead of letting
    # onnxruntime's own missing-file exception (not a FileNotFoundError or
    # RuntimeError) propagate as an unhandled 500.
    missing_onnx_dir = tmp_path / "missing-onnx-dir"
    settings = make_settings(tmp_path, BUILTIN_ONNX_DIR=str(missing_onnx_dir))
    devices = DevicesService(settings)
    probe = OnnxCpuFallbackProbe(settings, devices, GpuSessionCoordinator())

    with pytest.raises(RuntimeError, match="ONNX model file not found"):
        asyncio.run(probe.scan("realesrgan-x4plus", "cpu"))


def test_scan_calls_coordinator_acquire_before_creating_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: an unmanaged real ORT session created by this probe's
    # scan() with no involvement of GpuSessionCoordinator would contend with
    # a concurrent job's DirectML session on the same device -- see
    # GpuSessionCoordinator's docstring for the measured 10-16x slowdown
    # this project hit from unmanaged concurrent sessions.
    settings = make_settings(tmp_path)
    devices = DevicesService(settings)
    gpu_coordinator = GpuSessionCoordinator()
    probe = OnnxCpuFallbackProbe(settings, devices, gpu_coordinator)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))

    def fake_probe_cpu_fallback(
        model_path: str, model_id: str, device_id: str, device_ep: str, providers: list[str]
    ) -> CpuFallbackReport:
        return CpuFallbackReport(model_id, device_id, (), True)

    import app.services.onnx_cpu_fallback_probe as mod

    monkeypatch.setattr(mod, "probe_cpu_fallback", fake_probe_cpu_fallback)

    asyncio.run(probe.scan("realesrgan-x4plus", "cpu"))

    assert calls == [("cpu", probe)]


def test_onnx_cpu_fallback_probe_scan_raises_when_apollo_file_missing(tmp_path: Path) -> None:
    # Same guard, apollo branch: catalog() only includes ("apollo", device_id)
    # pairs when apollo_restore_model_path.exists() at catalog-build time, but
    # _resolve() doesn't recheck -- a file deleted between listing and
    # scanning must still surface as HTTP 400, not an unhandled 500.
    missing_apollo_model = tmp_path / "missing-apollo.onnx"
    settings = make_settings(tmp_path, APOLLO_RESTORE_MODEL=str(missing_apollo_model))
    devices = DevicesService(settings)
    probe = OnnxCpuFallbackProbe(settings, devices, GpuSessionCoordinator())

    with pytest.raises(RuntimeError, match="ONNX model file not found"):
        asyncio.run(probe.scan("apollo", "cpu"))
