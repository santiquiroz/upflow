from __future__ import annotations

import numpy as np

from app.services.onnx_cpu_fallback_probe import build_synthetic_inputs, hot_cpu_ops


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
