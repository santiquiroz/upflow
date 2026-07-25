from __future__ import annotations

from app.services.resource_probes import NullProbe


def test_null_probe_always_returns_none_for_any_device_id() -> None:
    probe = NullProbe()

    assert probe.free_capacity_mb("npu:0") is None
    assert probe.free_capacity_mb("dml:0") is None
    assert probe.free_capacity_mb("cpu") is None
