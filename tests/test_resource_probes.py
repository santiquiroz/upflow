from __future__ import annotations

import sys

import pytest

import app.services.devices_service as devices_service
import app.services.resource_probes as resource_probes
from app.services.resource_probes import NullProbe


def test_null_probe_always_returns_none_for_any_device_id() -> None:
    probe = NullProbe()

    assert probe.free_capacity_mb("npu:0") is None
    assert probe.free_capacity_mb("dml:0") is None
    assert probe.free_capacity_mb("cpu") is None


def test_system_ram_probe_returns_available_mb_from_psutil_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource_probes, "_probe_psutil_available_mb", lambda: 4096)
    probe = resource_probes.SystemRamProbe()

    assert probe.free_capacity_mb("cpu") == 4096


def test_system_ram_probe_returns_none_when_seam_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource_probes, "_probe_psutil_available_mb", lambda: None)
    probe = resource_probes.SystemRamProbe()

    assert probe.free_capacity_mb("cpu") is None


def test_probe_psutil_available_mb_survives_missing_psutil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same technique test_devices.py already uses for onnxruntime: a None
    # entry in sys.modules makes `import psutil` raise ImportError even
    # though the real package is installed in this venv.
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert resource_probes._probe_psutil_available_mb() is None


def test_dxgi_vram_probe_delegates_to_devices_service_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        devices_service,
        "_query_adapter_vram_info_mb",
        lambda index: (2048, 512) if index == 1 else None,
    )
    probe = resource_probes.DxgiVramProbe()

    assert probe.free_capacity_mb("dml:1") == 2048
    assert probe.own_usage_mb("dml:1") == 512
    assert probe.free_capacity_mb("dml:0") is None
    assert probe.own_usage_mb("dml:0") is None


def test_dxgi_vram_probe_returns_none_for_non_dml_device_id() -> None:
    probe = resource_probes.DxgiVramProbe()

    assert probe.free_capacity_mb("cpu") is None
    assert probe.free_capacity_mb("npu:0") is None


def test_dxgi_vram_probe_returns_none_for_malformed_dml_device_id() -> None:
    probe = resource_probes.DxgiVramProbe()

    assert probe.free_capacity_mb("dml:not-a-number") is None
