from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.routes import list_devices as list_devices_route
from app.config import Settings
from app.main import app
from app.services import devices_service
from app.services.devices_service import (
    CPU_DEVICE,
    DevicesService,
    OnnxRuntimeProbe,
)

# ---------------------------------------------------------------------------
# SP1 Task 1 - devices_service + API /devices + settings.
#
# Real enumeration (see app/services/devices_service.py):
#   - onnxruntime.get_available_providers() tells us whether the installed
#     onnxruntime build has "DmlExecutionProvider" compiled in at all. That
#     alone does NOT mean a DirectML-capable GPU is physically present.
#   - Actual GPU count/names come from real DXGI adapter enumeration
#     (ctypes call into dxgi.dll, no extra Python deps), filtering out
#     software adapters (e.g. "Microsoft Basic Render Driver").
#   - Both steps are exposed as monkeypatchable module-level seams
#     (`_probe_onnxruntime`, `_enumerate_gpu_adapter_names`) so unit tests
#     never depend on real onnxruntime or real GPU hardware being present.
# ---------------------------------------------------------------------------


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def fake_probe(providers: list[str]) -> OnnxRuntimeProbe:
    return OnnxRuntimeProbe(available_providers=providers)


def patch_no_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices_service, "_probe_onnxruntime", lambda: None)


def patch_onnxruntime_without_directml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        devices_service, "_probe_onnxruntime", lambda: fake_probe(["CPUExecutionProvider"])
    )


def patch_onnxruntime_with_directml(
    monkeypatch: pytest.MonkeyPatch, adapter_names: list[str]
) -> None:
    monkeypatch.setattr(
        devices_service,
        "_probe_onnxruntime",
        lambda: fake_probe(["DmlExecutionProvider", "CPUExecutionProvider"]),
    )
    monkeypatch.setattr(devices_service, "_enumerate_gpu_adapter_names", lambda: adapter_names)


# ---------------------------------------------------------------------------
# list_devices()
# ---------------------------------------------------------------------------


def test_list_devices_returns_only_cpu_when_onnxruntime_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_no_onnxruntime(monkeypatch)
    service = DevicesService(make_settings())

    devices = service.list_devices()

    assert devices == [CPU_DEVICE]


def test_list_devices_returns_only_cpu_when_directml_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_onnxruntime_without_directml(monkeypatch)
    service = DevicesService(make_settings())

    devices = service.list_devices()

    assert devices == [CPU_DEVICE]


def test_list_devices_lists_dml_devices_in_enumeration_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_onnxruntime_with_directml(
        monkeypatch, ["AMD Radeon RX 7800 XT", "AMD Radeon(TM) Graphics"]
    )
    service = DevicesService(make_settings())

    devices = service.list_devices()

    assert devices == [
        CPU_DEVICE,
        {"id": "dml:0", "kind": "gpu", "name": "AMD Radeon RX 7800 XT", "backend": "directml"},
        {"id": "dml:1", "kind": "gpu", "name": "AMD Radeon(TM) Graphics", "backend": "directml"},
    ]


def test_list_devices_falls_back_to_generic_name_when_adapter_name_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_onnxruntime_with_directml(monkeypatch, ["", "AMD Radeon RX 7800 XT"])
    service = DevicesService(make_settings())

    devices = service.list_devices()

    assert devices[1]["name"] == "GPU 0"
    assert devices[1]["id"] == "dml:0"
    assert devices[2]["name"] == "AMD Radeon RX 7800 XT"


def test_list_devices_returns_only_cpu_when_directml_supported_but_no_adapters_enumerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers e.g. onnxruntime-directml installed on a machine where DXGI
    # enumeration itself fails or reports zero hardware adapters.
    patch_onnxruntime_with_directml(monkeypatch, [])
    service = DevicesService(make_settings())

    devices = service.list_devices()

    assert devices == [CPU_DEVICE]


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_returns_matching_device(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_onnxruntime_with_directml(monkeypatch, ["AMD Radeon RX 7800 XT"])
    service = DevicesService(make_settings())

    device = service.validate("dml:0")

    assert device["id"] == "dml:0"
    assert device["name"] == "AMD Radeon RX 7800 XT"


def test_validate_raises_value_error_for_unknown_device(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_no_onnxruntime(monkeypatch)
    service = DevicesService(make_settings())

    with pytest.raises(ValueError, match="dml:0"):
        service.validate("dml:0")


def test_validate_accepts_cpu_even_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_no_onnxruntime(monkeypatch)
    service = DevicesService(make_settings())

    assert service.validate("cpu") == CPU_DEVICE


# ---------------------------------------------------------------------------
# resolve_default()
# ---------------------------------------------------------------------------


def test_resolve_default_returns_configured_gpu_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_onnxruntime_with_directml(monkeypatch, ["AMD Radeon RX 7800 XT"])
    service = DevicesService(make_settings(DEFAULT_DEVICE="dml:0"))

    default_device = service.resolve_default()

    assert default_device["id"] == "dml:0"


def test_resolve_default_falls_back_to_cpu_when_configured_gpu_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_no_onnxruntime(monkeypatch)
    service = DevicesService(make_settings(DEFAULT_DEVICE="dml:0"))

    default_device = service.resolve_default()

    assert default_device == CPU_DEVICE


def test_resolve_default_falls_back_to_cpu_when_no_gpu_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_onnxruntime_without_directml(monkeypatch)
    service = DevicesService(make_settings(DEFAULT_DEVICE="dml:0"))

    assert service.resolve_default() == CPU_DEVICE


def test_resolve_default_reuses_provided_snapshot_without_reenumerating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A single /devices request must not hit the hardware twice: the route
    # enumerates once and derives the default from that same snapshot
    # (avoids redundant COM calls and a TOCTOU list-vs-default mismatch).
    patch_onnxruntime_with_directml(monkeypatch, ["AMD Radeon RX 7800 XT"])
    service = DevicesService(make_settings(DEFAULT_DEVICE="dml:0"))
    snapshot = service.list_devices()

    def failing_probe() -> OnnxRuntimeProbe:
        raise AssertionError("resolve_default(snapshot) must not re-enumerate")

    monkeypatch.setattr(devices_service, "_probe_onnxruntime", failing_probe)

    default_device = service.resolve_default(snapshot)

    assert default_device["id"] == "dml:0"


# ---------------------------------------------------------------------------
# Settings: DEFAULT_DEVICE
# ---------------------------------------------------------------------------


def test_settings_default_device_defaults_to_dml0() -> None:
    settings = Settings(_env_file=None)

    assert settings.default_device == "dml:0"


def test_settings_default_device_override() -> None:
    settings = Settings(_env_file=None, DEFAULT_DEVICE="cpu")

    assert settings.default_device == "cpu"


# ---------------------------------------------------------------------------
# Real DXGI enumeration seam - never raises, always returns a list, on any
# platform/hardware. Exercises the actual ctypes/COM code path (unlike the
# tests above, which patch it out) without asserting specific hardware.
# ---------------------------------------------------------------------------


def test_enumerate_gpu_adapter_names_real_call_never_raises() -> None:
    result = devices_service._enumerate_gpu_adapter_names()

    assert isinstance(result, list)
    assert all(isinstance(name, str) for name in result)


# ---------------------------------------------------------------------------
# API: GET /api/v1/devices
# ---------------------------------------------------------------------------


async def test_devices_route_returns_devices_and_default_device_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_onnxruntime_with_directml(monkeypatch, ["AMD Radeon RX 7800 XT"])
    settings = make_settings(DEFAULT_DEVICE="dml:0")
    service = DevicesService(settings)

    response = await list_devices_route(devices=service)

    assert response.default_device_id == "dml:0"
    assert [item.id for item in response.devices] == ["cpu", "dml:0"]
    assert response.devices[1].name == "AMD Radeon RX 7800 XT"


def test_devices_endpoint_wired_through_app_state(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_onnxruntime_with_directml(monkeypatch, ["AMD Radeon RX 7800 XT"])

    with TestClient(app) as client:
        response = client.get("/api/v1/devices")

    assert response.status_code == 200
    data = response.json()
    assert data["defaultDeviceId"] == "dml:0"
    assert {device["id"] for device in data["devices"]} == {"cpu", "dml:0"}
    assert data["devices"][1]["backend"] == "directml"
