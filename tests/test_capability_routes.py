from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.capability_routes import get_capability_probe, router
from app.services.capability_probe import CapabilityProbe, Lever, LeverStatus
from app.config import Settings


class FakeCapabilityProbe:
    def __init__(self) -> None:
        self.rescan_called = False
        self.fix_called_with: str | None = None

    async def list_levers(self) -> list[Lever]:
        return [Lever("hags", "HAGS", LeverStatus.ok, "enabled", False)]

    async def rescan(self) -> list[Lever]:
        self.rescan_called = True
        return await self.list_levers()

    async def apply_fix(self, lever_id: str) -> Lever:
        self.fix_called_with = lever_id
        return Lever(lever_id, lever_id, LeverStatus.ok, "fixed", False)


def make_client(fake: FakeCapabilityProbe) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_capability_probe] = lambda: fake
    return TestClient(app)


def test_get_capabilities_returns_levers() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["levers"][0]["id"] == "hags"
    assert body["levers"][0]["fixable"] is False


def test_post_rescan_calls_rescan() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.post("/api/v1/capabilities/rescan")

    assert response.status_code == 200
    assert fake.rescan_called is True


def test_post_fix_calls_apply_fix_with_lever_id() -> None:
    fake = FakeCapabilityProbe()
    client = make_client(fake)

    response = client.post("/api/v1/capabilities/hags/fix")

    assert response.status_code == 200
    assert fake.fix_called_with == "hags"
    assert response.json()["lever"]["status"] == "ok"


from app.api.capability_routes import get_onnx_cpu_fallback_probe
from app.services.onnx_cpu_fallback_probe import CpuFallbackReport


class FakeOnnxCpuFallbackProbe:
    def catalog(self) -> list[tuple[str, str]]:
        return [("realesrgan-x4plus", "cpu")]

    def cached(self, model_id: str, device_id: str) -> CpuFallbackReport | None:
        return None

    async def scan(self, model_id: str, device_id: str) -> CpuFallbackReport:
        return CpuFallbackReport(model_id, device_id, ("Conv",), False)


class CachedOnnxCpuFallbackProbe:
    def catalog(self) -> list[tuple[str, str]]:
        return [("realesrgan-x4plus", "cpu")]

    def cached(self, model_id: str, device_id: str) -> CpuFallbackReport | None:
        return CpuFallbackReport(model_id, device_id, ("Conv",), False)

    async def scan(self, model_id: str, device_id: str) -> CpuFallbackReport:
        raise AssertionError("scan should not be called when reading cached diagnostics")


class FailingOnnxCpuFallbackProbe:
    def catalog(self) -> list[tuple[str, str]]:
        return []

    def cached(self, model_id: str, device_id: str) -> CpuFallbackReport | None:
        return None

    async def scan(self, model_id: str, device_id: str) -> CpuFallbackReport:
        raise KeyError(model_id)


def make_diagnostics_client(fake_probe, fake_onnx_probe) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_capability_probe] = lambda: fake_probe
    app.dependency_overrides[get_onnx_cpu_fallback_probe] = lambda: fake_onnx_probe
    return TestClient(app)


def test_get_onnx_diagnostics_lists_catalog_with_cached_results() -> None:
    client = make_diagnostics_client(FakeCapabilityProbe(), FakeOnnxCpuFallbackProbe())

    response = client.get("/api/v1/capabilities/onnx-diagnostics")

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert entries[0]["modelId"] == "realesrgan-x4plus"
    assert entries[0]["report"] is None


def test_get_onnx_diagnostics_shows_cached_report_when_present() -> None:
    client = make_diagnostics_client(FakeCapabilityProbe(), CachedOnnxCpuFallbackProbe())

    response = client.get("/api/v1/capabilities/onnx-diagnostics")

    assert response.status_code == 200
    entries = response.json()["entries"]
    assert entries[0]["report"] == {
        "modelId": "realesrgan-x4plus",
        "deviceId": "cpu",
        "hotOps": ["Conv"],
        "clean": False,
    }


def test_post_onnx_diagnostics_scan_runs_and_returns_report() -> None:
    client = make_diagnostics_client(FakeCapabilityProbe(), FakeOnnxCpuFallbackProbe())

    response = client.post("/api/v1/capabilities/onnx-diagnostics/realesrgan-x4plus/cpu/scan")

    assert response.status_code == 200
    report = response.json()["report"]
    assert report["hotOps"] == ["Conv"]
    assert report["clean"] is False


def test_post_onnx_diagnostics_scan_returns_400_on_unknown_model() -> None:
    client = make_diagnostics_client(FakeCapabilityProbe(), FailingOnnxCpuFallbackProbe())

    response = client.post("/api/v1/capabilities/onnx-diagnostics/unknown-model/cpu/scan")

    assert response.status_code == 400
