from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import HfAuthError
from app.services.compat_strategy import strategy_for
from app.services.hf_client import HfFile
from app.services.model_preflight import (
    MB,
    measure_devices,
    measure_disk,
    measure_free_ram,
    preflight_upscaler,
)

DEVICES = [
    {"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"},
    {"id": "dml:0", "kind": "gpu", "name": "RX 7800 XT", "backend": "directml"},
]


class FakeDevices:
    def __init__(self, devices: list[dict]) -> None:
        self._devices = devices

    def list_devices(self) -> list[dict]:
        return self._devices


class FakeProbe:
    def __init__(self, free_mb: int | None) -> None:
        self._free_mb = free_mb

    def free_capacity_mb(self, _device_id: str) -> int | None:
        return self._free_mb


class FakeHf:
    def __init__(self, files: list[HfFile] | Exception) -> None:
        self._files = files

    async def repo_files(self, _repo_id: str) -> list[HfFile]:
        if isinstance(self._files, Exception):
            raise self._files
        return self._files


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


async def run(tmp_path: Path, files: list[HfFile] | Exception, probes: dict | None = None):
    return await preflight_upscaler(
        hf_client=FakeHf(files),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes=probes if probes is not None else {},
        repo_id="owner/model",
        strategy=strategy_for("image"),
    )


# ---------------------------------------------------------------------------
# Las mediciones compartidas
# ---------------------------------------------------------------------------


def test_disk_is_measured_from_the_first_ancestor_that_exists(tmp_path: Path):
    # En una instalacion nueva el directorio de modelos todavia no existe.
    target = tmp_path / "no" / "existe" / "todavia"
    capacity = measure_disk(target)

    assert capacity is not None
    assert capacity.target_path == str(target)
    assert capacity.free_bytes > 0


def test_devices_report_unknown_vram_when_there_is_no_probe():
    rows = measure_devices(FakeDevices(DEVICES), {})
    assert [row.free_vram_bytes for row in rows] == [None, None]


def test_devices_report_the_probed_vram():
    rows = measure_devices(FakeDevices(DEVICES), {"gpu": FakeProbe(7000), "cpu": FakeProbe(32000)})
    by_id = {row.id: row for row in rows}
    assert by_id["dml:0"].free_vram_bytes == 7000 * MB
    assert by_id["cpu"].free_vram_bytes == 32000 * MB


def test_free_ram_is_none_without_a_cpu_probe():
    assert measure_free_ram({}) is None
    assert measure_free_ram({"gpu": FakeProbe(7000)}) is None


def test_free_ram_comes_from_the_cpu_probe():
    assert measure_free_ram({"cpu": FakeProbe(16000)}) == 16000 * MB


# ---------------------------------------------------------------------------
# Pre-flight de upscalers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_onnx_repo_is_ready_with_its_download_size(tmp_path: Path):
    report = await run(tmp_path, [HfFile(path="model.onnx", size=67 * MB)])

    assert report.compat == "ready_onnx"
    assert report.degraded is False
    assert report.download_bytes == 67 * MB


@pytest.mark.asyncio
async def test_the_download_size_is_the_file_the_installer_would_pick(tmp_path: Path):
    # pick_weight_file prefiere .onnx sobre .pth, asi que el tamaño reportado es
    # el del .onnx aunque el .pth sea mas grande.
    report = await run(
        tmp_path,
        [HfFile(path="model.pth", size=200 * MB), HfFile(path="model.onnx", size=67 * MB)],
    )
    assert report.download_bytes == 67 * MB


@pytest.mark.asyncio
async def test_the_largest_file_wins_among_equals(tmp_path: Path):
    report = await run(
        tmp_path,
        [HfFile(path="fp16.onnx", size=34 * MB), HfFile(path="fp32.onnx", size=67 * MB)],
    )
    assert report.download_bytes == 67 * MB


@pytest.mark.asyncio
async def test_a_repo_without_weights_reports_no_download_size(tmp_path: Path):
    report = await run(tmp_path, [HfFile(path="README.md", size=1)])

    assert report.compat == "incompatible"
    assert report.download_bytes is None


@pytest.mark.asyncio
async def test_capacity_is_measured_even_for_an_incompatible_repo(tmp_path: Path):
    # Medir no depende del veredicto: el reporte sirve igual.
    report = await run(tmp_path, [HfFile(path="README.md", size=1)])

    assert report.disk is not None
    assert len(report.devices) == 2


@pytest.mark.asyncio
async def test_an_auth_error_is_a_gated_verdict_and_not_a_degraded_report(tmp_path: Path):
    # 401/403 es exactamente lo que hay que decirle al usuario, no una falla de
    # medicion.
    report = await run(tmp_path, HfAuthError("403 Forbidden"))

    assert report.compat == "gated"
    assert report.degraded is False
    assert report.compat_reason_key == "compat.gatedAuth"
    assert "403" in report.compat_reason_params["detail"]


@pytest.mark.asyncio
async def test_any_other_network_failure_degrades_without_propagating(tmp_path: Path):
    report = await run(tmp_path, RuntimeError("hub unreachable"))

    assert report.degraded is True
    assert report.compat is None
    assert report.compat_reason_key is None
    # Y aun asi mide lo que no depende de la red.
    assert report.disk is not None
    assert len(report.devices) == 2


@pytest.mark.asyncio
async def test_the_report_carries_the_probed_capacity(tmp_path: Path):
    report = await run(
        tmp_path,
        [HfFile(path="model.onnx", size=1)],
        probes={"gpu": FakeProbe(7000), "cpu": FakeProbe(32000)},
    )
    assert report.free_ram_bytes == 32000 * MB
    assert {row.id: row.free_vram_bytes for row in report.devices}["dml:0"] == 7000 * MB


@pytest.mark.asyncio
async def test_the_report_never_claims_a_vram_verdict(tmp_path: Path):
    # A proposito: el factor de vram_estimate asume activaciones que crecen con la
    # resolucion, y un upscaler que hace tiling acota el pico con el tile. Reusar
    # ese numero daria un aviso confiado y equivocado.
    report = await run(tmp_path, [HfFile(path="model.onnx", size=67 * MB)])

    assert not hasattr(report, "estimated_peak_bytes")
    assert not hasattr(report, "precisions")


@pytest.mark.asyncio
async def test_the_compat_reason_travels_as_a_key(tmp_path: Path):
    report = await run(tmp_path, [HfFile(path="model.pth", size=1)])
    assert report.compat_reason_key == "compat.upscaler.needsConversion"
