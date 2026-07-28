from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.generation_preflight import preflight
from app.services.hf_client import HfFile

MB = 1024 * 1024


# Local a proposito en vez de importarlo de test_generation_installer: este
# modulo no necesita nada mas de ahi, y depender de ese archivo arrastraria
# todos sus imports de app.services.generation_installer sin motivo.
def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


class FakeHf:
    def __init__(self, files, index=None, fail=False):
        self._files = files
        self._index = index or {}
        self._fail = fail

    async def repo_files(self, repo_id):
        if self._fail:
            raise RuntimeError("HF caido")
        return self._files

    async def download(self, repo_id, filename, dest, progress_cb=None,
                       max_bytes=None, unlimited=False):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self._index), encoding="utf-8")
        return dest


class FakeDevices:
    def __init__(self, devices):
        self._devices = devices

    def list_devices(self):
        return self._devices


class FakeProbe:
    def __init__(self, by_id):
        self._by_id = by_id

    def free_capacity_mb(self, device_id):
        return self._by_id.get(device_id)


FILES = [
    HfFile(path="model_index.json", size=1024),
    HfFile(path="unet/diffusion_pytorch_model.safetensors", size=3278 * MB),
    HfFile(path="unet/diffusion_pytorch_model.fp16.safetensors", size=1639 * MB),
]
INDEX = {
    "_class_name": "StableDiffusionPipeline",
    "unet": ["diffusers", "UNet2DConditionModel"],
}
DEVICES = [
    {"id": "dml:0", "kind": "gpu", "name": "RX 7900 XTX", "backend": "directml"},
    {"id": "dml:1", "kind": "gpu", "name": "RX 6600", "backend": "directml"},
    {"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"},
]

ONNX_FILES = [
    HfFile(path="model_index.json", size=1024),
    HfFile(path="unet/model.onnx", size=100 * MB),
    HfFile(path="unet/diffusion_pytorch_model.safetensors", size=3278 * MB),
]


@pytest.mark.asyncio
async def test_report_has_one_row_per_enumerated_device(tmp_path: Path) -> None:
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={
            "gpu": FakeProbe({"dml:0": 23700, "dml:1": 7400}),
            "cpu": FakeProbe({"cpu": 16000}),
        },
        repo_id="owner/name",
    )
    assert [d.id for d in report.devices] == ["dml:0", "dml:1", "cpu"]
    assert report.devices[0].free_vram_bytes == 23700 * MB
    assert report.devices[1].free_vram_bytes == 7400 * MB


@pytest.mark.asyncio
async def test_report_prices_every_available_precision(tmp_path: Path) -> None:
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    by_precision = {p.precision: p for p in report.precisions}
    assert set(by_precision) == {"fp16", "fp32"}
    assert by_precision["fp16"].download_bytes < by_precision["fp32"].download_bytes
    assert by_precision["fp16"].estimated_peak_bytes > by_precision["fp16"].download_bytes


@pytest.mark.asyncio
async def test_ready_onnx_repo_offers_no_precision_choice(tmp_path: Path) -> None:
    # Sin paso de export no hay dtype que elegir (alcance de B en el spec).
    report = await preflight(
        hf_client=FakeHf(ONNX_FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert report.compat == "ready_onnx"
    assert report.precisions == []


@pytest.mark.asyncio
async def test_unmeasurable_probe_yields_null_not_zero(tmp_path: Path) -> None:
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={"gpu": FakeProbe({})},
        repo_id="owner/name",
    )
    assert report.devices[0].free_vram_bytes is None


@pytest.mark.asyncio
async def test_device_without_a_registered_probe_yields_null(tmp_path: Path) -> None:
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert all(d.free_vram_bytes is None for d in report.devices)


@pytest.mark.asyncio
async def test_hf_failure_degrades_instead_of_raising(tmp_path: Path) -> None:
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX, fail=True),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert report.degraded is True
    assert report.precisions == []
    assert report.compat is None
    # Los dispositivos siguen informandose: no dependen de HF.
    assert [d.id for d in report.devices] == ["dml:0", "dml:1", "cpu"]


@pytest.mark.asyncio
async def test_disk_free_bytes_reported(tmp_path: Path) -> None:
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert report.disk is not None
    assert report.disk.free_bytes > 0


@pytest.mark.asyncio
async def test_reference_resolution_is_echoed_and_scales_the_estimate(tmp_path: Path) -> None:
    at_512 = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
        width=512,
        height=512,
    )
    at_1024 = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
        width=1024,
        height=1024,
    )
    assert at_512.reference_width == 512
    assert at_1024.reference_height == 1024
    assert (
        at_1024.precisions[0].estimated_peak_bytes
        > at_512.precisions[0].estimated_peak_bytes
    )


# ---------------------------------------------------------------------------
# Casos que el smoke real contra Hugging Face (2026-07-28) expuso: los dos
# repos mas interesantes colapsaban en degraded=True, tapando un veredicto que
# el sistema SI conocia.
# ---------------------------------------------------------------------------

NO_INDEX_FILES = [
    HfFile(path="config.json", size=512),
    HfFile(path="weights.safetensors", size=900 * MB),
]


class AuthFailingHf:
    """repo_files levanta HfAuthError, como hace la HfClient real con 401/403.
    Caso real: black-forest-labs/FLUX.1-dev."""

    async def repo_files(self, repo_id):
        from app.exceptions import HfAuthError

        raise HfAuthError("El repo requiere aceptar su licencia en Hugging Face.")

    async def download(self, *args, **kwargs):  # pragma: no cover - no se llega
        raise AssertionError("no deberia descargar nada si repo_files fallo")


class IndexMissingHf:
    """repo_files anda pero el repo no tiene model_index.json, asi que bajarlo
    daria 404. Caso real: wikeeyang/Flux2-Klein-9B-True-V2."""

    def __init__(self, files):
        self._files = files

    async def repo_files(self, repo_id):
        return self._files

    async def download(self, repo_id, filename, dest, progress_cb=None,
                       max_bytes=None, unlimited=False):
        raise RuntimeError("404 Not Found")


@pytest.mark.asyncio
async def test_gated_repo_reports_gated_not_degraded(tmp_path: Path) -> None:
    # Un 401/403 es un VEREDICTO, no una falla de medicion: se sabe que el repo
    # es gated y eso es lo accionable ("configurá tu token"), no "no se pudo
    # evaluar".
    report = await preflight(
        hf_client=AuthFailingHf(),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="black-forest-labs/FLUX.1-dev",
    )
    assert report.compat == "gated"
    assert report.degraded is False
    assert report.compat_reason
    assert report.precisions == []


@pytest.mark.asyncio
async def test_repo_without_model_index_reports_incompatible_not_degraded(
    tmp_path: Path,
) -> None:
    # Se clasifica ANTES de pedir model_index.json, justamente porque el repo
    # incompatible es el que no lo tiene: pedirlo daria 404 y el reporte
    # degradado taparia el veredicto.
    report = await preflight(
        hf_client=IndexMissingHf(NO_INDEX_FILES),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="wikeeyang/Flux2-Klein-9B-True-V2",
    )
    assert report.compat == "incompatible"
    assert report.degraded is False
    assert report.precisions == []


@pytest.mark.asyncio
async def test_ready_onnx_repo_never_fetches_model_index(tmp_path: Path) -> None:
    # No necesita `declared` porque no ofrece eleccion de precision. Si lo
    # pidiera, el download del doble reventaria.
    report = await preflight(
        hf_client=IndexMissingHf(ONNX_FILES),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert report.compat == "ready_onnx"
    assert report.degraded is False


@pytest.mark.asyncio
async def test_index_read_failure_keeps_the_verdict_and_degrades_only_pricing(
    tmp_path: Path,
) -> None:
    # Repo con model_index en el listado pero cuya descarga falla: el veredicto
    # needs_conversion se conserva, solo se pierde el precio por precision.
    report = await preflight(
        hf_client=IndexMissingHf(FILES),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert report.compat == "needs_conversion"
    assert report.degraded is True
    assert report.precisions == []
