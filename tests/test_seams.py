"""Regresion de costuras de tiling en el reescalado de imagen (NCNN y ONNX).

Causa raiz (2026-09-02): el binario realesrgan-ncnn-vulkan no reescala a una
escala distinta de la del modelo. Con `-s 2` sobre realesrgan-x4plus arma cada
tile copiando el cuarto superior izquierdo del tile ampliado x4: la salida tiene
las dimensiones pedidas pero es un mosaico (PSNR 13 dB contra el x4 real). La
correccion corre el modelo a su escala nativa y reduce con Lanczos en Upflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import zoom

from app.config import Settings
from app.models import UpscaleJob
from app.services.devices_service import DevicesService
from app.services.engines.onnx_upscaler import OnnxUpscaler
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.scale_fit import fit_output_to_scale
from seam_detector import measure_seams

SEAM_THRESHOLD = 2.0
# Lo que el binario elige con `-t 0` en una GPU con > 1.9 GB de heap (medido, RX 7800 XT).
NCNN_AUTO_TILE = 200
ONNX_TILE = 64
ONNX_OVERLAP = 16
ASSETS = Path(__file__).parent / "assets" / "seams"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), **overrides)


def make_smooth_image(width: int, height: int) -> np.ndarray:
    # Gradientes suaves con manchas: el contenido donde una costura se nota.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    red = 90 + 80 * np.sin(xx / 97) + 40 * np.cos(yy / 61)
    green = 110 + 70 * np.cos((xx + yy) / 83)
    blue = 80 + 60 * np.sin(xx / 51) * np.cos(yy / 71)
    noise = np.random.default_rng(7).normal(0, 3, (height, width, 3))
    return np.clip(np.stack([red, green, blue], axis=-1) + noise, 0, 255).astype(np.uint8)


def write_smooth_source(tmp_path: Path, width: int, height: int) -> Path:
    source = tmp_path / "smooth.png"
    Image.fromarray(make_smooth_image(width, height), mode="RGB").save(source)
    return source


def make_job(
    source: Path, *, model: str, scale: int, native_scale: int, device: str, tile_size: int | None
) -> UpscaleJob:
    return UpscaleJob(
        source_path=source,
        original_filename=source.name,
        model_name=model,
        model_id=model,
        scale=scale,
        native_scale=native_scale,
        output_format="png",
        device=device,
        tile_size=tile_size,
    )


# ---------------------------------------------------------------- detector


def test_detector_flags_the_original_ncnn_scale2_output() -> None:
    report = measure_seams(ASSETS / "ncnn-x4plus-s2-tiled-before.jpg", NCNN_AUTO_TILE * 2)
    assert report.worst_ratio > 5, report


def test_detector_passes_the_native_scale_plus_lanczos_output() -> None:
    report = measure_seams(ASSETS / "ncnn-x4plus-s2-after.jpg", NCNN_AUTO_TILE * 2)
    assert report.worst_ratio < SEAM_THRESHOLD, report


def test_detector_sees_an_injected_grid_and_nothing_else() -> None:
    image = make_smooth_image(900, 600).astype(np.int16)
    clean = measure_seams(image.astype(np.uint8), 300)
    image[:, 300::300] += 12
    image[300::300, :] += 12
    seamed = measure_seams(np.clip(image, 0, 255).astype(np.uint8), 300)
    assert clean.worst_ratio < 1.5, clean
    assert seamed.worst_ratio > 5, seamed


# ---------------------------------------------------------------- ncnn (binario real)


@pytest.mark.parametrize("scale", [2, 4])
async def test_ncnn_x4plus_output_has_no_tile_grid(tmp_path: Path, scale: int) -> None:
    settings = make_settings(tmp_path)
    engine = RealEsrganNcnnEngine(settings)
    if not engine.available():
        pytest.skip("realesrgan-ncnn-vulkan pack not installed (vendor/realesrgan)")
    source = write_smooth_source(tmp_path, 640, 384)
    job = make_job(source, model="realesrgan-x4plus", scale=scale, native_scale=4, device="dml:0", tile_size=None)

    native = await engine.run(job)
    output = fit_output_to_scale(native, job, settings)

    with Image.open(output) as image:
        assert image.size == (640 * scale, 384 * scale)
    report = measure_seams(output, NCNN_AUTO_TILE * scale)
    assert report.worst_ratio < SEAM_THRESHOLD, report
    assert job.metadata["effective"]["nativeScale"] == 4
    assert job.metadata["effective"]["requestedScale"] == scale


# ---------------------------------------------------------------- onnx (sesion falsa)


class _IoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class Biased4xSession:
    """x4 bilineal con un sesgo que depende del contenido del tile: dos tiles
    vecinos reciben sesgos distintos, asi que sin mezcla en el solape la union
    seria un escalon visible."""

    def get_inputs(self) -> list[_IoInfo]:
        return [_IoInfo("input")]

    def get_outputs(self) -> list[_IoInfo]:
        return [_IoInfo("output")]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = input_feed["input"]
        bias = 0.04 if float(batch[0, 0, 0, 0]) > 0.45 else -0.04
        upscaled = zoom(batch, (1, 1, 4, 4), order=1)
        return [np.clip(upscaled + bias, 0.0, 1.0).astype(np.float32)]


def make_onnx_engine(tmp_path: Path) -> tuple[OnnxUpscaler, Settings]:
    settings = make_settings(tmp_path, ONNX_TILE_SIZE=ONNX_TILE)
    registry = ModelRegistry(settings)
    registry.register(
        ModelEntry(
            id="fake-4x",
            name="Fake 4x",
            kind=ModelKind.onnx,
            source="test",
            size_bytes=1,
            scale=4,
            file_path="onnx/fake-4x.onnx",
            status=ModelStatus.installed,
        )
    )
    engine = OnnxUpscaler(settings, registry, DevicesService(settings), GpuSessionCoordinator())
    engine._create_session = lambda model_id, device, entry: Biased4xSession()  # type: ignore[method-assign]
    return engine, settings


@pytest.mark.parametrize("scale", [2, 4])
async def test_onnx_tiled_output_has_no_tile_grid(tmp_path: Path, scale: int) -> None:
    engine, settings = make_onnx_engine(tmp_path)
    # 256x160 con tile 64 y solape 16: los tiles arrancan en multiplos de 48.
    source = write_smooth_source(tmp_path, 256, 160)
    job = make_job(source, model="fake-4x", scale=scale, native_scale=4, device="cpu", tile_size=ONNX_TILE)

    native = await engine.run(job)
    output = fit_output_to_scale(native, job, settings)

    with Image.open(output) as image:
        assert image.size == (256 * scale, 160 * scale)
    report = measure_seams(output, (ONNX_TILE - ONNX_OVERLAP) * scale)
    assert report.worst_ratio < SEAM_THRESHOLD, report
    assert job.metadata["effective"]["tileSize"] == ONNX_TILE
    assert job.metadata["effective"]["tileOverlap"] == ONNX_OVERLAP


async def test_onnx_without_overlap_shows_the_grid_the_detector_is_built_for(tmp_path: Path) -> None:
    # Control negativo: con solape 0 no hay mezcla y el sesgo por tile queda como escalon.
    engine, settings = make_onnx_engine(tmp_path)
    source = write_smooth_source(tmp_path, 256, 160)
    job = make_job(source, model="fake-4x", scale=4, native_scale=4, device="cpu", tile_size=ONNX_TILE)
    job.tile_overlap = 0

    native = await engine.run(job)

    report = measure_seams(native, ONNX_TILE * 4)
    assert report.worst_ratio > 3, report
