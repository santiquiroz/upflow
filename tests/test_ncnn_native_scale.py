from pathlib import Path

from PIL import Image
import pytest

from app.config import Settings
from app.models import UpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.realesrgan_ncnn import (
    RealEsrganNcnnEngine,
    ncnn_output_format,
    raise_on_ncnn_failure,
    vulkan_failure_line,
)
from app.services.job_manager import JobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.scale_fit import engine_output_path


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_image(path: Path, size: tuple[int, int] = (64, 40)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "green").save(path)
    return path


class Probe:
    def free_capacity_mb(self, device_id: str) -> int:
        return 12000


class FakeEngine:
    async def run(self, job: UpscaleJob) -> Path:
        with Image.open(job.source_path) as source:
            native_size = (source.width * job.native_scale, source.height * job.native_scale)
        output = engine_output_path(settings_for_engine, job)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", native_size, "purple").save(output)
        return output


settings_for_engine: Settings


def test_ncnn_helpers_normalize_formats_and_report_vulkan_failures() -> None:
    assert ncnn_output_format(Path("out.jpeg")) == "jpg"
    assert ncnn_output_format(Path("out.png")) == "png"
    assert ncnn_output_format(Path("out.webp")) == "webp"
    stderr = b"progress\nvkAllocateMemory failed -2\n"
    assert vulkan_failure_line(stderr) == "vkAllocateMemory failed -2"
    with pytest.raises(RuntimeError, match="Vulkan failure.*tile_size"):
        raise_on_ncnn_failure(0, stderr)
    with pytest.raises(RuntimeError, match="process failed"):
        raise_on_ncnn_failure(1, b"process failed")
    raise_on_ncnn_failure(0, b"0.00%\n50.00%\n")


def test_ncnn_command_uses_native_scale_and_tile_selection(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source = make_image(tmp_path / "source.png")
    job = UpscaleJob(
        source_path=source,
        original_filename="source.png",
        model_name="realesrgan-x4plus",
        scale=2,
        native_scale=4,
        output_format="png",
        device="dml:0",
        tile_size=64,
    )
    engine = RealEsrganNcnnEngine(settings, vram_probe=Probe())
    command = engine.build_command(job, tmp_path / "out.png", 64)

    assert command[command.index("-s") + 1] == "4"
    assert command[command.index("-t") + 1] == "64"
    assert command[command.index("-f") + 1] == "png"


def test_ncnn_zero_tile_uses_probe_capacity(tmp_path: Path) -> None:
    source = make_image(tmp_path / "source.png")
    job = UpscaleJob(
        source_path=source,
        original_filename="source.png",
        model_name="realesrgan-x4plus",
        scale=2,
        native_scale=4,
        output_format="png",
        device="dml:0",
        tile_size=0,
    )
    engine = RealEsrganNcnnEngine(make_settings(tmp_path), vram_probe=Probe())

    assert engine.tile_for(job) == 64


def make_manager(tmp_path: Path, *, registry: ModelRegistry | None = None) -> JobManager:
    global settings_for_engine
    settings_for_engine = make_settings(tmp_path)
    return JobManager(settings_for_engine, FakeEngine(), DeviceSemaphores(settings_for_engine), registry=registry)


def test_job_manager_resolves_builtin_native_scales(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    x4 = manager._resolve_builtin_model("realesrgan-x4plus", 2, "dml:0")
    anime = manager._resolve_builtin_model("realesr-animevideov3", 2, "dml:0")

    assert (x4.scale, x4.native_scale, x4.engine_model_name) == (2, 4, "realesrgan-x4plus")
    assert (anime.scale, anime.native_scale, anime.engine_model_name) == (2, 2, "realesr-animevideov3-x2")


def test_job_manager_resolves_onnx_scale_without_inventing_native_scale(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(
        ModelEntry(
            id="fake-4x",
            name="Fake",
            kind=ModelKind.onnx,
            source="test",
            size_bytes=1,
            scale=4,
            file_path="onnx/fake-4x.onnx",
            status=ModelStatus.installed,
        )
    )
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings), registry=registry)

    resolution = manager._resolve_onnx_model("fake-4x", 2)
    assert (resolution.scale, resolution.native_scale) == (2, 4)
    native_resolution = manager._resolve_onnx_model("fake-4x", 4)
    assert (native_resolution.scale, native_resolution.native_scale) == (4, 4)


async def test_job_manager_create_job_validates_tiles_and_records_native_scale(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    source = make_image(tmp_path / "input.png")

    job = await manager.create_job(
        source_path=source,
        original_filename="a.png",
        model_name="realesrgan-x4plus",
        scale=2,
        output_format="png",
        device="dml:0",
        tile_size=64,
        tile_overlap=8,
    )
    assert (job.native_scale, job.tile_size, job.tile_overlap) == (4, 64, 8)
    with pytest.raises(ValueError, match="at least 32"):
        await manager.create_job(
            source_path=source,
            original_filename="a.png",
            model_name="realesrgan-x4plus",
            scale=2,
            output_format="png",
            device="dml:0",
            tile_size=16,
        )


async def test_job_manager_run_engine_fits_native_output_to_requested_scale(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    source = make_image(tmp_path / "input.png", (20, 12))
    job = UpscaleJob(
        source_path=source,
        original_filename="a.png",
        model_name="realesrgan-x4plus",
        scale=2,
        native_scale=4,
        output_format="png",
        id="fit-job",
    )

    await manager._run_engine(job)

    assert job.output_path == manager.settings.outputs_path / "fit-job.png"
    assert not (manager.settings.outputs_path / "fit-job.native.png").exists()
    with Image.open(job.output_path) as output:
        assert output.size == (40, 24)
