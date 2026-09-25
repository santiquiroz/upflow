from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.models import JobStatus, UpscaleJob
from app.services import capabilities
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import ALLOWED_RESTORE_FORMATS, JobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.photo_restore_chain import UnknownRestoreStep
from app.services.photo_restore_job import PhotoRestoreJobRunner, PreStage
from app.services.photo_restore_pipeline import RestoreTooLarge, StepCall, StepOutcome
from app.services.photo_restorer_registry import validate_step_ready
from app.services.restore_outputs import saved_bit_depth
from app.services.restore_provenance import UpscaleInfo
from app.services.scale_fit import engine_output_path

ONNX_MODEL = "fake-onnx-2x"
BUILTIN_MODEL = "realesrgan-x4plus"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def write_image(path: Path, size: tuple[int, int] = (48, 32), fmt: str = "PNG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    pixels = rng.integers(40, 200, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format=fmt)
    return path


def fake_resolve(capability_id: str, settings: Settings, registry: object | None, missing=()):
    return capabilities.ResolvedCapability(
        id=capability_id,
        domain="image",
        label_key=f"capability.{capability_id}",
        status="needs_setup" if missing else "available",
        provisioning="vendored_pack",
        job_kind="image",
        strategies=("model",),
        missing_packs=tuple(missing),
    )


def always_ready(settings: Settings, step_id: str, *, uses_model: bool = True) -> None:
    del settings, step_id, uses_model


def core_pack_missing(settings: Settings, step_id: str, *, uses_model: bool = True) -> None:
    def resolve(capability_id, settings, registry):
        missing = ("restore-core",) if capability_id == "image.restoreModels" else ()
        return fake_resolve(capability_id, settings, registry, missing)

    validate_step_ready(settings, step_id, uses_model=uses_model, resolve=resolve)


class FakeRestoreEngine:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.released_before_ncnn: list[str] = []

    def begin_phase(self, device: str) -> None:
        self.phases.append(device)

    def release_before_ncnn(self, device: str) -> bool:
        self.released_before_ncnn.append(device)
        return True


def brighten(image: np.ndarray, call: StepCall) -> StepOutcome:
    call.progress("restore_tone", 1, 2)
    call.progress("restore_tone", 2, 2)
    return StepOutcome(np.clip(image * np.float32(1.1), 0.0, 1.0))


def make_runner(settings: Settings, engine: FakeRestoreEngine | None = None, **overrides) -> PhotoRestoreJobRunner:
    kwargs = {"step_runners": {"tone": brighten}, "app_version": "9.9.9", "check_ready": always_ready}
    kwargs.update(overrides)
    return PhotoRestoreJobRunner(settings, engine or FakeRestoreEngine(), **kwargs)


class FakeDevices:
    def __init__(self, devices: list[dict] | None = None) -> None:
        self._devices = devices or [{"id": "cpu", "kind": "cpu"}, {"id": "dml:0", "kind": "gpu"}]

    def list_devices(self) -> list[dict]:
        return self._devices

    def validate(self, device_id: str) -> dict:
        for device in self._devices:
            if device["id"] == device_id:
                return device
        raise ValueError(f"Unknown device id: {device_id!r}")


class SpyRouter:
    def __init__(self) -> None:
        self.kinds: list[ModelKind] = []

    @asynccontextmanager
    async def acquire_auto(self, devices, kind):
        self.kinds.append(kind)
        yield "dml:0"


class NeverCalledEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        raise AssertionError("the SR engine must not run for this job")


class FakeSrEngine(UpscaleEngine):
    def __init__(self, settings: Settings, native_scale: int = 2) -> None:
        self.settings = settings
        self.native_scale = native_scale
        self.jobs: list[UpscaleJob] = []

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        self.jobs.append(job)
        with Image.open(job.source_path) as image:
            size = (image.width * self.native_scale, image.height * self.native_scale)
            upscaled = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        target = engine_output_path(self.settings, job)
        target.parent.mkdir(parents=True, exist_ok=True)
        upscaled.save(target, format="PNG")
        return target


def make_registry(settings: Settings) -> ModelRegistry:
    registry = ModelRegistry(settings)
    entry = ModelEntry(
        id=ONNX_MODEL,
        name="Fake ONNX 2x",
        kind=ModelKind.onnx,
        source="https://huggingface.co/example/fake",
        size_bytes=1_000,
        scale=2,
        arch="fake",
        file_path="onnx/fake.onnx",
        status=ModelStatus.installed,
    )
    registry._entries[ONNX_MODEL] = entry  # noqa: SLF001
    return registry


def make_manager(settings: Settings, *, runner=None, engine=None, onnx_engine=None, router=None, registry=None):
    semaphores = DeviceSemaphores(settings)
    return JobManager(
        settings,
        engine or NeverCalledEngine(),
        semaphores,
        onnx_engine=onnx_engine,
        registry=registry,
        devices=FakeDevices(),
        device_router=router,
        restore_runner=runner,
    )


async def create_restore_job(manager: JobManager, source: Path, **overrides) -> UpscaleJob:
    kwargs = dict(
        source_path=source,
        original_filename="Grandma 1952.png",
        model_name=BUILTIN_MODEL,
        scale=1,
        output_format="png",
        device="cpu",
        restore_steps=["tone"],
    )
    kwargs.update(overrides)
    return await manager.create_job(**kwargs)


async def run_to_end(manager: JobManager, job: UpscaleJob, timeout: float = 30.0) -> None:
    await manager.start()
    try:
        for _ in range(int(timeout / 0.02)):
            if job.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"job did not finish: {job.status}")
    finally:
        await manager.stop()


# ---------------------------------------------------------------------------
# create_job: validacion
# ---------------------------------------------------------------------------


def test_restore_steps_are_stored_in_catalog_order(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    job = asyncio.run(
        create_restore_job(manager, write_image(tmp_path / "in.png"), restore_steps=["tone", "descreen", "denoise"])
    )

    assert job.restore_steps == ["descreen", "denoise", "tone"]
    assert job.scale == 1
    assert job.model_id is None


def test_unknown_restore_step_is_refused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(UnknownRestoreStep, match="sharpen"):
        asyncio.run(create_restore_job(manager, write_image(tmp_path / "in.png"), restore_steps=["sharpen"]))

    assert manager.jobs == {}


def test_scale_one_without_restore_steps_is_refused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(ValueError, match="Scale must be one of"):
        asyncio.run(create_restore_job(manager, write_image(tmp_path / "in.png"), restore_steps=None))


def test_scale_one_with_restore_steps_is_accepted(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    job = asyncio.run(create_restore_job(manager, write_image(tmp_path / "in.png")))

    assert job.scale == 1 and job.native_scale == 1


def test_missing_pack_names_the_pack_without_a_script(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings, check_ready=core_pack_missing))

    with pytest.raises(ValueError, match="restauración de fotos") as raised:
        asyncio.run(create_restore_job(manager, write_image(tmp_path / "in.png"), restore_steps=["denoise"]))

    assert ".ps1" not in str(raised.value)


def test_classic_repair_does_not_need_the_core_pack(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings, check_ready=core_pack_missing))

    job = asyncio.run(
        create_restore_job(
            manager,
            write_image(tmp_path / "in.png"),
            restore_steps=["repair"],
            restore_options={"repair": {"engine": "classic"}},
        )
    )

    assert job.restore_steps == ["repair"]


def test_tiff_is_accepted_only_for_restore_jobs(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))
    tiff = write_image(tmp_path / "scan.tif", fmt="TIFF")

    job = asyncio.run(create_restore_job(manager, tiff))
    with pytest.raises(ValueError, match="Unsupported image format: TIFF"):
        asyncio.run(
            manager.create_job(
                source_path=tiff, original_filename="scan.tif", model_name=BUILTIN_MODEL, scale=2, output_format="png"
            )
        )

    assert "TIFF" in ALLOWED_RESTORE_FORMATS
    assert job.restore_steps == ["tone"]


def test_input_pixel_cap_is_a_too_large_error(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, RESTORE_MAX_INPUT_PIXELS=1000)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(RestoreTooLarge) as raised:
        asyncio.run(create_restore_job(manager, write_image(tmp_path / "in.png")))

    assert raised.value.code == "restore.error.tooLarge"


def test_output_pixel_cap_counts_the_requested_scale(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, RESTORE_MAX_OUTPUT_PIXELS=48 * 32 * 3)
    manager = make_manager(settings, runner=make_runner(settings))
    source = write_image(tmp_path / "in.png")

    asyncio.run(create_restore_job(manager, source))
    with pytest.raises(RestoreTooLarge, match="At 2"):
        asyncio.run(
            create_restore_job(manager, source, scale=2, restore_options={"upscale_mode": "classic"})
        )


def test_preview_crop_outside_the_photo_is_refused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(ValueError, match="outside"):
        asyncio.run(
            create_restore_job(
                manager, write_image(tmp_path / "in.png"), restore_options={"preview_crop": [40, 0, 16, 16]}
            )
        )


def test_restore_without_a_runner_is_refused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=None)

    with pytest.raises(ValueError, match="not configured"):
        asyncio.run(create_restore_job(manager, write_image(tmp_path / "in.png")))


def test_analysis_session_is_refused_until_sessions_exist(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(ValueError, match="sessions are not available"):
        asyncio.run(create_restore_job(manager, write_image(tmp_path / "in.png"), restore_session="abc"))


def test_restore_options_without_steps_are_refused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(ValueError, match="at least one restore step"):
        asyncio.run(
            create_restore_job(
                manager, write_image(tmp_path / "in.png"), scale=2, restore_steps=[], restore_options={"badge": False}
            )
        )


def test_upscale_mode_must_match_the_scale(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(ValueError, match="does not match scale"):
        asyncio.run(
            create_restore_job(manager, write_image(tmp_path / "in.png"), restore_options={"upscale_mode": "classic"})
        )


def test_invalid_photo_date_is_refused(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    with pytest.raises(ValueError, match="YYYY"):
        asyncio.run(
            create_restore_job(manager, write_image(tmp_path / "in.png"), restore_options={"photo_date": "circa 1950"})
        )


# ---------------------------------------------------------------------------
# _restore_kind con device=auto
# ---------------------------------------------------------------------------


def test_auto_dsp_only_restore_runs_on_cpu_without_the_gpu_permit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    router = SpyRouter()
    manager = make_manager(settings, runner=make_runner(settings), router=router)

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(manager, write_image(tmp_path / "in.png"), device="auto")
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    assert job.device == "cpu"
    assert router.kinds == []


def test_auto_restore_with_a_model_step_routes_as_onnx(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    router = SpyRouter()
    runner = make_runner(settings, step_runners={"denoise": lambda image, call: StepOutcome(image)})
    manager = make_manager(settings, runner=runner, router=router)

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(manager, write_image(tmp_path / "in.png"), device="auto", restore_steps=["denoise"])
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    assert router.kinds == [ModelKind.onnx]
    assert job.device == "dml:0"


def test_restore_kind_follows_the_sr_backend(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings), registry=make_registry(settings))
    base = UpscaleJob(source_path=tmp_path / "x.png", original_filename="x.png", model_name="m", scale=2, output_format="png")

    ncnn = UpscaleJob(**{**_fields(base), "model_id": BUILTIN_MODEL, "restore_steps": ["tone"]})
    onnx = UpscaleJob(**{**_fields(base), "model_id": ONNX_MODEL, "restore_steps": ["tone"]})
    classic = UpscaleJob(
        **{**_fields(base), "restore_steps": ["tone"], "restore_options": {"upscale_mode": "classic"}}
    )
    fast_repair = UpscaleJob(**{**_fields(base), "scale": 1, "restore_steps": ["repair"]})
    classic_repair = UpscaleJob(
        **{**_fields(base), "scale": 1, "restore_steps": ["repair"], "restore_options": {"repair": {"engine": "classic"}}}
    )

    assert manager._restore_kind(ncnn) == ModelKind.builtin_ncnn  # noqa: SLF001
    assert manager._restore_kind(onnx) == ModelKind.onnx  # noqa: SLF001
    assert manager._restore_kind(classic) is None  # noqa: SLF001
    assert manager._restore_kind(fast_repair) == ModelKind.onnx  # noqa: SLF001
    assert manager._restore_kind(classic_repair) is None  # noqa: SLF001


def _fields(job: UpscaleJob) -> dict:
    return {
        "source_path": job.source_path,
        "original_filename": job.original_filename,
        "model_name": job.model_name,
        "scale": job.scale,
        "output_format": job.output_format,
    }


# ---------------------------------------------------------------------------
# _run_engine: pre -> SR -> post y salidas
# ---------------------------------------------------------------------------


def test_restore_job_writes_outputs_sidecar_and_progress(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = FakeRestoreEngine()
    manager = make_manager(settings, runner=make_runner(settings, engine))

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(manager, write_image(tmp_path / "uploads" / "in.png"))
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    outputs = settings.outputs_path
    names = {path.name for path in outputs.iterdir()}
    assert names == {f"{job.id}{suffix}" for suffix in (".png", ".view.jpg", ".preview.jpg", ".beforeafter.jpg", ".restore.json")}
    assert job.output_path == outputs / f"{job.id}.png"
    sidecar = json.loads((outputs / f"{job.id}.restore.json").read_text(encoding="utf-8"))
    assert sidecar["upflow"]["version"] == "9.9.9"
    assert [step["id"] for step in sidecar["steps"]] == ["tone"]
    assert {entry["role"] for entry in sidecar["outputs"]} == {"restored", "view", "preview", "beforeafter"}
    restore = job.metadata["restore"]
    assert restore["downloadNames"]["restored"] == "Grandma 1952_restored.png"
    assert restore["viewFullResolution"] is True
    assert job.metadata["progress"] == pytest.approx(1.0)
    assert [stage["key"] for stage in job.metadata["stages"]] == ["restore_tone", "saving"]
    assert engine.phases == ["cpu", "cpu"]
    assert not (settings.video_work_path / job.id).exists()


def test_restore_progress_uses_the_step_stage_not_upscaling(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    seen: list[tuple[str, object, object]] = []

    def spy_tone(image: np.ndarray, call: StepCall) -> StepOutcome:
        call.progress("restore_tone", 1, 3)
        seen.append((job_ref[0].metadata["stage"], job_ref[0].metadata["framesDone"], job_ref[0].metadata["framesTotal"]))
        return StepOutcome(image)

    job_ref: list[UpscaleJob] = []
    manager = make_manager(settings, runner=make_runner(settings, step_runners={"tone": spy_tone}))

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(manager, write_image(tmp_path / "in.png"))
        job_ref.append(job)
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    assert seen == [("restore_tone", 1, 3)]


def test_ai_upscale_runs_on_a_png_intermediate_moved_to_video_work(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    sr = FakeSrEngine(settings)
    moved: list[Path] = []

    class RecordingRunner(PhotoRestoreJobRunner):
        def finish(self, job, stage, upscaled, upscale, cancel_event=None):
            moved.extend(sorted((settings.video_work_path / job.id).iterdir()))
            assert upscaled.shape[:2] == (64, 96)
            return super().finish(job, stage, upscaled, upscale, cancel_event)

    runner = RecordingRunner(settings, FakeRestoreEngine(), step_runners={"tone": brighten}, app_version="1", check_ready=always_ready)
    manager = make_manager(settings, runner=runner, onnx_engine=sr, registry=make_registry(settings))

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager, write_image(tmp_path / "in.jpg", fmt="JPEG"), scale=2, output_format="jpg", model_id=ONNX_MODEL
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    derived = sr.jobs[0]
    assert derived.output_format == "png"
    assert derived.source_path == settings.video_work_path / job.id / "pre.png"
    assert derived.id == job.id
    assert [path.name for path in moved] == ["pre.png", "sr.png"]
    assert not (settings.outputs_path / f"{job.id}.png").exists()
    with Image.open(job.output_path) as result:
        assert result.size == (96, 64) and result.format == "JPEG"
    sidecar = json.loads((settings.outputs_path / f"{job.id}.restore.json").read_text(encoding="utf-8"))
    assert sidecar["upscale"] == {
        "mode": "ai",
        "scale": 2.0,
        "model": ONNX_MODEL,
        "generative": True,
        "backend": "onnx",
        "precision": None,
    }
    assert "generativeUpscale" in sidecar["compositeReasons"]


def test_failed_post_phase_leaves_no_orphans_in_outputs(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    sr = FakeSrEngine(settings)

    def broken_colorize(image: np.ndarray, call: StepCall) -> StepOutcome:
        raise RuntimeError("colorize exploded")

    runner = make_runner(settings, step_runners={"tone": brighten, "colorize": broken_colorize})
    manager = make_manager(settings, runner=runner, onnx_engine=sr, registry=make_registry(settings))

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager,
            write_image(tmp_path / "in.png"),
            scale=2,
            output_format="jpg",
            model_id=ONNX_MODEL,
            restore_steps=["tone", "colorize"],
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.failed
    assert "colorize exploded" in (job.error or "")
    assert sr.jobs, "the SR engine should have run before the post phase"
    assert list(settings.outputs_path.iterdir()) == []
    assert not (settings.video_work_path / job.id).exists()


def test_classic_upscale_resizes_without_the_sr_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager, write_image(tmp_path / "in.png"), scale=3, restore_options={"upscale_mode": "classic"}
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    with Image.open(job.output_path) as result:
        assert result.size == (144, 96)
    assert job.metadata["restore"]["upscale"]["mode"] == "classic"
    assert "upscaling" in [stage["key"] for stage in job.metadata["stages"]]


def test_builtin_sr_releases_restore_sessions_before_ncnn(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = FakeRestoreEngine()
    sr = FakeSrEngine(settings, native_scale=4)
    manager = make_manager(settings, runner=make_runner(settings, engine), engine=sr)

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager, write_image(tmp_path / "in.png"), scale=2, device="dml:0", model_id=BUILTIN_MODEL
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    assert engine.released_before_ncnn == ["dml:0"]
    with Image.open(job.output_path) as result:
        assert result.size == (96, 64)


def test_preview_crop_produces_only_the_preview_artifact(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager, write_image(tmp_path / "in.png"), restore_options={"preview_crop": [4, 4, 16, 8]}
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    assert [path.name for path in settings.outputs_path.iterdir()] == [f"{job.id}.preview.jpg"]
    with Image.open(job.output_path) as preview:
        assert preview.size == (16, 8)
    assert job.metadata["restore"]["previewCrop"] == [4, 4, 16, 8]


# ---------------------------------------------------------------------------
# Cancelacion: el permiso del device se suelta solo cuando el hilo termino
# ---------------------------------------------------------------------------


def test_cancel_keeps_the_device_permit_until_the_worker_thread_finishes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    thread_done = threading.Event()

    class BlockingRunner(PhotoRestoreJobRunner):
        def run_pre(self, job, cancel_event=None) -> PreStage:
            entered.set()
            release.wait(timeout=10)
            thread_done.set()
            raise RuntimeError("stopped")

    runner = BlockingRunner(settings, FakeRestoreEngine(), step_runners={"tone": brighten}, check_ready=always_ready)
    manager = make_manager(settings, runner=runner)

    async def scenario() -> tuple[int, bool, UpscaleJob]:
        job = await create_restore_job(manager, write_image(tmp_path / "in.png"))
        await manager.start()
        try:
            while not entered.is_set():
                await asyncio.sleep(0.01)
            manager.cancel_job(job.id)
            for _ in range(20):
                await asyncio.sleep(0.01)
            held_while_running = manager.device_semaphores.in_flight("cpu")
            finished_early = thread_done.is_set()
            release.set()
            for _ in range(500):
                if manager.device_semaphores.in_flight("cpu") == 0:
                    break
                await asyncio.sleep(0.01)
            return held_while_running, finished_early, job
        finally:
            release.set()
            await manager.stop()

    held, finished_early, job = asyncio.run(scenario())

    assert held == 1
    assert finished_early is False
    assert thread_done.is_set()
    assert manager.device_semaphores.in_flight("cpu") == 0
    assert job.status == JobStatus.cancelled


# ---------------------------------------------------------------------------
# Salidas: sin color, insignia y profundidad de bits
# ---------------------------------------------------------------------------


def sepia_colorize(image: np.ndarray, call: StepCall) -> StepOutcome:
    tinted = np.clip(image * np.array([1.05, 0.95, 0.8], dtype=np.float32), 0.0, 1.0)
    return StepOutcome(tinted, details={"model": "ddcolor-tiny"})


def test_colorized_job_keeps_the_uncolored_output_and_is_composite(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    runner = make_runner(settings, step_runners={"tone": brighten, "colorize": sepia_colorize})
    manager = make_manager(settings, runner=runner)

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager, write_image(tmp_path / "in.png", size=(96, 64)), restore_steps=["tone", "colorize"]
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    assert (settings.outputs_path / f"{job.id}.uncolored.png").exists()
    restore = job.metadata["restore"]
    assert restore["downloadNames"]["restored"] == "Grandma 1952_colorized.png"
    assert "colorize" in restore["compositeReasons"]
    assert restore["badge"] is True
    assert "uncolored" in restore["artifacts"]
    assert [stage["key"] for stage in job.metadata["stages"]] == ["restore_tone", "restore_colorize", "saving"]


def test_badge_can_be_turned_off_for_the_main_output(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    runner = make_runner(settings, step_runners={"colorize": sepia_colorize})
    manager = make_manager(settings, runner=runner)

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager,
            write_image(tmp_path / "in.png", size=(96, 64)),
            restore_steps=["colorize"],
            restore_options={"badge": False},
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    assert job.metadata["restore"]["badge"] is False
    assert job.metadata["restore"]["digitalSourceType"].endswith("compositeWithTrainedAlgorithmicMedia")


def test_sixteen_bit_png_stays_sixteen_bit_without_ai_upscaling(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings, runner=make_runner(settings))
    source = tmp_path / "scan16.png"
    pixels = np.random.default_rng(1).integers(0, 65535, size=(32, 48, 3), dtype=np.uint16)
    cv2.imwrite(str(source), pixels)

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(manager, source)
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    saved = cv2.imread(str(job.output_path), cv2.IMREAD_UNCHANGED)
    assert saved.dtype == np.uint16


@pytest.mark.parametrize(
    "fmt,input_bits,mode,expected",
    [("png", 16, "none", 16), ("png", 16, "classic", 16), ("png", 16, "ai", 8), ("jpg", 16, "none", 8), ("webp", 8, "none", 8)],
)
def test_saved_bit_depth(fmt: str, input_bits: int, mode: str, expected: int) -> None:
    assert saved_bit_depth(fmt, input_bits, UpscaleInfo(mode=mode)) == expected


def test_default_runners_restore_a_dsp_only_job_end_to_end(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    runner = PhotoRestoreJobRunner(settings, FakeRestoreEngine(), app_version="1", check_ready=always_ready)
    manager = make_manager(settings, runner=runner)

    async def scenario() -> UpscaleJob:
        job = await create_restore_job(
            manager,
            write_image(tmp_path / "in.png", size=(96, 64)),
            restore_steps=["tone", "descreen"],
            restore_options={"tone": {"strength": 0.5}},
        )
        await run_to_end(manager, job)
        return job

    job = asyncio.run(scenario())

    assert job.status == JobStatus.completed, job.error
    steps = job.metadata["restore"]["steps"]
    assert [step["id"] for step in steps] == ["descreen", "tone"]
    assert steps[1]["params"] == {"strength": 0.5}
    assert all(step["strategy"] == "dsp" for step in steps)
