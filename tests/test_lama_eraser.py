from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.config import Settings
from app.services.engines.lama_eraser import ERASER_LAMA_MODEL_ID, LamaEraser, _unknown_hole
from app.services.inpaint_mask import clean_protect_mask, grow_per_component, grow_small_hulls


def make_settings(tmp_path: Path, installed: bool = True) -> Settings:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "big-lama.pt"
    if installed:
        model.write_bytes(b"fake-torchscript")
    return Settings(
        _env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), LAMA_MODEL=str(model),
        MIGAN_MODEL=str(tmp_path / "absent-migan.onnx"),
    )


def photo(size: tuple[int, int] = (320, 240)) -> Image.Image:
    width, height = size
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[:, :, 0] = np.linspace(10, 100, width, dtype=np.uint8)[None, :]
    data[:, :, 1] = 80
    data[:, :, 2] = np.linspace(10, 100, height, dtype=np.uint8)[:, None]
    return Image.fromarray(data)


def mask_box(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    return Image.fromarray(mask)


class FakeModule:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.inference_modes: list[bool] = []

    def eval(self):
        return self

    def __call__(self, image, mask):
        self.calls.append((image.clone(), mask.clone()))
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return torch.full_like(image, 0.75)


def make_eraser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    eraser = LamaEraser(make_settings(tmp_path))
    module = FakeModule()
    monkeypatch.setattr(eraser, "_create_module", lambda: module)
    return eraser, module


def test_unknown_hole_without_protect_returns_hard_mask_without_dilation(monkeypatch):
    import cv2

    mask = np.zeros((400, 500), dtype=np.uint8)
    mask[180:200, 220:240] = 127
    protect = np.zeros(mask.shape, dtype=bool)

    def fail_dilation(*args, **kwargs):
        pytest.fail("An unprotected mask must not be dilated")

    monkeypatch.setattr(cv2, "dilate", fail_dilation)

    result = _unknown_hole(mask, protect)

    np.testing.assert_array_equal(result, (mask > 0).astype(np.uint8))
    assert result.dtype == np.uint8


def test_unknown_hole_leaves_distant_protect_outside_the_hole():
    mask = np.zeros((500, 600), dtype=np.uint8)
    mask[200:220, 200:220] = 255
    protect = np.zeros(mask.shape, dtype=bool)
    protect[200:220, 400:420] = True

    result = _unknown_hole(mask, protect)

    np.testing.assert_array_equal(result, (mask > 0).astype(np.uint8))


def test_unknown_hole_adds_adjacent_protect_within_the_ellipse():
    import cv2

    mask = np.zeros((500, 600), dtype=np.uint8)
    mask[220, 220] = 255
    protect = np.zeros(mask.shape, dtype=bool)
    protect[220, 221] = True
    protect[220, 370] = True
    protect[220, 371] = True
    protect[370, 370] = True
    hard = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (301, 301))
    expected = hard | (protect & (cv2.dilate(hard, kernel) > 0)).astype(np.uint8)

    result = _unknown_hole(mask, protect)

    np.testing.assert_array_equal(result, expected)
    assert result[220, 221] == result[220, 370] == 1
    assert result[220, 371] == result[370, 370] == 0


@pytest.mark.parametrize(
    "box,expected_shape",
    [((220, 200, 240, 215), (315, 320)), ((0, 0, 20, 15), (165, 170))],
)
def test_unknown_hole_dilates_only_the_expanded_clamped_window(monkeypatch, box, expected_shape):
    import cv2

    mask = np.zeros((500, 650), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    protect = np.ones(mask.shape, dtype=bool)
    hard = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (301, 301))
    dilate = cv2.dilate
    expected = dilate(hard, kernel)
    input_shapes = []

    def record_dilation(source, *args, **kwargs):
        input_shapes.append(source.shape)
        return dilate(source, *args, **kwargs)

    monkeypatch.setattr(cv2, "dilate", record_dilation)

    result = _unknown_hole(mask, protect)

    np.testing.assert_array_equal(result, expected)
    assert input_shapes == [expected_shape]


def test_unknown_hole_with_empty_hard_mask_adds_no_protected_pixels():
    mask = np.zeros((40, 50), dtype=np.uint8)
    protect = np.ones(mask.shape, dtype=bool)

    result = _unknown_hole(mask, protect)

    np.testing.assert_array_equal(result, mask)


@pytest.mark.asyncio
async def test_empty_mask_preserves_every_pixel_without_model_call(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo()

    result = await eraser.erase(base, Image.new("L", base.size), "dml:0")

    np.testing.assert_array_equal(result, base)
    assert module.calls == []


@pytest.mark.asyncio
async def test_only_the_grown_feathered_mask_can_change(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo()
    mask = mask_box(base.size, (130, 95, 180, 140))
    grown = grow_per_component(grow_small_hulls(np.asarray(mask)), None, None)

    result = await eraser.erase(base, mask, "cpu")

    assert result.size == base.size
    before, after = np.asarray(base), np.asarray(result)
    np.testing.assert_array_equal(after[grown == 0], before[grown == 0])
    assert np.all(after[110, 150] > before[110, 150] + 50)
    assert module.calls


@pytest.mark.asyncio
async def test_protected_pixels_are_preserved_even_inside_the_edit(tmp_path, monkeypatch):
    eraser, _ = make_eraser(tmp_path, monkeypatch)
    base = photo()
    mask = mask_box(base.size, (100, 80, 180, 160))
    protect = mask_box(base.size, (150, 50, 210, 180))

    result = await eraser.erase(base, mask, "cpu", protect_image=protect)

    protected = np.asarray(protect) > 0
    np.testing.assert_array_equal(np.asarray(result)[protected], np.asarray(base)[protected])
    assert np.asarray(result)[110, 120, 0] > 150


@pytest.mark.asyncio
async def test_protect_islands_are_written_but_the_expanded_subject_is_preserved(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo((400, 300))
    mask = mask_box(base.size, (130, 100, 280, 180))
    protect = np.asarray(mask_box(base.size, (220, 60, 330, 200))).copy()
    protect[135:138, 170:173] = 255
    cleaned = clean_protect_mask(protect > 127)

    result = await eraser.erase(base, mask, "cpu", protect_image=Image.fromarray(protect))

    before, after = np.asarray(base), np.asarray(result)
    np.testing.assert_array_equal(after[cleaned], before[cleaned])
    np.testing.assert_array_equal(after[120:160, 214:220], before[120:160, 214:220])
    assert np.all(after[135:138, 170:173, 0] > before[135:138, 170:173, 0] + 50)
    assert module.calls


@pytest.mark.asyncio
async def test_cleaned_protect_is_excluded_before_growth_and_grouping(tmp_path, monkeypatch):
    from app.services.engines import lama_eraser

    eraser, _ = make_eraser(tmp_path, monkeypatch)
    base = photo()
    mask = mask_box(base.size, (100, 80, 180, 160))
    protect = mask_box(base.size, (150, 50, 210, 180))
    cleaned = clean_protect_mask(np.asarray(protect) > 127)
    growth_inputs = []
    grouping_inputs = []
    group_masks = lama_eraser.edit_groups

    def record_growth(source):
        growth_inputs.append(source.copy())
        return grow_small_hulls(source)

    def record_grouping(source, *args):
        grouping_inputs.append(source.copy())
        return group_masks(source, *args)

    monkeypatch.setattr(lama_eraser, "grow_small_hulls", record_growth)
    monkeypatch.setattr(lama_eraser, "edit_groups", record_grouping)

    await eraser.erase(base, mask, "cpu", protect_image=protect)

    assert len(growth_inputs) == len(grouping_inputs) == 1
    assert not growth_inputs[0][cleaned].any()
    assert not grouping_inputs[0][cleaned].any()
    assert grouping_inputs[0][110, 120] == 255


@pytest.mark.asyncio
async def test_edit_entirely_inside_protection_preserves_image_without_inference(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo()
    mask = mask_box(base.size, (140, 90, 170, 120))
    protect = mask_box(base.size, (100, 50, 210, 180))

    result = await eraser.erase(base, mask, "cpu", protect_image=protect)

    np.testing.assert_array_equal(result, base)
    assert module.calls == []


@pytest.mark.asyncio
async def test_nearby_protected_subject_is_unknown_to_the_model(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    data = np.asarray(photo((600, 400))).copy()
    data[160:200, 310:335] = (255, 0, 0)
    base = Image.fromarray(data)
    mask = mask_box(base.size, (240, 150, 280, 210))
    protect = mask_box(base.size, (310, 160, 335, 200))

    result = await eraser.erase(base, mask, "cpu", protect_image=protect)

    assert len(module.calls) == 1
    image, model_mask = module.calls[0]
    subject = (image[0, 0] == 1) & (image[0, 1] == 0) & (image[0, 2] == 0)
    assert subject.any()
    assert torch.all(model_mask[0, 0][subject] == 1)
    np.testing.assert_array_equal(np.asarray(result)[160:200, 310:335], data[160:200, 310:335])


@pytest.mark.asyncio
async def test_large_crop_is_downscaled_before_cpu_inference(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo((2400, 1800))
    mask = mask_box(base.size, (300, 300, 1900, 1300))

    result = await eraser.erase(base, mask, "dml:0")

    assert result.size == base.size
    image, model_mask = module.calls[0]
    assert max(image.shape[-2:]) == 1600
    assert all(side % 8 == 0 for side in image.shape[-2:])
    assert image.shape[:2] == (1, 3)
    assert model_mask.shape == (1, 1, *image.shape[-2:])
    assert image.dtype == model_mask.dtype == torch.float32
    assert image.device.type == model_mask.device.type == "cpu"
    assert 0 <= image.min() <= image.max() <= 1
    assert set(torch.unique(model_mask).tolist()) == {0.0, 1.0}
    assert module.inference_modes == [True]


@pytest.mark.asyncio
async def test_small_crop_never_upscales_and_has_zero_mask_padding(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo((67, 45))
    mask = mask_box(base.size, (0, 0, 67, 45))

    result = await eraser.erase(base, mask, "cpu")

    image, model_mask = module.calls[0]
    assert image.shape == (1, 3, 48, 72)
    assert torch.all(model_mask[:, :, 45:, :] == 0)
    assert torch.all(model_mask[:, :, :, 67:] == 0)
    np.testing.assert_allclose(image[0, :, 45, 0], image[0, :, 43, 0])
    np.testing.assert_allclose(image[0, :, 0, 67], image[0, :, 0, 65])
    assert result.size == base.size


@pytest.mark.asyncio
async def test_module_loads_on_cpu_once_and_is_cached(tmp_path, monkeypatch):
    eraser = LamaEraser(make_settings(tmp_path))
    module = FakeModule()
    loads = []

    def load(path, map_location):
        loads.append((Path(path), map_location))
        return module

    monkeypatch.setattr(torch.jit, "load", load)
    base = photo()
    mask = mask_box(base.size, (100, 80, 140, 120))

    await eraser.erase(base, mask, "cpu")
    await eraser.erase(base, mask, "dml:0")

    assert loads == [(eraser.settings.lama_model_path, "cpu")]
    assert len(module.calls) == 2


@pytest.mark.asyncio
async def test_disjoint_marks_use_separate_context_groups(tmp_path, monkeypatch):
    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo((1200, 400))
    mask = np.asarray(mask_box(base.size, (60, 160, 90, 190))).copy()
    mask[160:190, 1050:1080] = 255

    await eraser.erase(base, Image.fromarray(mask), "cpu")

    assert len(module.calls) == 2
    assert all(max(image.shape[-2:]) < 500 for image, _ in module.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["MAX_COMPONENTS", "MAX_EDIT_GROUPS"])
async def test_excess_edit_groups_fall_back_to_one_bounded_inference(tmp_path, monkeypatch, guard):
    from app.services import inpaint_pipeline

    eraser, module = make_eraser(tmp_path, monkeypatch)
    base = photo((1200, 400))
    mask = np.asarray(mask_box(base.size, (60, 160, 90, 190))).copy()
    mask[160:190, 1050:1080] = 255
    monkeypatch.setattr(inpaint_pipeline, guard, 1)

    await eraser.erase(base, Image.fromarray(mask), "cpu")

    assert len(module.calls) == 1
    assert max(module.calls[0][0].shape[-2:]) <= 1600


@pytest.mark.asyncio
async def test_capabilities_offer_lama_only_when_installed(tmp_path):
    from app.api.routes import generation_capabilities
    from app.services.engines.auto_eraser import ERASER_AUTO_MODEL_ID
    from app.services.model_registry import ModelRegistry

    class FakeDevices:
        def list_devices(self):
            return [{"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"}]

    installed = make_settings(tmp_path / "installed")
    missing = make_settings(tmp_path / "missing", installed=False)

    offered = await generation_capabilities(
        registry=ModelRegistry(installed), devices_service=FakeDevices(), settings=installed,
    )
    absent = await generation_capabilities(
        registry=ModelRegistry(missing), devices_service=FakeDevices(), settings=missing,
    )

    entries = [entry for entry in offered.models if entry.id == ERASER_LAMA_MODEL_ID]
    assert len(entries) == 1
    assert offered.models[0].id == ERASER_AUTO_MODEL_ID
    assert entries[0].supports_inpaint is True
    assert entries[0].erase_only is True
    assert "LaMa" in entries[0].name
    assert not any(entry.id == ERASER_LAMA_MODEL_ID for entry in absent.models)


@pytest.mark.asyncio
async def test_capabilities_list_eraser_auto_first_when_lama_installed(tmp_path):
    from app.api.routes import generation_capabilities
    from app.services.engines.auto_eraser import ERASER_AUTO_MODEL_ID
    from app.services.model_registry import ModelRegistry

    class FakeDevices:
        def list_devices(self):
            return [{"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"}]

    installed = make_settings(tmp_path / "installed-auto")
    missing = make_settings(tmp_path / "missing-auto", installed=False)
    offered = await generation_capabilities(
        registry=ModelRegistry(installed),
        devices_service=FakeDevices(),
        settings=installed,
    )
    absent = await generation_capabilities(
        registry=ModelRegistry(missing),
        devices_service=FakeDevices(),
        settings=missing,
    )

    assert offered.models[0].id == ERASER_AUTO_MODEL_ID
    assert offered.models[0].erase_only is True
    assert offered.models[0].supports_inpaint is True
    assert not any(model.id == ERASER_AUTO_MODEL_ID for model in absent.models)


@pytest.mark.asyncio
async def test_job_manager_dispatches_lama_with_protect_mask(tmp_path):
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.generation_job_manager import GenerationJobManager
    from app.services.model_registry import ModelRegistry

    class FakeEraser:
        def __init__(self):
            self.calls = []

        def available(self):
            return True

        async def erase(self, base_image, mask_image, device, protect_image=None):
            self.calls.append((base_image, mask_image, device, protect_image))
            return Image.new("RGB", base_image.size, (192, 192, 192))

    settings = make_settings(tmp_path)
    lama = FakeEraser()
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=DeviceSemaphores(settings),
        registry=ModelRegistry(settings), upscale_engine=None, lama_eraser=lama,
    )
    base_path, mask_path, protect_path = (tmp_path / name for name in ("base.png", "mask.png", "protect.png"))
    photo().save(base_path)
    mask_box((320, 240), (100, 80, 140, 120)).save(mask_path)
    mask_box((320, 240), (140, 80, 180, 120)).save(protect_path)

    job = await manager.create_job(
        prompt="borrar", model_id=ERASER_LAMA_MODEL_ID, device="cpu",
        init_image_path=base_path, mask_image_path=mask_path, protect_mask_path=protect_path,
    )
    await manager._run_engine(job)

    assert len(lama.calls) == 1
    base, mask, device, protect = lama.calls[0]
    assert base.size == mask.size == protect.size == (320, 240)
    assert device == "cpu"
    assert np.asarray(protect)[100, 160] == 255
    assert job.output_path.is_file()
    with Image.open(job.output_path) as result:
        assert result.getpixel((0, 0)) == (192, 192, 192)


@pytest.mark.asyncio
async def test_job_manager_rejects_missing_lama_with_pack_message(tmp_path):
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.generation_job_manager import GenerationJobManager
    from app.services.missing_pack import PACK_LABELS
    from app.services.model_registry import ModelRegistry

    settings = make_settings(tmp_path, installed=False)
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=DeviceSemaphores(settings),
        registry=ModelRegistry(settings), upscale_engine=None, lama_eraser=LamaEraser(settings),
    )

    with pytest.raises(ValueError, match=PACK_LABELS["lama"]):
        await manager.create_job(prompt="borrar", model_id=ERASER_LAMA_MODEL_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_base,with_mask", [(False, False), (True, False), (False, True)])
async def test_lama_job_requires_both_base_and_mask(tmp_path, with_base, with_mask):
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.generation_job_manager import GenerationJobManager
    from app.services.model_registry import ModelRegistry

    settings = make_settings(tmp_path)
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=DeviceSemaphores(settings),
        registry=ModelRegistry(settings), upscale_engine=None, lama_eraser=LamaEraser(settings),
    )
    base_path, mask_path = tmp_path / "base.png", tmp_path / "mask.png"
    photo().save(base_path)
    mask_box((320, 240), (100, 80, 140, 120)).save(mask_path)

    with pytest.raises(ValueError):
        await manager.create_job(
            prompt="borrar", model_id=ERASER_LAMA_MODEL_ID,
            init_image_path=base_path if with_base else None,
            mask_image_path=mask_path if with_mask else None,
        )

    assert manager.jobs == {}


@pytest.mark.parametrize("requested_device", [None, "dml:0"])
def test_lama_reserves_cpu_even_when_gpu_requested_or_default(tmp_path, requested_device):
    from app.models import GenerationJob
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.generation_job_manager import GenerationJobManager
    from app.services.model_registry import ModelRegistry

    settings = make_settings(tmp_path).model_copy(update={"default_device": "dml:0"})
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=DeviceSemaphores(settings),
        registry=ModelRegistry(settings), upscale_engine=None, lama_eraser=LamaEraser(settings),
    )
    lama = GenerationJob(prompt="borrar", model_id=ERASER_LAMA_MODEL_ID, device=requested_device)
    migan = GenerationJob(prompt="borrar", model_id="eraser-migan", device=requested_device)

    assert manager._reservation_device(lama) == "cpu"
    assert manager._reservation_device(migan) == "dml:0"
