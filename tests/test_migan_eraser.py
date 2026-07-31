from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.engines.migan_eraser import (
    ERASER_MODEL_ID,
    MiganEraser,
    MiganUnavailableError,
    build_migan_inputs,
)


def make_settings(tmp_path: Path, installed: bool = True) -> Settings:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "migan.onnx"
    if installed:
        model.write_bytes(b"onnx")
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), MIGAN_MODEL=str(model))


def photo(size: tuple[int, int] = (200, 120)) -> Image.Image:
    width, height = size
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    data[:, :, 1] = 90
    data[:, :, 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    return Image.fromarray(data)


def mask_box(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    return Image.fromarray(mask, mode="L")


# --- contrato de entrada (lo que el modelo espera de verdad) ----------------


def test_inputs_are_uint8_nchw() -> None:
    image, mask = build_migan_inputs(photo((64, 48)), mask_box((64, 48), (10, 10, 30, 30)))

    assert image.dtype == np.uint8 and mask.dtype == np.uint8
    assert image.shape == (1, 3, 48, 64)
    assert mask.shape == (1, 1, 48, 64)
    assert image.max() > 1  # 0..255 crudo, sin normalizar


def test_mask_polarity_is_inverted_for_the_model() -> None:
    # Nuestra convención: 255 = editar. La del modelo: 255 = CONSERVAR.
    # Mandarla sin invertir regenera toda la foto MENOS lo que se quería borrar.
    _, mask = build_migan_inputs(photo((64, 48)), mask_box((64, 48), (10, 10, 30, 30)))

    assert mask[0, 0, 20, 20] == 0  # dentro de lo marcado -> hueco a rellenar
    assert mask[0, 0, 2, 2] == 255  # afuera -> region conocida


def test_mask_is_hardened_to_0_or_255() -> None:
    # El modelo trata la máscara como alpha continuo: un borde difuminado le
    # entrega un relleno a medias en vez de un borrado limpio.
    soft = np.full((48, 64), 128, dtype=np.uint8)
    soft[20:30, 20:30] = 200
    _, mask = build_migan_inputs(photo((64, 48)), Image.fromarray(soft, mode="L"))

    assert set(np.unique(mask).tolist()) <= {0, 255}


# --- servicio ---------------------------------------------------------------


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, _outputs, inputs):
        self.calls.append(inputs)
        image = inputs["image"]
        # El modelo real devuelve la foto con el hueco rellenado; acá se marca
        # todo en rojo para que el test distinga lo generado de lo original.
        result = np.zeros_like(image)
        result[:, 0, :, :] = 255
        return [result]


def make_eraser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[MiganEraser, FakeSession]:
    eraser = MiganEraser(make_settings(tmp_path))
    session = FakeSession()
    monkeypatch.setattr(eraser, "_create_session", lambda device: session)
    return eraser, session


@pytest.mark.asyncio
async def test_erase_returns_the_original_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eraser, _ = make_eraser(tmp_path, monkeypatch)
    base = photo((200, 120))

    result = await eraser.erase(base, mask_box((200, 120), (80, 40, 120, 80)), "dml:0")

    assert result.size == (200, 120)


@pytest.mark.asyncio
async def test_pixels_outside_the_mask_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El modelo sangra fuera del hueco (medido: hasta 26/255 alrededor). La
    # composición final tiene que devolver el original ahí, exacto.
    eraser, _ = make_eraser(tmp_path, monkeypatch)
    base = photo((200, 120))

    result = await eraser.erase(base, mask_box((200, 120), (80, 40, 120, 80)), "dml:0")

    before = np.asarray(base)
    after = np.asarray(result)
    np.testing.assert_array_equal(after[0:20, 0:20], before[0:20, 0:20])
    np.testing.assert_array_equal(after[100:120, 180:200], before[100:120, 180:200])


@pytest.mark.asyncio
async def test_the_marked_area_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eraser, _ = make_eraser(tmp_path, monkeypatch)
    base = photo((200, 120))

    result = await eraser.erase(base, mask_box((200, 120), (80, 40, 120, 80)), "dml:0")

    center = np.asarray(result)[60, 100]
    assert center[0] > 200 and center[1] < 60


@pytest.mark.asyncio
async def test_an_empty_mask_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Con la máscara vacía el modelo NO es no-op: devuelve otra imagen. Se corta antes.
    eraser, session = make_eraser(tmp_path, monkeypatch)
    base = photo((64, 48))

    result = await eraser.erase(base, mask_box((64, 48), (0, 0, 0, 0)), "dml:0")

    np.testing.assert_array_equal(np.asarray(result), np.asarray(base))
    assert session.calls == []


@pytest.mark.asyncio
async def test_missing_model_gives_an_actionable_error(tmp_path: Path) -> None:
    eraser = MiganEraser(make_settings(tmp_path, installed=False))

    with pytest.raises(MiganUnavailableError, match="download-migan"):
        await eraser.erase(photo(), mask_box((200, 120), (10, 10, 20, 20)), "cpu")


def test_available_reflects_the_model_on_disk(tmp_path: Path) -> None:
    assert MiganEraser(make_settings(tmp_path)).available() is True
    assert MiganEraser(make_settings(tmp_path / "otro", installed=False)).available() is False


def test_model_id_is_stable() -> None:
    # Viaja en la API y en el selector del Editor.
    assert ERASER_MODEL_ID == "eraser-migan"


# --- el motor aparece como una opción MÁS del selector ----------------------


@pytest.mark.asyncio
async def test_capabilities_offer_the_eraser_only_when_installed(tmp_path: Path) -> None:
    from app.api.routes import generation_capabilities
    from app.services.model_registry import ModelRegistry

    class FakeDevices:
        def list_devices(self):
            return [{"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"}]

    installed = make_settings(tmp_path / "con")
    missing = make_settings(tmp_path / "sin", installed=False)

    with_eraser = await generation_capabilities(
        registry=ModelRegistry(installed), devices_service=FakeDevices(), settings=installed
    )
    without = await generation_capabilities(
        registry=ModelRegistry(missing), devices_service=FakeDevices(), settings=missing
    )

    eraser = [m for m in with_eraser.models if m.id == ERASER_MODEL_ID]
    assert len(eraser) == 1
    # Se ofrece para quitar, y se marca para que Reemplazar no lo muestre.
    assert eraser[0].supports_inpaint is True
    assert eraser[0].erase_only is True
    assert [m for m in without.models if m.id == ERASER_MODEL_ID] == []


@pytest.mark.asyncio
async def test_job_manager_rejects_the_eraser_when_not_installed(tmp_path: Path) -> None:
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.generation_job_manager import GenerationJobManager
    from app.services.model_registry import ModelRegistry

    settings = make_settings(tmp_path, installed=False)
    manager = GenerationJobManager(
        settings,
        engine=None,
        device_semaphores=DeviceSemaphores(settings),
        registry=ModelRegistry(settings),
        upscale_engine=None,
        migan_eraser=MiganEraser(settings),
    )

    with pytest.raises(ValueError, match="download-migan|no está instalado"):
        await manager.create_job(prompt="x", model_id=ERASER_MODEL_ID)
