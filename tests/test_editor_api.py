from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from app.api.editor_routes import segment_object
from app.api.routes import upload_init_image
from app.config import get_settings
from app.schemas import EditorSegmentRequest
from app.services.editor_segmenter import SegmenterUnavailableError
from app.services.storage import StorageService


class FakeSegmenter:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[tuple] = []

    def available(self) -> bool:
        return self._available

    async def segment(self, image_path, x, y, device):
        if not self._available:
            raise SegmenterUnavailableError("corré scripts/download-mobilesam.ps1")
        self.calls.append((image_path, x, y, device))
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        buffer = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buffer, format="PNG")
        return buffer.getvalue()


async def upload_token() -> str:
    settings = get_settings()
    StorageService(settings).ensure_directories()
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buffer, format="PNG")
    response = await upload_init_image(
        file=UploadFile(filename="base.png", file=io.BytesIO(buffer.getvalue())),
        storage=StorageService(settings),
        settings=settings,
    )
    return response.init_image_token


@pytest.mark.asyncio
async def test_segment_happy_path_returns_png() -> None:
    token = await upload_token()
    segmenter = FakeSegmenter()

    response = await segment_object(
        EditorSegmentRequest(imageToken=token, x=3, y=3), segmenter=segmenter
    )

    assert response.media_type == "image/png"
    assert Image.open(io.BytesIO(response.body)).mode == "L"
    assert segmenter.calls and segmenter.calls[0][3] in ("cpu",) or segmenter.calls[0][3].startswith("dml:")


@pytest.mark.asyncio
async def test_segment_unknown_token_is_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await segment_object(
            EditorSegmentRequest(imageToken="deadbeef", x=1, y=1), segmenter=FakeSegmenter()
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_segment_unavailable_is_503_with_hint() -> None:
    token = await upload_token()

    with pytest.raises(HTTPException) as exc_info:
        await segment_object(
            EditorSegmentRequest(imageToken=token, x=1, y=1), segmenter=FakeSegmenter(available=False)
        )

    assert exc_info.value.status_code == 503
    assert "download-mobilesam" in exc_info.value.detail


async def _upload_png(image: Image.Image, name: str) -> str:
    settings = get_settings()
    StorageService(settings).ensure_directories()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = await upload_init_image(
        file=UploadFile(filename=name, file=io.BytesIO(buffer.getvalue())),
        storage=StorageService(settings),
        settings=settings,
    )
    return response.init_image_token


@pytest.mark.asyncio
async def test_insert_object_composites_and_stages_the_result() -> None:
    from app.api.editor_routes import insert_object
    from app.api.routes import _resolve_init_image
    from app.schemas import InsertObjectRequest
    import base64

    target_token = await _upload_png(Image.new("RGB", (16, 16), (0, 0, 255)), "target.png")
    source_token = await _upload_png(Image.new("RGB", (8, 8), (255, 0, 0)), "source.png")
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (2, 2, 6, 6))
    mask_token = await _upload_png(mask, "mask.png")

    response = await insert_object(
        InsertObjectRequest(
            targetToken=target_token, sourceToken=source_token, sourceMaskToken=mask_token,
            x=4, y=4, width=8, height=8, featherPx=0, matchColor=False, harmonize=False,
        ),
        request=None,
    )

    assert response.job_id is None
    composite = Image.open(io.BytesIO(base64.b64decode(response.composite_png_base64)))
    assert composite.size == (16, 16)
    # El objeto rojo quedó pegado dentro del rectángulo pedido...
    assert composite.getpixel((7, 7))[0] > 200
    # ...y afuera el destino sigue azul intacto.
    assert composite.getpixel((1, 1))[2] > 200
    # El composite queda staged: sirve como imagen de partida de más ediciones.
    assert _resolve_init_image(get_settings(), response.composite_token) is not None


@pytest.mark.asyncio
async def test_insert_object_harmonize_rejects_4ch_models() -> None:
    from types import SimpleNamespace

    from app.api.editor_routes import insert_object
    from app.schemas import InsertObjectRequest

    target_token = await _upload_png(Image.new("RGB", (16, 16), (0, 0, 255)), "target.png")
    source_token = await _upload_png(Image.new("RGB", (8, 8), (255, 0, 0)), "source.png")
    mask = Image.new("L", (8, 8), 255)
    mask_token = await _upload_png(mask, "mask.png")

    # Un modelo cuyo model_index NO declara clase de inpainting dedicada:
    # la armonización a strength parcial lo reinventaría, así que se rechaza.
    entry = SimpleNamespace(file_path="no-existe", kind=None, source="hf:x/y")
    manager = SimpleNamespace(registry=SimpleNamespace(get=lambda model_id: entry))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(generation_job_manager=manager)))

    with pytest.raises(HTTPException) as excinfo:
        await insert_object(
            InsertObjectRequest(
                targetToken=target_token, sourceToken=source_token, sourceMaskToken=mask_token,
                x=0, y=0, width=8, height=8, harmonize=True, modelId="gen--x--y",
            ),
            request=request,
        )
    assert excinfo.value.status_code == 400
    assert "inpainting dedicado" in excinfo.value.detail
