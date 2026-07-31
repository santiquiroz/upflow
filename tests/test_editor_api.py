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
