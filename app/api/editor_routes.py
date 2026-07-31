from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.auth_deps import require
from app.schemas import EditorSegmentRequest
from app.services.auth.permissions import Permission
from app.services.editor_segmenter import EditorSegmenter, SegmenterUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/editor", tags=["editor"])


def get_editor_segmenter(request: Request) -> EditorSegmenter:
    return request.app.state.editor_segmenter


@router.get("/capabilities")
async def editor_capabilities(
    segmenter: EditorSegmenter = Depends(get_editor_segmenter),
) -> dict[str, bool]:
    return {"tapSelect": segmenter.available()}


@router.post(
    "/segment",
    dependencies=[Depends(require(Permission.jobs_create))],
    responses={200: {"content": {"image/png": {}}}},
)
async def segment_object(
    payload: EditorSegmentRequest,
    segmenter: EditorSegmenter = Depends(get_editor_segmenter),
) -> Response:
    """Máscara del objeto tocado (PNG blanco=objeto), coords en píxeles de la
    imagen original subida con POST /generation/init-image."""
    from app.api.routes import _resolve_init_image
    from app.config import get_settings

    settings = get_settings()
    image_path = _resolve_init_image(settings, payload.image_token)
    if image_path is None:
        raise HTTPException(status_code=400, detail="imageToken is required")
    device = payload.device or settings.default_device
    if device != "cpu" and not device.startswith("dml:"):
        raise HTTPException(status_code=400, detail=f"Unsupported device: {device!r}")
    try:
        png = await segmenter.segment(image_path, payload.x, payload.y, device)
    except SegmenterUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Editor segmentation failed")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc
    return Response(content=png, media_type="image/png")
