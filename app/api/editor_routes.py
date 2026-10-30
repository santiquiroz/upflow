from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.auth_deps import require
from app.schemas import (
    EditorHealRequest,
    EditorPeopleRequest,
    EditorPeopleResponse,
    EditorSegmentRequest,
    InitImageResponse,
    InsertObjectRequest,
    InsertObjectResponse,
)
from app.services.auth.permissions import Permission
from app.services.erase_auto import SMALL_ZONE_SIDE
from app.services.engines.lama_eraser import LamaEraser
from app.services.editor_people import EditorPeople
from app.services.editor_segmenter import EditorSegmenter, SegmenterUnavailableError
from app.services.missing_pack import missing_pack_message
from app.services.person_detector import PersonDetectorUnavailableError
from app.services.engines.onnx_common import readable_error_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/editor", tags=["editor"])


def get_editor_segmenter(request: Request) -> EditorSegmenter:
    return request.app.state.editor_segmenter


def get_editor_people(request: Request) -> EditorPeople:
    return request.app.state.editor_people


def _ensure_heal_mask_is_valid(mask) -> None:
    import numpy as np

    ys, xs = np.where(mask > 0)
    if not xs.size:
        raise HTTPException(status_code=400, detail="La máscara está vacía.")
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    if max(right - left, bottom - top) > SMALL_ZONE_SIDE:
        raise HTTPException(status_code=400, detail="La zona es grande: usá Quitar.")


def _prepare_heal_arrays(image_path: Path, mask_path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as source:
        base = source.convert("RGB")
        canvas = np.asarray(base, dtype=np.uint8).copy()
    with Image.open(mask_path) as source_mask:
        gray_mask = source_mask.convert("L")
        if gray_mask.size != base.size:
            gray_mask = gray_mask.resize(base.size, Image.Resampling.NEAREST)
        mask = np.where(np.asarray(gray_mask) > 127, 255, 0).astype(np.uint8)
    _ensure_heal_mask_is_valid(mask)
    return canvas, mask


def _base_original_filename(image_path: Path, token: str) -> tuple[str, str]:
    original_filename = image_path.name.removeprefix(f"{token}-")
    return original_filename, Path(original_filename).stem


def _heal_destination(settings, image_path: Path, image_token: str) -> tuple[str, Path, str]:
    from uuid import uuid4

    original_filename, stem = _base_original_filename(image_path, image_token)
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{stem}.png"
    return token, destination, original_filename


def _heal_blocking(
    image_path: Path, mask_path: Path, lama: LamaEraser, destination: Path
) -> tuple[int, int]:
    import numpy as np
    from PIL import Image

    from app.services.inpaint_mask import grow_per_component, grow_small_hulls

    canvas, group_mask = _prepare_heal_arrays(image_path, mask_path)
    group_mask = grow_per_component(grow_small_hulls(group_mask), None, None)
    healed = lama.fill_group(
        canvas, group_mask, np.zeros(group_mask.shape, dtype=bool)
    )
    Image.fromarray(healed).save(destination, format="PNG", compress_level=1)
    return int(healed.shape[1]), int(healed.shape[0])


@router.post(
    "/heal",
    response_model=InitImageResponse,
    status_code=201,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def heal(payload: EditorHealRequest, request: Request) -> InitImageResponse:
    import asyncio

    from app.api.routes import _resolve_init_image
    from app.config import get_settings

    settings = get_settings()
    image_path = _resolve_init_image(settings, payload.image_token)
    mask_path = _resolve_init_image(settings, payload.mask_token)
    if image_path is None or mask_path is None:
        raise HTTPException(status_code=400, detail="Faltan tokens de imagen")

    lama: LamaEraser = request.app.state.lama_eraser
    if not lama.available():
        raise HTTPException(status_code=503, detail=missing_pack_message("lama"))

    token, destination, original_filename = _heal_destination(
        settings, image_path, payload.image_token
    )
    width, height = await asyncio.to_thread(
        _heal_blocking, image_path, mask_path, lama, destination
    )

    return InitImageResponse(
        init_image_token=token,
        original_filename=original_filename,
        width=width,
        height=height,
    )


def _resolve_editor_image_and_device(
    image_token: str, requested_device: str | None
) -> tuple[Path, str]:
    from app.api.routes import _resolve_init_image
    from app.config import get_settings

    image_path = _resolve_init_image(get_settings(), image_token)
    if image_path is None:
        raise HTTPException(status_code=400, detail="imageToken is required")
    device = requested_device or "cpu"
    if device != "cpu" and not device.startswith("dml:"):
        raise HTTPException(status_code=400, detail=f"Unsupported device: {device!r}")
    return image_path, device


@router.get("/capabilities")
async def editor_capabilities(
    segmenter: EditorSegmenter = Depends(get_editor_segmenter),
    people: EditorPeople = Depends(get_editor_people),
) -> dict[str, bool]:
    return {"tapSelect": segmenter.available(), "people": people.available()}


@router.post(
    "/people",
    response_model=EditorPeopleResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def detect_people(
    payload: EditorPeopleRequest,
    people: EditorPeople = Depends(get_editor_people),
) -> EditorPeopleResponse:
    image_path, device = _resolve_editor_image_and_device(
        payload.image_token, payload.device
    )
    try:
        detected_people = await people.people(image_path, device)
    except PersonDetectorUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        message = readable_error_text(exc)
        logger.exception("Editor people detection failed: %s", message)
        raise HTTPException(
            status_code=500, detail=f"People detection failed: {message}"
        ) from exc
    return EditorPeopleResponse(people=detected_people)


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
    image_path, device = _resolve_editor_image_and_device(
        payload.image_token, payload.device
    )
    # MobileSAM es chico y en GPU compite por la VRAM con el job de generación en curso.
    try:
        if payload.box is not None:
            png = await segmenter.segment_box(image_path, tuple(payload.box), device)
        else:
            png = await segmenter.segment(image_path, payload.x, payload.y, device)
    except SegmenterUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        message = readable_error_text(exc)
        logger.exception("Editor segmentation failed: %s", message)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {message}") from exc
    return Response(content=png, media_type="image/png")


HARMONIZE_STRENGTH = 0.35
HARMONIZE_STEPS = 25
HARMONIZE_GUIDANCE = 5.0
HARMONIZE_PROMPT = "same scene, seamless integration, consistent lighting and grain, high quality"
JOB_DIMENSION_MULTIPLE = 64
JOB_DIMENSION_MAX = 1024


def _mask_preparation_for(harmonize_blend: float) -> dict[str, int]:
    """Preparación de máscara que le toca al motor según el perfil pedido.

    El perfil continuo ya trae su propia dilatación y su propia caída: volver a
    dilatarlo en el motor (MaxFilter de 8 px) empujaría el 255 de la costura
    hacia adentro y se comería justo el centro que la máscara viene a preservar.
    Con el perfil uniforme clásico se dejan los defaults del motor, para que el
    comportamiento viejo siga siendo exactamente el viejo.
    """
    return {} if harmonize_blend >= 1.0 else {"mask_dilate_px": 0, "mask_feather_px": 0}


def _snap_job_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, JOB_DIMENSION_MAX / max(width, height, 1))

    def snap(value: int) -> int:
        snapped = round(value * scale / JOB_DIMENSION_MULTIPLE) * JOB_DIMENSION_MULTIPLE
        return min(JOB_DIMENSION_MAX, max(JOB_DIMENSION_MULTIPLE, snapped))

    return snap(width), snap(height)


@router.post(
    "/insert-object",
    response_model=InsertObjectResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def insert_object(payload: InsertObjectRequest, request: Request) -> InsertObjectResponse:
    """Pega un objeto recortado de otra imagen en el destino, integrado.

    Fase 0 (siempre): composite con feather + igualación de color MK.
    Fase 1 (harmonize=True): pase de inpaint 9ch sobre la zona pegada — exige un
    modelo de inpainting dedicado, porque uno de 4 canales fuerza strength 1.0 y
    reinventaría el objeto recién pegado.
    Fase 2 (harmonizeBlend < 1): esa zona no va a intensidad plana sino con un
    perfil continuo (máximo en la costura, casi cero en el centro) que el inpaint
    aplica por difusión diferencial — se re-genera la unión, no el objeto.
    """
    import asyncio
    import base64
    import io
    from uuid import uuid4

    from PIL import Image

    from app.api.routes import (
        _entry_inpaint_only,
        _resolve_init_image,
        get_generation_job_manager,
    )
    from app.api.auth_deps import current_user_from_request
    from app.config import get_settings
    from app.services.object_transfer import PasteSpec, transfer_object

    settings = get_settings()
    target_path = _resolve_init_image(settings, payload.target_token)
    source_path = _resolve_init_image(settings, payload.source_token)
    mask_path = _resolve_init_image(settings, payload.source_mask_token)
    if target_path is None or source_path is None or mask_path is None:
        raise HTTPException(status_code=400, detail="Faltan tokens de imagen")

    # Gates del harmonize ANTES del compositing: rechazar un modelo inválido
    # después de computar el composite desperdicia el trabajo y confunde.
    manager = None
    if payload.harmonize:
        if not payload.model_id:
            raise HTTPException(status_code=400, detail="harmonize requiere modelId")
        manager = get_generation_job_manager(request)
        entry = manager.registry.get(payload.model_id) if manager.registry else None
        if entry is None:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        if not _entry_inpaint_only(settings, entry):
            raise HTTPException(
                status_code=400,
                detail=(
                    "La armonización necesita un modelo de inpainting dedicado (9 canales): "
                    "creá la versión de inpainting de tu modelo desde Modelos, o instalá uno."
                ),
            )

    target_object_mask_path = (
        _resolve_init_image(settings, payload.target_mask_token)
        if payload.target_mask_token
        else None
    )

    def run_transfer():
        import numpy as np

        from app.services.inpaint_mask import dilate_mask, mask_bbox

        with Image.open(target_path) as target_img:
            target = target_img.convert("RGB")
        with Image.open(source_path) as source_img:
            source = source_img.convert("RGB")
        with Image.open(mask_path) as mask_img:
            source_mask = mask_img.convert("L")

        target_object = None
        if target_object_mask_path is not None:
            with Image.open(target_object_mask_path) as tm:
                target_object = tm.convert("L")
            if target_object.size != target.size:
                target_object = target_object.resize(target.size, Image.BILINEAR)
            bbox = mask_bbox(np.asarray(target_object))
            if bbox is None:
                raise ValueError("La máscara del objeto a reemplazar está vacía")
            left, top, right, bottom = bbox
            spec = PasteSpec(
                x=left, y=top, width=max(8, right - left), height=max(8, bottom - top),
                feather_px=payload.feather_px, match_color=payload.match_color,
                harmonize_blend=payload.harmonize_blend,
            )
        else:
            spec = PasteSpec(
                x=payload.x, y=payload.y, width=payload.width, height=payload.height,
                feather_px=payload.feather_px, match_color=payload.match_color,
                harmonize_blend=payload.harmonize_blend,
            )
        composite, paste_mask = transfer_object(source, source_mask, target, spec)
        if target_object is not None:
            # La armonización debe cubrir también lo que SOBRA del objeto viejo
            # (el nuevo puede ser más chico): unión de máscaras, con el viejo
            # dilatado para que el modelo repinte sus bordes desde el contexto.
            old = dilate_mask(np.asarray(target_object), 12)
            union = np.maximum(np.asarray(paste_mask), old)
            paste_mask = Image.fromarray(union, mode="L")
        return composite, paste_mask

    try:
        composite, paste_mask = await asyncio.to_thread(run_transfer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = uuid4().hex
    composite_path = settings.uploads_path / f"{token}-insert.png"
    composite.save(composite_path)
    mask_token = uuid4().hex
    paste_mask_path = settings.uploads_path / f"{mask_token}-insert-mask.png"
    paste_mask.save(paste_mask_path)

    job_id = None
    if payload.harmonize and manager is not None:
        from app.exceptions import QueueFullError, QuotaExceededError

        # El job exige dimensiones múltiplo de 64 en 64..1024; el composite
        # hereda el tamaño arbitrario del destino. Escalar preservando aspecto
        # y redondear — sin esto, casi cualquier foto real daba 500.
        job_width, job_height = _snap_job_dimensions(composite.width, composite.height)
        try:
            job = await manager.create_job(
                prompt=payload.prompt or HARMONIZE_PROMPT,
                model_id=payload.model_id,
                steps=HARMONIZE_STEPS,
                guidance=HARMONIZE_GUIDANCE,
                width=job_width,
                height=job_height,
                device=payload.device,
                init_image_path=composite_path,
                strength=HARMONIZE_STRENGTH,
                mask_image_path=paste_mask_path,
                **_mask_preparation_for(payload.harmonize_blend),
                owner=current_user_from_request(request),
            )
        except QueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except QuotaExceededError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job_id = job.id

    buffer = io.BytesIO()
    composite.save(buffer, format="PNG")
    return InsertObjectResponse(
        composite_token=token,
        composite_png_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
        job_id=job_id,
    )
