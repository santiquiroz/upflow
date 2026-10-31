from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.api.editor_routes import (
    detect_people as editor_people_route,
    editor_capabilities,
    segment_object,
)
from app.api.routes import download_init_image, upload_init_image
from app.config import get_settings
from app.schemas import EditorPeopleRequest, EditorSegmentRequest
from app.services.editor_people import EditorPeople
from app.services.person_detector import PersonBox, PersonDetectorUnavailableError
from app.services.editor_segmenter import SegmenterUnavailableError
from app.services.storage import StorageService


class FakeSegmenter:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[tuple] = []
        self.box_calls: list[tuple] = []

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

    async def segment_box(self, image_path, box, device):
        if not self._available:
            raise SegmenterUnavailableError("corré scripts/download-mobilesam.ps1")
        self.box_calls.append((image_path, box, device))
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        buffer = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buffer, format="PNG")
        return buffer.getvalue()


class FakePeople:
    def __init__(self, people: list, available: bool = True) -> None:
        self._people = people
        self._available = available
        self.calls: list[tuple] = []

    def available(self) -> bool:
        return self._available

    async def people(self, image_path, device):
        if not self._available:
            raise PersonDetectorUnavailableError("corré scripts/download-person-detector.ps1")
        self.calls.append((image_path, device))
        return self._people


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
    assert segmenter.calls[0][3] == "cpu"


def test_segment_request_requires_point_or_box() -> None:
    with pytest.raises(ValidationError):
        EditorSegmentRequest(imageToken="ab", x=1, y=1, box=[0, 0, 5, 5])
    with pytest.raises(ValidationError):
        EditorSegmentRequest(imageToken="ab")
    with pytest.raises(ValidationError):
        EditorSegmentRequest(imageToken="ab", box=[5, 0, 1, 5])


@pytest.mark.asyncio
async def test_segment_route_with_box_calls_segment_box() -> None:
    token = await upload_token()
    segmenter = FakeSegmenter()

    response = await segment_object(
        EditorSegmentRequest(imageToken=token, box=[1, 2, 6, 7]), segmenter=segmenter
    )

    assert response.media_type == "image/png"
    assert segmenter.box_calls[0][1:] == ((1.0, 2.0, 6.0, 7.0), "cpu")


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


@pytest.mark.asyncio
async def test_segment_explicit_dml_device_is_respected() -> None:
    token = await upload_token()
    segmenter = FakeSegmenter()

    await segment_object(
        EditorSegmentRequest(imageToken=token, x=3, y=3, device="dml:0"), segmenter=segmenter
    )

    assert segmenter.calls[0][3] == "dml:0"


@pytest.mark.asyncio
async def test_segment_garbled_onnx_error_returns_readable_detail() -> None:
    token = await upload_token()

    class FailingSegmenter(FakeSegmenter):
        async def segment(self, image_path, x, y, device):
            raise UnicodeDecodeError(
                "utf-8", "acción".encode("cp1252"), 3, 4, "invalid continuation byte"
            )

    with pytest.raises(HTTPException) as exc_info:
        await segment_object(
            EditorSegmentRequest(imageToken=token, x=1, y=1), segmenter=FailingSegmenter()
        )

    assert exc_info.value.status_code == 500
    assert "acci" in exc_info.value.detail
    assert "codec can't decode" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_people_returns_roles_and_faces() -> None:
    from app.schemas import EditorPerson

    token = await upload_token()
    person = EditorPerson(
        id=0, box=[10, 20, 110, 220], score=0.9, role="subject",
        face=[30, 40, 60, 70], anchor=[45, 55],
    )

    response = await editor_people_route(
        EditorPeopleRequest(imageToken=token), people=FakePeople([person])
    )

    assert response.people[0].role == "subject"
    assert response.people[0].box == [10, 20, 110, 220]
    assert response.people[0].face == [30, 40, 60, 70]
    assert response.people[0].anchor == [45, 55]


@pytest.mark.asyncio
async def test_people_unavailable_is_503_with_pack_hint() -> None:
    token = await upload_token()

    with pytest.raises(HTTPException) as exc_info:
        await editor_people_route(
            EditorPeopleRequest(imageToken=token), people=FakePeople([], available=False)
        )

    assert exc_info.value.status_code == 503
    assert "download-person-detector" in exc_info.value.detail


@pytest.mark.asyncio
async def test_people_second_call_hits_cache(tmp_path, monkeypatch) -> None:
    import app.services.editor_people as editor_people_module
    from app.schemas import EditorPerson
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)

    class FakeDetector:
        calls = 0

        def available(self):
            return True

        def detect(self, rgb, device):
            self.calls += 1
            return [PersonBox((10, 10, 50, 60), 0.9)]

    monkeypatch.setattr(editor_people_module, "classify_roles", lambda rgb, detections: ["subject"])
    monkeypatch.setattr(editor_people_module, "face_box", lambda rgb, box: (20, 15, 35, 30))
    detector = FakeDetector()
    service = EditorPeople(detector)

    first = await service.people(image_path, "cpu")
    second = await service.people(image_path, "cpu")

    assert first == second == [
        EditorPerson(
            id=0,
            box=[10, 10, 50, 60],
            score=0.9,
            role="subject",
            face=[20, 15, 35, 30],
            anchor=[27, 22],
        )
    ]
    assert detector.calls == 1


@pytest.mark.asyncio
async def test_people_cache_returns_independent_models(tmp_path, monkeypatch) -> None:
    import app.services.editor_people as editor_people_module
    from PIL import Image

    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)

    class FakeDetector:
        def available(self):
            return True

        def detect(self, rgb, device):
            return [PersonBox((10, 10, 50, 60), 0.9)]

    monkeypatch.setattr(editor_people_module, "classify_roles", lambda rgb, detections: ["subject"])
    monkeypatch.setattr(editor_people_module, "face_box", lambda rgb, box: None)
    service = EditorPeople(FakeDetector())

    first = await service.people(image_path, "cpu")
    first[0].box[0] = 999
    second = await service.people(image_path, "cpu")

    assert second[0].box == [10, 10, 50, 60]
    assert second[0].anchor == [30, 16]


@pytest.mark.asyncio
async def test_people_checks_availability_before_opening_image(monkeypatch, tmp_path) -> None:
    from app.services.person_detector import PersonDetectorUnavailableError

    class UnavailableDetector:
        def available(self):
            return False

    def fail_open(_path):
        raise AssertionError("image must not be decoded")

    monkeypatch.setattr("PIL.Image.open", fail_open)
    service = EditorPeople(UnavailableDetector())

    with pytest.raises(PersonDetectorUnavailableError):
        await service.people(tmp_path / "missing.png", "cpu")


@pytest.mark.asyncio
async def test_people_without_detections_returns_empty_list() -> None:
    token = await upload_token()

    response = await editor_people_route(
        EditorPeopleRequest(imageToken=token), people=FakePeople([])
    )

    assert response.people == []


@pytest.mark.asyncio
async def test_people_unknown_token_is_400() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await editor_people_route(
            EditorPeopleRequest(imageToken="deadbeef"), people=FakePeople([])
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_people_invalid_device_is_400() -> None:
    token = await upload_token()

    with pytest.raises(HTTPException) as exc_info:
        await editor_people_route(
            EditorPeopleRequest(imageToken=token, device="cuda:0"), people=FakePeople([])
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_capabilities_reports_people() -> None:
    from app.schemas import EditorPerson

    capability = await editor_capabilities(
        segmenter=FakeSegmenter(),
        people=FakePeople([
            EditorPerson(id=0, box=[0, 0, 1, 1], score=0.9, role="subject", anchor=[0, 0])
        ]),
    )

    assert capability == {"tapSelect": True, "people": True}


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


@pytest.mark.asyncio
async def test_insert_object_replace_mode_fits_the_target_object_bbox() -> None:
    from app.api.editor_routes import insert_object
    from app.schemas import InsertObjectRequest
    import base64

    target_token = await _upload_png(Image.new("RGB", (32, 32), (0, 0, 255)), "target.png")
    source_token = await _upload_png(Image.new("RGB", (8, 8), (255, 0, 0)), "source.png")
    source_mask = Image.new("L", (8, 8), 255)
    source_mask_token = await _upload_png(source_mask, "mask.png")
    # Objeto del destino a reemplazar: cuadrado en (8,8)-(24,24).
    target_mask = Image.new("L", (32, 32), 0)
    target_mask.paste(255, (8, 8, 24, 24))
    target_mask_token = await _upload_png(target_mask, "target-mask.png")

    response = await insert_object(
        InsertObjectRequest(
            targetToken=target_token, sourceToken=source_token,
            sourceMaskToken=source_mask_token, targetMaskToken=target_mask_token,
            featherPx=0, matchColor=False, harmonize=False,
        ),
        request=None,
    )

    composite = Image.open(io.BytesIO(base64.b64decode(response.composite_png_base64)))
    # El objeto quedó DENTRO del bbox del objeto reemplazado (x/y/width/height
    # del request se ignoran en modo reemplazo)...
    assert composite.getpixel((16, 16))[0] > 200
    # ...y fuera del bbox el destino sigue intacto.
    assert composite.getpixel((2, 2))[2] > 200


# --- armonización fase 2: máscara continua ----------------------------------


class RecordingManager:
    """Manager falso con un modelo de inpainting DEDICADO instalado."""

    def __init__(self, models_path) -> None:
        from types import SimpleNamespace

        pipeline_dir = models_path / "gen--fake--inpaint"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        (pipeline_dir / "model_index.json").write_text(
            '{"_class_name": "StableDiffusionXLInpaintPipeline"}', encoding="utf-8"
        )
        entry = SimpleNamespace(file_path="gen--fake--inpaint", kind=None, source="hf:x/y")
        self.registry = SimpleNamespace(get=lambda model_id: entry)
        self.calls: list[dict] = []

    async def create_job(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        return SimpleNamespace(id="job-1")


def _harmonize_request(manager):
    from types import SimpleNamespace

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(generation_job_manager=manager)),
        # current_user_from_request lee esto cuando no hay auth multi-usuario
        state=SimpleNamespace(user=None),
        scope={"type": "http"},
    )


async def _run_harmonize(manager, **overrides):
    from app.api.editor_routes import insert_object
    from app.schemas import InsertObjectRequest

    target_token = await _upload_png(Image.new("RGB", (160, 160), (0, 0, 255)), "target.png")
    source_token = await _upload_png(Image.new("RGB", (80, 80), (255, 0, 0)), "source.png")
    mask_token = await _upload_png(Image.new("L", (80, 80), 255), "mask.png")

    payload = dict(
        targetToken=target_token, sourceToken=source_token, sourceMaskToken=mask_token,
        x=40, y=40, width=80, height=80, featherPx=0, matchColor=False,
        harmonize=True, modelId="gen--fake--inpaint",
    )
    payload.update(overrides)
    return await insert_object(InsertObjectRequest(**payload), request=_harmonize_request(manager))


@pytest.mark.asyncio
async def test_harmonize_defaults_to_the_continuous_mask_and_skips_engine_dilation() -> None:
    settings = get_settings()
    manager = RecordingManager(settings.models_path)

    response = await _run_harmonize(manager)

    assert response.job_id == "job-1"
    call = manager.calls[0]
    # el perfil continuo ya trae su dilatación y su caída: re-dilatarlo en el
    # motor empujaría la costura hacia adentro y borraría el centro preservado
    assert call["mask_dilate_px"] == 0
    assert call["mask_feather_px"] == 0
    mask = np.asarray(Image.open(call["mask_image_path"]).convert("L"))
    assert mask.max() == 255
    assert mask[80, 80] == 0  # centro del objeto: preservado


@pytest.mark.asyncio
async def test_harmonize_blend_one_keeps_the_uniform_mask_and_engine_defaults() -> None:
    settings = get_settings()
    manager = RecordingManager(settings.models_path)

    await _run_harmonize(manager, harmonizeBlend=1.0)

    call = manager.calls[0]
    # comportamiento viejo alcanzable: máscara uniforme y el motor prepara como siempre
    assert "mask_dilate_px" not in call and "mask_feather_px" not in call
    mask = np.asarray(Image.open(call["mask_image_path"]).convert("L"))
    assert (mask[40:120, 40:120] == 255).all()


@pytest.mark.asyncio
async def test_insert_object_request_rejects_a_blend_outside_zero_one() -> None:
    import pydantic

    from app.schemas import InsertObjectRequest

    with pytest.raises(pydantic.ValidationError):
        InsertObjectRequest(
            targetToken="a", sourceToken="b", sourceMaskToken="c", harmonizeBlend=1.4
        )


@pytest.mark.asyncio
async def test_insert_object_replace_mode_rejects_an_empty_target_mask() -> None:
    from app.api.editor_routes import insert_object
    from app.schemas import InsertObjectRequest

    target_token = await _upload_png(Image.new("RGB", (16, 16), (0, 0, 255)), "target.png")
    source_token = await _upload_png(Image.new("RGB", (8, 8), (255, 0, 0)), "source.png")
    source_mask_token = await _upload_png(Image.new("L", (8, 8), 255), "mask.png")
    empty_mask_token = await _upload_png(Image.new("L", (16, 16), 0), "empty.png")

    with pytest.raises(HTTPException) as excinfo:
        await insert_object(
            InsertObjectRequest(
                targetToken=target_token, sourceToken=source_token,
                sourceMaskToken=source_mask_token, targetMaskToken=empty_mask_token,
                harmonize=False,
            ),
            request=None,
        )
    assert excinfo.value.status_code == 400


class FakeLama:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[tuple] = []
        self.fill_thread_ids: list[int] = []

    def available(self) -> bool:
        return self._available

    def fill_group(self, canvas, group_mask, protect):
        self.fill_thread_ids.append(threading.get_ident())
        self.calls.append((group_mask.copy(), protect.copy()))
        result = canvas.copy()
        result[group_mask > 0] = (240, 20, 30)
        return result


@pytest.mark.asyncio
async def test_init_image_download_returns_the_uploaded_bytes() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 8, 7)).save(buffer, format="PNG")
    settings = get_settings()
    StorageService(settings).ensure_directories()
    uploaded = await upload_init_image(
        file=UploadFile(filename="photo.png", file=io.BytesIO(buffer.getvalue())),
        storage=StorageService(settings),
        settings=settings,
    )

    response = download_init_image(uploaded.init_image_token, settings=settings)

    assert Path(response.path).read_bytes() == buffer.getvalue()
    assert response.media_type == "image/png"


@pytest.mark.asyncio
async def test_init_image_download_serves_decoded_format_not_filename_suffix() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 8, 7)).save(buffer, format="PNG")
    settings = get_settings()
    StorageService(settings).ensure_directories()
    uploaded = await upload_init_image(
        file=UploadFile(filename="x.html", file=io.BytesIO(buffer.getvalue())),
        storage=StorageService(settings),
        settings=settings,
    )

    response = download_init_image(uploaded.init_image_token, settings=settings)

    assert response.media_type == "image/png"
    assert "text/html" not in response.headers["content-type"]


@pytest.mark.asyncio
async def test_init_image_download_sends_nosniff() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 8, 7)).save(buffer, format="PNG")
    settings = get_settings()
    StorageService(settings).ensure_directories()
    uploaded = await upload_init_image(
        file=UploadFile(filename="photo.png", file=io.BytesIO(buffer.getvalue())),
        storage=StorageService(settings),
        settings=settings,
    )

    response = download_init_image(uploaded.init_image_token, settings=settings)

    assert response.headers["x-content-type-options"] == "nosniff"


def test_init_image_download_rejects_bad_and_unknown_tokens() -> None:
    settings = get_settings()
    with pytest.raises(HTTPException) as bad:
        download_init_image("../etc", settings=settings)
    with pytest.raises(HTTPException) as unknown:
        download_init_image("ab" * 16, settings=settings)

    assert bad.value.status_code == 400
    assert unknown.value.status_code == 400


def _heal_request_context(lama) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(lama_eraser=lama)))


@pytest.mark.asyncio
async def test_heal_returns_new_token_with_filled_region() -> None:
    from app.api.editor_routes import heal
    from app.api.routes import _resolve_init_image
    from app.schemas import EditorHealRequest

    image = Image.new("RGB", (128, 128), (1, 2, 3))
    mask = Image.new("L", (128, 128), 0)
    mask.paste(255, (60, 60, 64, 64))
    image_token = await _upload_png(image, "base.photo.png")
    mask_token = await _upload_png(mask, "brush.png")
    lama = FakeLama()
    test_thread_id = threading.get_ident()

    response = await heal(
        EditorHealRequest(imageToken=image_token, maskToken=mask_token),
        request=_heal_request_context(lama),
    )

    assert response.init_image_token != image_token
    assert response.original_filename == "base.photo.png"
    output_path = _resolve_init_image(get_settings(), response.init_image_token)
    assert output_path is not None
    assert output_path.name.endswith("-base.photo.png")
    with Image.open(output_path) as output:
        result = output.convert("RGB")
    assert result.size == image.size
    assert result.getpixel((61, 61)) == (240, 20, 30)
    assert result.getpixel((10, 10)) == (1, 2, 3)
    assert lama.fill_thread_ids[0] != test_thread_id
    group_mask, protect = lama.calls[0]
    assert group_mask.dtype == np.uint8
    assert np.count_nonzero(group_mask) > 16
    assert protect.dtype == np.bool_
    assert not protect.any()


@pytest.mark.asyncio
async def test_heal_rejects_large_stroke() -> None:
    from app.api.editor_routes import heal
    from app.schemas import EditorHealRequest

    image_token = await _upload_png(Image.new("RGB", (720, 8), (1, 2, 3)), "wide.png")
    mask = Image.new("L", (720, 8), 0)
    mask.paste(255, (10, 1, 711, 7))
    mask_token = await _upload_png(mask, "brush.png")

    with pytest.raises(HTTPException) as exc_info:
        await heal(
            EditorHealRequest(imageToken=image_token, maskToken=mask_token),
            request=_heal_request_context(FakeLama()),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "La zona es grande: usá Quitar."


@pytest.mark.asyncio
async def test_heal_rejects_empty_mask() -> None:
    from app.api.editor_routes import heal
    from app.schemas import EditorHealRequest

    image_token = await _upload_png(Image.new("RGB", (16, 12), (1, 2, 3)), "base.png")
    mask_token = await _upload_png(Image.new("L", (16, 12), 0), "empty-mask.png")

    with pytest.raises(HTTPException) as exc_info:
        await heal(
            EditorHealRequest(imageToken=image_token, maskToken=mask_token),
            request=_heal_request_context(FakeLama()),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "La máscara está vacía."


@pytest.mark.asyncio
async def test_heal_without_lama_is_503() -> None:
    from app.api.editor_routes import heal
    from app.schemas import EditorHealRequest

    image_token = await _upload_png(Image.new("RGB", (16, 12), (1, 2, 3)), "base.png")
    mask = Image.new("L", (16, 12), 0)
    mask.paste(255, (4, 3, 8, 7))
    mask_token = await _upload_png(mask, "brush.png")

    with pytest.raises(HTTPException) as exc_info:
        await heal(
            EditorHealRequest(imageToken=image_token, maskToken=mask_token),
            request=_heal_request_context(FakeLama(available=False)),
        )

    assert exc_info.value.status_code == 503
