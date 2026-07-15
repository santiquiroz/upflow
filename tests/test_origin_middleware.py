from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.security import is_origin_allowed
from app.services import devices_service

ALLOWED_ORIGINS = frozenset({"http://127.0.0.1:8090", "http://localhost:8090"})
FOREIGN_ORIGIN = "http://evil.example.com"


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def patch_fake_gpu_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes DevicesService resolve a fake DirectML GPU as the default device.

    Origin-middleware tests exercise the real /api/v1/jobs route end to end,
    which now resolves a default device via real hardware enumeration. On a
    machine with no DirectML GPU that default would fall back to "cpu",
    which is correctly rejected for the builtin ncnn model used here -- these
    tests care about origin handling, not device selection, so hardware
    detection is faked the same way tests/test_devices.py does.
    """
    monkeypatch.setattr(devices_service, "_probe_onnxruntime", lambda: devices_service.OnnxRuntimeProbe(
        available_providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    ))
    monkeypatch.setattr(devices_service, "_enumerate_gpu_adapter_names", lambda: ["Fake GPU"])


# ---------------------------------------------------------------------------
# 3.6 — pure decision function
# ---------------------------------------------------------------------------


def test_get_request_with_foreign_origin_is_allowed() -> None:
    assert is_origin_allowed("GET", FOREIGN_ORIGIN, None, ALLOWED_ORIGINS) is True


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_state_changing_request_with_no_origin_and_no_referer_is_allowed(method: str) -> None:
    assert is_origin_allowed(method, None, None, ALLOWED_ORIGINS) is True


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_state_changing_request_with_foreign_origin_is_rejected(method: str) -> None:
    assert is_origin_allowed(method, FOREIGN_ORIGIN, None, ALLOWED_ORIGINS) is False


def test_post_with_allowed_origin_is_allowed() -> None:
    assert is_origin_allowed("POST", "http://127.0.0.1:8090", None, ALLOWED_ORIGINS) is True


def test_post_with_allowed_referer_and_no_origin_is_allowed() -> None:
    assert is_origin_allowed("POST", None, "http://localhost:8090/", ALLOWED_ORIGINS) is True


def test_post_with_foreign_referer_and_no_origin_is_rejected() -> None:
    assert is_origin_allowed("POST", None, f"{FOREIGN_ORIGIN}/some/page", ALLOWED_ORIGINS) is False


def test_origin_header_takes_precedence_over_referer() -> None:
    assert (
        is_origin_allowed("POST", "http://127.0.0.1:8090", f"{FOREIGN_ORIGIN}/page", ALLOWED_ORIGINS)
        is True
    )


# ---------------------------------------------------------------------------
# 3.6 — wired into the app
# ---------------------------------------------------------------------------


def test_get_health_with_foreign_origin_passes() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health", headers={"Origin": FOREIGN_ORIGIN})

    assert response.status_code == 200


def test_post_job_with_no_origin_and_no_referer_passes_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fake_gpu_present(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/jobs",
            files={"file": ("photo.png", make_png_bytes(), "image/png")},
            data={"model_name": "realesrgan-x4plus", "scale": "4", "output_format": "png"},
        )

    assert response.status_code == 202


def test_post_job_with_foreign_origin_is_rejected_with_403() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/jobs",
            headers={"Origin": FOREIGN_ORIGIN},
            files={"file": ("photo.png", make_png_bytes(), "image/png")},
            data={"model_name": "realesrgan-x4plus", "scale": "4", "output_format": "png"},
        )

    assert response.status_code == 403


def test_post_job_with_allowed_origin_passes_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fake_gpu_present(monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/jobs",
            headers={"Origin": "http://127.0.0.1:8090"},
            files={"file": ("photo.png", make_png_bytes(), "image/png")},
            data={"model_name": "realesrgan-x4plus", "scale": "4", "output_format": "png"},
        )

    assert response.status_code == 202
