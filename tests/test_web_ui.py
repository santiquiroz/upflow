from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def get_index_html() -> str:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        return response.text


def test_apply_video_profile_reads_snake_case_catalog_fields() -> None:
    html = get_index_html()

    assert "profile.model_key" in html
    assert "profile.video_codec" in html
    assert "profile.video_preset" in html


def test_apply_video_profile_does_not_reference_camel_case_fields() -> None:
    html = get_index_html()

    assert "profile.modelKey" not in html
    assert "profile.videoCodec" not in html
    assert "profile.videoPreset" not in html


def test_output_fps_display_is_normalized_through_a_formatter() -> None:
    html = get_index_html()

    assert "formatFps(data.metadata.outputFps)" in html
