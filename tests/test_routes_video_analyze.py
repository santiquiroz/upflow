from __future__ import annotations

import io
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services.media_tools import MediaTools

TINY_VIDEO_BYTES = b"fake mp4 bytes for upload staging, not decoded by this test"


def _uploaded_files() -> list:
    return list(get_settings().uploads_path.glob("*"))


def test_analyze_video_returns_audio_and_subtitle_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ffprobe_json(self, path):
        return {
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264"},
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "disposition": {"default": 1},
                    "tags": {"language": "jpn"},
                },
            ]
        }

    monkeypatch.setattr(MediaTools, "ffprobe_json", fake_ffprobe_json)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/video/analyze",
            files={"file": ("clip.mp4", io.BytesIO(TINY_VIDEO_BYTES), "video/mp4")},
        )

    assert response.status_code == 200
    body = response.json()
    assert "uploadToken" in body
    assert body["audioTracks"] == [
        {"index": 1, "codec": "aac", "channels": 2, "isDefault": True, "language": "jpn"}
    ]
    assert body["subtitleTracks"] == []


def test_analyze_video_rejects_non_video_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ffprobe_json_raises(self, path):
        raise subprocess.CalledProcessError(1, ["ffprobe"])

    monkeypatch.setattr(MediaTools, "ffprobe_json", fake_ffprobe_json_raises)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/video/analyze",
            files={"file": ("not-a-video.txt", io.BytesIO(b"hello"), "text/plain")},
        )

    assert response.status_code == 400
    assert _uploaded_files() == []


def test_analyze_video_returns_500_and_cleans_up_when_ffprobe_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ffprobe_json_raises(self, path):
        raise RuntimeError("FFmpeg/FFprobe not available. Run scripts/download-ffmpeg.ps1 first.")

    monkeypatch.setattr(MediaTools, "ffprobe_json", fake_ffprobe_json_raises)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/video/analyze",
            files={"file": ("clip.mp4", io.BytesIO(TINY_VIDEO_BYTES), "video/mp4")},
        )

    assert response.status_code == 500
    assert _uploaded_files() == []


def test_analyze_video_returns_500_and_cleans_up_on_unexpected_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ffprobe_json_raises(self, path):
        raise ValueError("malformed ffprobe output")

    monkeypatch.setattr(MediaTools, "ffprobe_json", fake_ffprobe_json_raises)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/video/analyze",
            files={"file": ("clip.mp4", io.BytesIO(TINY_VIDEO_BYTES), "video/mp4")},
        )

    assert response.status_code == 500
    assert _uploaded_files() == []


def test_analyze_video_rejects_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a file that exceeds max_video_upload_mb (default 5000 MB); this will
    # cause save_upload to raise ValueError. We use BytesIO with seek to avoid memory bloat.
    oversized = io.BytesIO(b"x" * (100 * 1024 * 1024 + 1))  # 100 MB + 1 byte

    settings = get_settings()
    original_max = settings.max_video_upload_mb
    try:
        # Temporarily set a small limit to trigger the size check without needing 5GB
        settings.max_video_upload_mb = 0.001  # ~1 KB
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/video/analyze",
                files={"file": ("huge.mp4", oversized, "video/mp4")},
            )

        assert response.status_code == 400
        assert "exceeds" in response.json()["detail"].lower() or "size" in response.json()["detail"].lower()
        assert _uploaded_files() == []
    finally:
        settings.max_video_upload_mb = original_max
