from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import AUDIO_ENHANCE_MODES
from app.main import app
from app.services.engines.audio_enhance import AudioEnhancer

# ---------------------------------------------------------------------------
# Task 20 (6.1c) - main.py lifespan builds an AudioEnhancer per mode and
# injects them into VideoUpscaler (same DI pattern as rife_engine).
# ---------------------------------------------------------------------------


def test_lifespan_wires_audio_enhancers_into_video_upscaler() -> None:
    with TestClient(app):
        upscaler = app.state.video_job_manager.upscaler

        assert set(upscaler.audio_enhancers.keys()) == set(AUDIO_ENHANCE_MODES)
        for mode, enhancer in upscaler.audio_enhancers.items():
            assert isinstance(enhancer, AudioEnhancer)
            assert enhancer.mode == mode
