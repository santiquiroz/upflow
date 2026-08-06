from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.routes import synthesize_speech, tts_capabilities
from app.config import Settings
from app.schemas import SynthesizeSpeechRequest

# ---------------------------------------------------------------------------
# La sintesis devuelve el WAV directo y no crea un job: 1,90 s de audio salen en
# menos de medio segundo (medido), asi que encolarlo seria mas caro que hacerlo.
# ---------------------------------------------------------------------------


class FakeEngine:
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.calls: list[dict] = []

    def available(self, model_dir: Path) -> bool:
        return self._available

    def synthesize(self, *, model_dir: Path, phonemes: str, voice: str):
        self.calls.append({"phonemes": phonemes, "voice": voice})
        return np.linspace(-0.4, 0.4, 24000, dtype=np.float32)


def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path))


@pytest.mark.asyncio
async def test_capabilities_report_the_installed_voices(tmp_path: Path):
    model_dir = tmp_path / "kokoro" / "voices"
    model_dir.mkdir(parents=True)
    (model_dir / "af_heart.bin").write_bytes(b"")
    (model_dir.parent / "model.onnx").write_bytes(b"")
    (model_dir.parent / "tokenizer.json").write_text('{"model":{"vocab":{}}}', encoding="utf-8")

    response = await tts_capabilities(
        engine=FakeEngine(), settings_dep=settings(tmp_path), model_dir=model_dir.parent
    )

    assert response.available is True
    assert response.voices == ["af_heart"]


@pytest.mark.asyncio
async def test_capabilities_name_the_missing_pack_instead_of_a_command(tmp_path: Path):
    response = await tts_capabilities(
        engine=FakeEngine(available=False),
        settings_dep=settings(tmp_path),
        model_dir=tmp_path / "no-esta",
    )

    assert response.available is False
    assert response.missing_pack == "kokoro"
    # La regresion que importa: este test antes EXIGIA que el mensaje nombrara
    # el script, o sea que codificaba como contrato justo lo que sacamos.
    assert ".ps1" not in (response.reason or "")


@pytest.mark.asyncio
async def test_synthesis_returns_a_wav(tmp_path: Path):
    engine = FakeEngine()
    response = await synthesize_speech(
        SynthesizeSpeechRequest(text="hello world", voice="af_heart", language="en"),
        engine=engine,
        settings_dep=settings(tmp_path),
        model_dir=tmp_path,
    )

    assert response.media_type == "audio/wav"
    assert response.body[:4] == b"RIFF"
    # Lo que llega al motor son FONEMAS, no el texto crudo.
    assert engine.calls[0]["phonemes"] != "hello world"
    assert engine.calls[0]["phonemes"]


@pytest.mark.asyncio
async def test_empty_text_is_refused_before_touching_the_model(tmp_path: Path):
    engine = FakeEngine()
    with pytest.raises(HTTPException) as excinfo:
        await synthesize_speech(
            SynthesizeSpeechRequest(text="   ", voice="af_heart", language="en"),
            engine=engine,
            settings_dep=settings(tmp_path),
            model_dir=tmp_path,
        )

    assert excinfo.value.status_code == 400
    assert engine.calls == []


@pytest.mark.asyncio
async def test_an_unavailable_engine_is_a_clear_409(tmp_path: Path):
    with pytest.raises(HTTPException) as excinfo:
        await synthesize_speech(
            SynthesizeSpeechRequest(text="hola", voice="af_heart", language="es"),
            engine=FakeEngine(available=False),
            settings_dep=settings(tmp_path),
            model_dir=tmp_path,
        )

    assert excinfo.value.status_code == 409
