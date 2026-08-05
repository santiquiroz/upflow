from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.tts_kokoro import (
    KokoroTtsEngine,
    TtsUnavailable,
    assert_usable_audio,
    available_voices,
    load_vocab,
    load_voice,
)

# ---------------------------------------------------------------------------
# El fallo mas caro de este modelo NO es una excepcion: la variante fp16 carga,
# corre, devuelve audio de la duracion correcta y ese audio es todo NaN (medido
# 2026-08-04). Nada avisa. Por eso el motor revisa el audio antes de entregarlo.
# ---------------------------------------------------------------------------


def make_model_dir(tmp_path: Path, *, with_model: bool = True) -> Path:
    model_dir = tmp_path / "kokoro"
    (model_dir / "voices").mkdir(parents=True)
    (model_dir / "tokenizer.json").write_text(
        json.dumps({"model": {"vocab": {"a": 10, "ə": 11}}}), encoding="utf-8"
    )
    if with_model:
        (model_dir / "model.onnx").write_bytes(b"no-es-un-modelo-real")
    np.zeros((20, 256), dtype=np.float32).tofile(model_dir / "voices" / "af_heart.bin")
    return model_dir


class TestAudioGuard:
    def test_nan_audio_is_refused_instead_of_delivered(self) -> None:
        with pytest.raises(TtsUnavailable, match="NaN"):
            assert_usable_audio(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    def test_infinite_audio_is_refused_too(self) -> None:
        with pytest.raises(TtsUnavailable):
            assert_usable_audio(np.array([np.inf], dtype=np.float32))

    def test_empty_audio_is_refused(self) -> None:
        with pytest.raises(TtsUnavailable, match="vac"):
            assert_usable_audio(np.array([], dtype=np.float32))

    def test_real_audio_passes(self) -> None:
        assert_usable_audio(np.array([0.1, -0.2, 0.3], dtype=np.float32))


class TestModelDirectory:
    def test_reads_the_vocabulary_the_model_ships_with(self, tmp_path: Path) -> None:
        assert load_vocab(make_model_dir(tmp_path)) == {"a": 10, "ə": 11}

    def test_a_missing_tokenizer_says_which_file_is_missing(self, tmp_path: Path) -> None:
        (tmp_path / "vacio").mkdir()
        with pytest.raises(TtsUnavailable, match="tokenizer.json"):
            load_vocab(tmp_path / "vacio")

    def test_lists_the_installed_voices(self, tmp_path: Path) -> None:
        assert available_voices(make_model_dir(tmp_path)) == ["af_heart"]

    def test_no_voices_directory_gives_an_empty_list_not_a_crash(self, tmp_path: Path) -> None:
        assert available_voices(tmp_path / "no-existe") == []

    def test_a_voice_loads_with_one_row_per_length(self, tmp_path: Path) -> None:
        voices = load_voice(make_model_dir(tmp_path), "af_heart")
        assert voices.shape == (20, 256)

    def test_an_unknown_voice_names_itself_in_the_error(self, tmp_path: Path) -> None:
        with pytest.raises(TtsUnavailable, match="no-existe"):
            load_voice(make_model_dir(tmp_path), "no-existe")


class TestAvailability:
    def test_reports_unavailable_without_the_model_file(self, tmp_path: Path) -> None:
        engine = KokoroTtsEngine(Settings(_env_file=None, RUNTIME_DIR=str(tmp_path)))
        assert engine.available(make_model_dir(tmp_path, with_model=False)) is False

    def test_reports_available_once_the_model_is_there(self, tmp_path: Path) -> None:
        engine = KokoroTtsEngine(Settings(_env_file=None, RUNTIME_DIR=str(tmp_path)))
        assert engine.available(make_model_dir(tmp_path)) is True


class FakeSession:
    """Devuelve audio de largo proporcional a 1/velocidad, como el modelo real:
    pedir mas velocidad da menos muestras."""

    def __init__(self) -> None:
        self.last_speed: float | None = None

    def run(self, _outputs, feeds):
        self.last_speed = float(feeds["speed"][0])
        return [np.ones(int(1000 / self.last_speed), dtype=np.float32)]


class TestSynthesisSpeed:
    def make_engine(self, tmp_path: Path) -> tuple[KokoroTtsEngine, Path, FakeSession]:
        model_dir = make_model_dir(tmp_path)
        engine = KokoroTtsEngine(Settings(RUNTIME_DIR=str(tmp_path), _env_file=None))
        session = FakeSession()
        engine._cache[str(model_dir)] = (session, load_vocab(model_dir))
        return engine, model_dir, session

    def test_the_default_speed_is_normal(self, tmp_path: Path) -> None:
        engine, model_dir, session = self.make_engine(tmp_path)

        engine.synthesize(model_dir=model_dir, phonemes="a", voice="af_heart")

        assert session.last_speed == 1.0

    def test_the_requested_speed_reaches_the_model(self, tmp_path: Path) -> None:
        # El doblaje sintetiza a la velocidad que hace falta para entrar en el
        # hueco del original: sin esto cada linea sale a destiempo.
        engine, model_dir, session = self.make_engine(tmp_path)

        audio = engine.synthesize(
            model_dir=model_dir, phonemes="a", voice="af_heart", speed=1.5
        )

        assert session.last_speed == 1.5
        assert len(audio) < 1000

    def test_an_impossible_speed_is_refused_instead_of_making_noise(self, tmp_path: Path) -> None:
        engine, model_dir, _session = self.make_engine(tmp_path)

        with pytest.raises(ValueError):
            engine.synthesize(model_dir=model_dir, phonemes="a", voice="af_heart", speed=0.0)
