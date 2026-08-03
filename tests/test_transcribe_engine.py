from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.services.engines.transcribe_onnx import (
    CHUNK_SAMPLES,
    CHUNK_SECONDS,
    MAX_TOKENS_PER_CHUNK,
    TARGET_SAMPLE_RATE,
    TranscribeEngine,
    TranscribeRequest,
    build_load_kwargs,
    load_audio_mono_16k,
    split_into_chunks,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


def write_wav(
    path: Path, seconds: float, sample_rate: int = TARGET_SAMPLE_RATE, channels: int = 1
) -> Path:
    samples = int(sample_rate * seconds)
    t = np.linspace(0, seconds, samples, endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    data = tone if channels == 1 else np.stack([tone] * channels, axis=1)
    sf.write(str(path), data, sample_rate)
    return path


class FakeProcessor:
    """Imita al procesador de whisper en lo que importa: recibe un trozo y su
    resultado se decodifica a texto. Registra los largos que vio."""

    def __init__(self) -> None:
        self.chunk_lengths: list[int] = []
        self.generate_kwargs: list[dict[str, Any]] = []

    def __call__(self, chunk, sampling_rate: int, return_tensors: str):
        self.chunk_lengths.append(len(chunk))
        assert sampling_rate == TARGET_SAMPLE_RATE
        return type("Features", (), {"input_features": f"features-{len(self.chunk_lengths)}"})()

    def batch_decode(self, tokens, skip_special_tokens: bool):
        return [tokens]


class FakeModel:
    def __init__(self, processor: FakeProcessor, texts: list[str] | None = None) -> None:
        self.processor = processor
        self.texts = texts
        self.calls = 0

    def generate(self, input_features, **kwargs):
        self.processor.generate_kwargs.append(kwargs)
        self.calls += 1
        if self.texts is not None:
            return self.texts[self.calls - 1]
        return f"trozo{self.calls}"


class RecordingCoordinator:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, Any]] = []

    def acquire(self, device: str, owner: Any) -> None:
        self.acquired.append((device, owner))


def make_engine(
    tmp_path: Path, texts: list[str] | None = None
) -> tuple[TranscribeEngine, FakeModel, FakeProcessor, RecordingCoordinator]:
    coordinator = RecordingCoordinator()
    engine = TranscribeEngine(make_settings(tmp_path), coordinator)
    processor = FakeProcessor()
    model = FakeModel(processor, texts)
    engine._create_model = lambda model_dir, device: (model, processor)  # type: ignore[method-assign]
    return engine, model, processor, coordinator


async def transcribe(
    engine: TranscribeEngine, tmp_path: Path, audio: Path, language: str | None = None
) -> str:
    return await engine.run(
        model_id="asr--whisper-tiny",
        model_dir=tmp_path,
        audio_path=audio,
        request=TranscribeRequest(language=language),
        device="cpu",
        progress_cb=lambda _done, _total: None,
    )


# ---------------------------------------------------------------------------
# Carga de audio
# ---------------------------------------------------------------------------


def test_stereo_is_mixed_down_to_mono(tmp_path: Path):
    audio = load_audio_mono_16k(write_wav(tmp_path / "s.wav", 1.0, channels=2))
    assert audio.ndim == 1


def test_a_different_sample_rate_is_resampled(tmp_path: Path):
    # Whisper asume 16 kHz y no remuestrea por su cuenta.
    audio = load_audio_mono_16k(write_wav(tmp_path / "s.wav", 2.0, sample_rate=44100))
    assert abs(len(audio) - 2 * TARGET_SAMPLE_RATE) < TARGET_SAMPLE_RATE // 10


def test_audio_already_at_16k_is_left_alone(tmp_path: Path):
    audio = load_audio_mono_16k(write_wav(tmp_path / "s.wav", 1.0))
    assert len(audio) == TARGET_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Troceo -- la parte que evita perder audio en silencio
# ---------------------------------------------------------------------------


def test_audio_shorter_than_a_chunk_is_one_chunk():
    assert len(split_into_chunks(np.zeros(TARGET_SAMPLE_RATE * 5, dtype="float32"))) == 1


def test_audio_exactly_one_chunk_long_is_one_chunk():
    assert len(split_into_chunks(np.zeros(CHUNK_SAMPLES, dtype="float32"))) == 1


def test_long_audio_is_split_and_nothing_is_dropped():
    """El extractor RELLENA O TRUNCA todo a 30 segundos: medido, 3 s / 30 s / 120 s
    producen el mismo tensor. Sin trocear, dos minutos de audio pierden noventa
    segundos sin que nada avise."""
    total = CHUNK_SAMPLES * 4 + 1234
    chunks = split_into_chunks(np.arange(total, dtype="float32"))

    assert len(chunks) == 5
    assert sum(len(chunk) for chunk in chunks) == total
    # Y en el orden correcto, sin huecos ni solapes.
    assert np.array_equal(np.concatenate(chunks), np.arange(total, dtype="float32"))


def test_empty_audio_is_no_chunks():
    assert split_into_chunks(np.zeros(0, dtype="float32")) == []


def test_the_chunk_size_matches_what_the_extractor_expects():
    assert CHUNK_SAMPLES == TARGET_SAMPLE_RATE * CHUNK_SECONDS
    assert CHUNK_SECONDS == 30


# ---------------------------------------------------------------------------
# El motor
# ---------------------------------------------------------------------------


async def test_a_short_audio_runs_a_single_pass(tmp_path: Path):
    engine, model, processor, _c = make_engine(tmp_path, texts=["hola"])
    text = await transcribe(engine, tmp_path, write_wav(tmp_path / "a.wav", 4.0))

    assert model.calls == 1
    assert text == "hola"
    assert processor.chunk_lengths == [4 * TARGET_SAMPLE_RATE]


async def test_a_long_audio_transcribes_every_chunk_and_joins_them(tmp_path: Path):
    engine, model, _p, _c = make_engine(tmp_path, texts=["uno", "dos", "tres"])
    audio = write_wav(tmp_path / "a.wav", CHUNK_SECONDS * 2 + 5)

    text = await transcribe(engine, tmp_path, audio)

    assert model.calls == 3
    assert text == "uno dos tres"


async def test_an_empty_chunk_does_not_leave_double_spaces(tmp_path: Path):
    engine, _m, _p, _c = make_engine(tmp_path, texts=["uno", "", "tres"])
    audio = write_wav(tmp_path / "a.wav", CHUNK_SECONDS * 2 + 5)

    assert await transcribe(engine, tmp_path, audio) == "uno tres"


async def test_progress_is_reported_per_chunk(tmp_path: Path):
    engine, _m, _p, _c = make_engine(tmp_path, texts=["a", "b", "c"])
    seen: list[tuple[int, int]] = []

    await engine.run(
        model_id="m",
        model_dir=tmp_path,
        audio_path=write_wav(tmp_path / "a.wav", CHUNK_SECONDS * 2 + 5),
        request=TranscribeRequest(),
        device="cpu",
        progress_cb=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


async def test_silence_returns_empty_text_without_crashing(tmp_path: Path):
    engine, _m, _p, _c = make_engine(tmp_path, texts=[""])
    assert await transcribe(engine, tmp_path, write_wav(tmp_path / "a.wav", 1.0)) == ""


async def test_a_requested_language_reaches_generate(tmp_path: Path):
    engine, _m, processor, _c = make_engine(tmp_path, texts=["hola"])
    await transcribe(engine, tmp_path, write_wav(tmp_path / "a.wav", 2.0), language="es")

    assert processor.generate_kwargs[0]["language"] == "es"


async def test_no_language_is_not_passed_at_all(tmp_path: Path):
    # Pasar None dejaria que el modelo lo trate como un idioma llamado "None"; el
    # campo ausente es lo que le dice "detectalo vos".
    engine, _m, processor, _c = make_engine(tmp_path, texts=["hola"])
    await transcribe(engine, tmp_path, write_wav(tmp_path / "a.wav", 2.0))

    assert "language" not in processor.generate_kwargs[0]


async def test_every_chunk_gets_the_token_ceiling(tmp_path: Path):
    # Sin techo, un modelo que entra en bucle corre para siempre. Y entrar en bucle
    # es exactamente lo que hace whisper sobre entrada sin habla.
    engine, _m, processor, _c = make_engine(tmp_path, texts=["a", "b"])
    await transcribe(engine, tmp_path, write_wav(tmp_path / "a.wav", CHUNK_SECONDS + 2))

    for kwargs in processor.generate_kwargs:
        assert kwargs["max_new_tokens"] == MAX_TOKENS_PER_CHUNK


async def test_the_device_is_acquired_from_the_coordinator(tmp_path: Path):
    engine, _m, _p, coordinator = make_engine(tmp_path, texts=["hola"])
    await transcribe(engine, tmp_path, write_wav(tmp_path / "a.wav", 1.0))

    assert coordinator.acquired == [("cpu", engine)]


async def test_the_model_is_loaded_once_and_reused(tmp_path: Path):
    coordinator = RecordingCoordinator()
    engine = TranscribeEngine(make_settings(tmp_path), coordinator)
    processor = FakeProcessor()
    creations: list[str] = []

    def create(model_dir: Path, device: str):
        creations.append(device)
        return FakeModel(processor, ["hola"] * 10), processor

    engine._create_model = create  # type: ignore[method-assign]
    audio = write_wav(tmp_path / "a.wav", 1.0)

    for _ in range(3):
        await transcribe(engine, tmp_path, audio)

    assert creations == ["cpu"]


async def test_releasing_a_device_drops_its_cached_model(tmp_path: Path):
    coordinator = RecordingCoordinator()
    engine = TranscribeEngine(make_settings(tmp_path), coordinator)
    processor = FakeProcessor()
    creations: list[str] = []

    def create(model_dir: Path, device: str):
        creations.append(device)
        return FakeModel(processor, ["hola"] * 10), processor

    engine._create_model = create  # type: ignore[method-assign]
    audio = write_wav(tmp_path / "a.wav", 1.0)

    await transcribe(engine, tmp_path, audio)
    engine.release_device("cpu")
    await transcribe(engine, tmp_path, audio)

    assert creations == ["cpu", "cpu"]


# ---------------------------------------------------------------------------
# La bandera que decide si transcribe o devuelve ruido
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "dml:0"])
def test_the_model_is_never_loaded_with_the_merged_decoder(device: str):
    """Con el decoder merged, que es el DEFAULT, DirectML carga, corre sin excepcion
    y devuelve BASURA: medido 2026-07-29, "I know Kung Fu." se convierte en
    "Ioti plur plur plur fighting...".

    Los demas tests usan un doble de _create_model, asi que ninguno tocaria esta
    bandera. Por eso los kwargs se armaron en una funcion aparte: se verifican sin
    mockear optimum ni transformers.
    """
    kwargs = build_load_kwargs(device)

    assert kwargs["use_merged"] is False
    # IO binding con DirectML esta vetado en este repo.
    assert kwargs["use_io_binding"] is False
    assert "provider" in kwargs


def test_the_gpu_device_selects_a_gpu_provider():
    assert "Dml" in build_load_kwargs("dml:0")["provider"]


def test_the_cpu_device_selects_the_cpu_provider():
    assert build_load_kwargs("cpu")["provider"] == "CPUExecutionProvider"


# --- el acelerador nativo tambien tiene que llegar a la transcripcion ------


def test_transcription_uses_the_native_ep_when_there_is_one(monkeypatch, tmp_path):
    """Antes armaba providers DirectML a mano y se saltaba el ep_registry: con
    un plugin instalado, todo aceleraba menos transcribir y generar."""
    from app.config import Settings
    from app.services import ep_registry

    llamadas = []

    def fake_loader_kwargs(device, settings, **kw):
        llamadas.append(device)
        return {"providers": [], "provider_options": [], "session_options": object()}

    monkeypatch.setattr(ep_registry, "loader_kwargs", fake_loader_kwargs)

    kwargs = build_load_kwargs("dml:0", Settings(_env_file=None))

    assert llamadas == ["dml:0"]
    assert kwargs["providers"] == []
    # Las dos banderas medidas siguen puestas: sin use_merged=False Whisper
    # devuelve basura con DirectML.
    assert kwargs["use_merged"] is False
    assert kwargs["use_io_binding"] is False
