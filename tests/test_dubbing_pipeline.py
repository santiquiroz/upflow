from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.dubbing_pipeline import DubbingPipeline, DubbingUnavailable
from app.services.subtitles import TranscriptSegment

# ---------------------------------------------------------------------------
# El doblaje encadena piezas que ya existen: transcribir, traducir, sintetizar y
# muxear. Lo que se prueba aca es el ENCADENADO — que cada linea se sintetice a
# la velocidad que la hace entrar en su hueco, que el orden se conserve, y que
# lo que no entro se cuente en vez de pasar callado.
#
# Los motores reales no se tocan: son colaboradores explicitos y aca entran
# dobles. Probar el encadenado no necesita descargar un modelo.
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000


class FakeTranslation:
    def __init__(self) -> None:
        self.pedidos: list[list[str]] = []

    def translate(self, texts: list[str], pair) -> list[str]:
        # El motor real conserva los vacios en su lugar: el llamador empareja
        # por indice y descartarlos correria todos los tiempos.
        self.pedidos.append(list(texts))
        return [f"{t} traducido" if t.strip() else t for t in texts]


class FakeTts:
    """Sintetiza un segundo de audio dividido por la velocidad, como el modelo
    real: mas velocidad, menos muestras."""

    def __init__(self, natural_seconds: float = 1.0) -> None:
        self.natural_seconds = natural_seconds
        self.velocidades: list[float] = []
        self.voces: list[str] = []

    def available(self, _model_dir: Path) -> bool:
        return True

    def synthesize(self, *, model_dir: Path, phonemes: str, voice: str, speed: float = 1.0):
        self.velocidades.append(speed)
        self.voces.append(voice)
        muestras = int(self.natural_seconds * SAMPLE_RATE / speed)
        return np.full(muestras, 0.5, dtype=np.float32)


def make_pipeline(**kwargs) -> DubbingPipeline:
    argumentos = {
        "translation": FakeTranslation(),
        "tts": FakeTts(),
        "tts_model_dir": Path("modelo"),
        "phonemize": lambda texto, idioma: "aaa",
        "sample_rate": SAMPLE_RATE,
    }
    argumentos.update(kwargs)
    return DubbingPipeline(**argumentos)  # type: ignore[arg-type]


SEGMENTS = [
    TranscriptSegment(start=0.0, end=2.0, text="hello"),
    TranscriptSegment(start=3.0, end=5.0, text="world"),
]


class TestBuildTrack:
    def test_every_line_is_translated_before_being_spoken(self) -> None:
        translation = FakeTranslation()
        pipeline = make_pipeline(translation=translation)

        pipeline.build_track(SEGMENTS, source_language="en", target_language="es", total_seconds=5.0)

        assert translation.pedidos == [["hello", "world"]]

    def test_a_line_that_fits_is_spoken_at_normal_speed(self) -> None:
        # Un segundo de voz en un hueco de dos: no hay que apurar nada.
        tts = FakeTts(natural_seconds=1.0)
        pipeline = make_pipeline(tts=tts)

        pipeline.build_track(SEGMENTS, source_language="en", target_language="es", total_seconds=5.0)

        # Primera pasada a 1.0 para medir; como entra, no hay segunda pasada.
        assert tts.velocidades == [1.0, 1.0]

    def test_a_line_that_does_not_fit_is_spoken_again_faster(self) -> None:
        # Cuatro segundos de voz en un hueco de dos: 2x, que el tope baja a 1,6.
        tts = FakeTts(natural_seconds=4.0)
        pipeline = make_pipeline(tts=tts)

        pipeline.build_track(SEGMENTS, source_language="en", target_language="es", total_seconds=5.0)

        assert tts.velocidades[0] == 1.0
        assert tts.velocidades[1] == pytest.approx(1.6)

    def test_the_track_puts_each_line_at_its_own_second(self) -> None:
        pipeline = make_pipeline(tts=FakeTts(natural_seconds=0.5))

        resultado = pipeline.build_track(
            SEGMENTS, source_language="en", target_language="es", total_seconds=5.0
        )

        # Silencio entre el final de la primera (0,5 s) y el arranque de la
        # segunda (3 s).
        assert resultado.track[int(1.5 * SAMPLE_RATE)] == pytest.approx(0.0)
        assert resultado.track[int(3.1 * SAMPLE_RATE)] != 0.0

    def test_lines_that_never_fit_are_counted_not_hidden(self) -> None:
        # Cuatro segundos de voz en huecos de dos: ni al tope de velocidad entra.
        pipeline = make_pipeline(tts=FakeTts(natural_seconds=4.0))

        resultado = pipeline.build_track(
            SEGMENTS, source_language="en", target_language="es", total_seconds=5.0
        )

        assert resultado.overflowing == 2

    def test_an_empty_line_produces_silence_and_no_synthesis(self) -> None:
        tts = FakeTts()
        pipeline = make_pipeline(tts=tts)

        pipeline.build_track(
            [TranscriptSegment(start=0.0, end=1.0, text="   ")],
            source_language="en",
            target_language="es",
            total_seconds=1.0,
        )

        assert tts.velocidades == []

    def test_text_that_cannot_be_phonemized_is_skipped_instead_of_killing_the_job(self) -> None:
        # Una linea rara no puede tirar abajo el doblaje entero del video.
        tts = FakeTts()
        pipeline = make_pipeline(tts=tts, phonemize=lambda texto, idioma: "")

        resultado = pipeline.build_track(
            SEGMENTS, source_language="en", target_language="es", total_seconds=5.0
        )

        assert tts.velocidades == []
        assert float(np.max(np.abs(resultado.track))) == 0.0

    def test_without_a_target_language_there_is_nothing_to_dub(self) -> None:
        pipeline = make_pipeline()

        with pytest.raises(DubbingUnavailable):
            pipeline.build_track(
                SEGMENTS, source_language="en", target_language="", total_seconds=5.0
            )

    def test_dubbing_into_the_same_language_is_refused(self) -> None:
        pipeline = make_pipeline()

        with pytest.raises(DubbingUnavailable):
            pipeline.build_track(
                SEGMENTS, source_language="en", target_language="en", total_seconds=5.0
            )


class TestVoiceChoice:
    def test_dubbing_into_spanish_uses_a_spanish_voice_when_there_is_one(self) -> None:
        tts = FakeTts()
        pipeline = make_pipeline(tts=tts, available_voices=["af_heart", "ef_dora"])

        pipeline.build_track(
            SEGMENTS, source_language="en", target_language="es", total_seconds=5.0
        )

        assert set(tts.voces) == {"ef_dora"}

    def test_without_a_voice_of_that_language_it_still_dubs(self) -> None:
        # Con acento es peor que sin acento, pero mucho mejor que no doblar.
        tts = FakeTts()
        pipeline = make_pipeline(tts=tts, available_voices=["af_heart"])

        pipeline.build_track(
            SEGMENTS, source_language="en", target_language="es", total_seconds=5.0
        )

        assert set(tts.voces) == {"af_heart"}
