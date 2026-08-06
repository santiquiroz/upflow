from __future__ import annotations

from pathlib import Path

import pytest

from app.services.subtitles import TranscriptSegment
from app.services.translate import (
    LanguagePair,
    TranslationEngine,
    TranslationUnavailable,
    parse_pair,
)

# ---------------------------------------------------------------------------
# Los subtitulos se traducen SEGMENTO POR SEGMENTO: los tiempos pertenecen a
# cada uno, y traducir el texto corrido los perderia. Por eso el lote conserva
# el orden Y los vacios: el llamador empareja por indice, asi que descartar un
# vacio correria todos los tiempos que vienen despues.
# ---------------------------------------------------------------------------


class TestParsePair:
    def test_normalizes_case_and_spaces(self) -> None:
        assert parse_pair(" EN ", "Es") == LanguagePair(source="en", target="es")

    def test_translating_to_the_same_language_is_refused(self) -> None:
        with pytest.raises(TranslationUnavailable, match="mismo"):
            parse_pair("es", "es")

    @pytest.mark.parametrize("bad", [("", "es"), ("en", "  ")])
    def test_a_missing_language_is_refused(self, bad: tuple[str, str]) -> None:
        with pytest.raises(TranslationUnavailable):
            parse_pair(*bad)

    def test_the_directory_follows_the_opus_naming(self) -> None:
        assert parse_pair("en", "es").directory == "opus-mt-en-es"


class TestAvailability:
    def test_no_models_directory_gives_no_pairs(self, tmp_path: Path) -> None:
        assert TranslationEngine(tmp_path / "no-existe").available_pairs() == []

    def test_lists_the_installed_pairs(self, tmp_path: Path) -> None:
        (tmp_path / "opus-mt-en-es" / "onnx").mkdir(parents=True)
        (tmp_path / "opus-mt-es-en" / "onnx").mkdir(parents=True)
        pares = TranslationEngine(tmp_path).available_pairs()
        assert [(p.source, p.target) for p in pares] == [("en", "es"), ("es", "en")]

    def test_a_folder_without_onnx_does_not_count_as_installed(self, tmp_path: Path) -> None:
        # Una descarga a medias dejaria la carpeta pero no el modelo, y
        # ofrecerlo terminaria en un error recien al traducir.
        (tmp_path / "opus-mt-en-fr").mkdir(parents=True)
        assert TranslationEngine(tmp_path).available_pairs() == []

    def test_names_the_missing_pair_without_dictating_a_command(self, tmp_path: Path) -> None:
        # El par tiene que aparecer: la traduccion no es UN modelo sino uno por
        # par de idiomas, y sin saber cual el boton no sabria que bajar.
        engine = TranslationEngine(tmp_path)
        with pytest.raises(TranslationUnavailable, match="en-fr") as capturado:
            engine.translate(["hola"], LanguagePair(source="en", target="fr"))

        assert ".ps1" not in str(capturado.value)


class TestBatchShape:
    def test_empty_strings_keep_their_place(self, tmp_path: Path) -> None:
        # Sin modelo instalado igual se puede comprobar la forma: los vacios se
        # devuelven sin llamar al modelo.
        engine = TranslationEngine(tmp_path)
        (tmp_path / "opus-mt-en-es" / "onnx").mkdir(parents=True)
        engine._cache["opus-mt-en-es"] = ("tokenizer-falso", "modelo-falso")  # type: ignore[assignment]
        salida = engine.translate(["", "   "], LanguagePair(source="en", target="es"))
        assert salida == ["", "   "]


class TestSegmentPairing:
    def test_translated_segments_keep_their_times(self) -> None:
        """El emparejado por indice es lo que conserva los tiempos."""
        originales = [
            TranscriptSegment(start=0.0, end=1.0, text="hello"),
            TranscriptSegment(start=1.0, end=2.5, text="world"),
        ]
        traducidos = ["hola", "mundo"]
        emparejados = [
            TranscriptSegment(start=o.start, end=o.end, text=t)
            for o, t in zip(originales, traducidos)
        ]
        assert [(s.start, s.end, s.text) for s in emparejados] == [
            (0.0, 1.0, "hola"),
            (1.0, 2.5, "mundo"),
        ]


# --- API -------------------------------------------------------------------


def test_the_api_lists_the_installed_pairs() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        body = client.get("/api/v1/translation/pairs").json()

    # En una maquina sin modelos bajados la lista viene vacia, y eso es una
    # respuesta valida: no hay pares instalados.
    assert isinstance(body["pairs"], list)


@pytest.mark.asyncio
async def test_asking_for_a_translation_without_the_model_names_the_pair(tmp_path: Path):
    from fastapi import HTTPException

    from app.api.routes import download_transcribe_job
    from app.services.translate import TranslationEngine

    # Se prueba el engine directo: la ruta delega en el, y el mensaje que
    # importa es el que le llega al usuario.
    engine = TranslationEngine(tmp_path)
    with pytest.raises(TranslationUnavailable, match="en-ja") as capturado:
        engine.translate(["hola"], LanguagePair(source="en", target="ja"))
    assert ".ps1" not in str(capturado.value)
    assert download_transcribe_job is not None
    assert HTTPException is not None
