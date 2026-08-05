from __future__ import annotations

import pytest

from app.services.phonemize import espeak_language, text_to_phonemes

# espeak nombra los idiomas distinto que la app. Mandarle "es" le da castellano
# de Espana; el usuario de esta app espera el de America Latina.


class TestLanguageMapping:
    def test_spanish_defaults_to_latin_american(self) -> None:
        assert espeak_language("es") == "es-419"

    def test_english_defaults_to_american(self) -> None:
        assert espeak_language("en") == "en-us"

    def test_no_language_falls_back_to_english(self) -> None:
        assert espeak_language(None) == "en-us"

    def test_an_explicit_variant_is_respected(self) -> None:
        # Quien pide "en-gb" lo pide a proposito.
        assert espeak_language("en-gb") == "en-gb"


class TestPhonemization:
    def test_empty_text_gives_no_phonemes_without_touching_espeak(self) -> None:
        assert text_to_phonemes("   ") == ""

    def test_english_produces_ipa(self) -> None:
        out = text_to_phonemes("hello world", "en")
        assert out, "espeak no devolvio nada"
        # Alguno de estos simbolos IPA tiene que aparecer en cualquier frase
        # inglesa razonable.
        assert any(symbol in out for symbol in "ɪəʊɹˈː")

    def test_spanish_also_produces_ipa(self) -> None:
        out = text_to_phonemes("hola mundo", "es")
        assert out
        assert " " in out, "las palabras tienen que quedar separadas"

    def test_stress_marks_survive(self) -> None:
        # Sin acento la frase sale plana.
        assert "ˈ" in text_to_phonemes("computer", "en")
