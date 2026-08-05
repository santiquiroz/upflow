"""Texto a fonemas IPA con espeak-ng.

Kokoro no recibe texto: su vocabulario son 115 simbolos con IPA adentro. Este es
el paso que falta entre lo que escribe el usuario y lo que el modelo entiende.

espeak-ng entra por `espeakng-loader`, que trae la DLL y los datos como wheel:
no hace falta instalar espeak-ng aparte en la maquina del usuario.
"""

from __future__ import annotations

import threading

# Cargar la libreria es global y no reentrante: dos hilos configurandola a la
# vez es una forma conocida de llevarse el proceso por delante.
_SETUP_LOCK = threading.Lock()
_configured = False

# espeak nombra los idiomas distinto que la app: "es" es castellano de Espana y
# "es-419" el de America Latina, que es el que quiere la mayoria de los usuarios
# de esta app.
LANGUAGE_ALIASES = {
    "en": "en-us",
    "es": "es-419",
    "pt": "pt-br",
}


class PhonemizerUnavailable(RuntimeError):
    pass


def espeak_language(language: str | None) -> str:
    if not language:
        return "en-us"
    return LANGUAGE_ALIASES.get(language.lower(), language.lower())


def _configure_once() -> None:
    global _configured
    with _SETUP_LOCK:
        if _configured:
            return
        try:
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise PhonemizerUnavailable(
                "Falta espeak-ng. Instalalo con scripts/download-kokoro-tts.ps1"
            ) from exc
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        _configured = True


def text_to_phonemes(text: str, language: str | None = None) -> str:
    if not text.strip():
        return ""
    _configure_once()
    from phonemizer.backend import EspeakBackend

    backend = EspeakBackend(
        espeak_language(language),
        preserve_punctuation=True,
        # El acento cambia la prosodia: sin el, la frase sale plana.
        with_stress=True,
    )
    return backend.phonemize([text])[0].strip()
