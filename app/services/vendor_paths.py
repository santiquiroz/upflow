"""Donde viven los modelos que se bajan aparte con los scripts de `scripts/`.

Estaban calculados a mano dentro de las rutas. Al aparecer el doblaje, que usa
traduccion Y sintesis desde el manager de jobs, importar `routes` desde ahi
habria cerrado un ciclo: por eso las rutas viven aca y las usan los dos.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings


def vendor_root(settings: Settings) -> Path:
    return Path(settings.runtime_dir).parent / "vendor"


def kokoro_dir(settings: Settings) -> Path:
    return vendor_root(settings) / "kokoro"


def translation_dir(settings: Settings) -> Path:
    return vendor_root(settings) / "translation"
