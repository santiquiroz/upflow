# La pestaña "Restore photo" y el carril IA de CCTV dependen de packs que no estan
# publicados (D1a/b/c y D2 pendientes): apagados, el servidor dice por que en vez de
# mandar a bajar un pack que no existe.

from __future__ import annotations

from app.config import Settings

RESTORE_DISABLED_MESSAGE = (
    "Photo restoration is turned off in this release: its AI model packs are not published yet."
)
CCTV_AI_DISABLED_MESSAGE = (
    'The CCTV "AI enhancement (visual only)" lane is turned off in this release: '
    "its AI model pack is not published yet."
)


class FeatureDisabledError(ValueError):
    pass


def ensure_restore_enabled(settings: Settings) -> None:
    if not settings.restore_photo_enabled:
        raise FeatureDisabledError(RESTORE_DISABLED_MESSAGE)


def ensure_cctv_ai_enabled(settings: Settings) -> None:
    if not settings.cctv_ai_enabled:
        raise FeatureDisabledError(CCTV_AI_DISABLED_MESSAGE)
