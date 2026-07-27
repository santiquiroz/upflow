from __future__ import annotations

# ---------------------------------------------------------------------------
# ¿Este encoder de hardware puede con ESTA resolución, en ESTA máquina?
#
# Antes se deducía del NOMBRE de la GPU y se asumía que el encoder existía y
# aguantaba cualquier tamaño. Dos formas de fallar con eso, ambas medidas:
#   - el encoder puede no estar (driver viejo, build de ffmpeg sin AMF/NVENC/QSV,
#     GPU virtualizada) y el fallo aparece recién al encodear de verdad;
#   - hay techos de resolución por familia: h264_amf revienta a 7680x4320 con
#     "encoder->Init() failed with error 5" mientras hevc_amf y av1_amf pasan.
#     NVENC y QSV tienen límites análogos según generación, así que una tabla
#     fija de topes envejece mal y hay que mantenerla por vendor.
#
# Probar es más barato y más honesto que tabular: un encode de un frame sintético
# a la resolución real responde la pregunta exacta en la máquina exacta. El
# resultado se cachea por (encoder, ancho, alto) — la resolución forma parte de
# la clave justamente porque el techo es lo que se está buscando.
# ---------------------------------------------------------------------------

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Un encoder de software siempre puede: libx264/libx265 no dependen del driver
# ni tienen techos prácticos a estas resoluciones, y son el destino del fallback.
_SOFTWARE_ENCODERS = frozenset({"libx264", "libx265", "libsvtav1", "libaom-av1", "librav1e"})

_PROBE_TIMEOUT_SECONDS = 120


class EncoderCapabilityProbe:
    """Cachea si un encoder de hardware sirve para una resolución dada."""

    def __init__(
        self,
        ffmpeg_binary: Path,
        runner: Callable[[str, int, int], bool] | None = None,
        timeout_seconds: int = _PROBE_TIMEOUT_SECONDS,
    ) -> None:
        self._ffmpeg_binary = ffmpeg_binary
        self._timeout_seconds = timeout_seconds
        # Seam inyectable: los unit tests no corren ffmpeg de verdad.
        self._runner = runner or self._trial_encode
        self._cache: dict[tuple[str, int, int], bool] = {}

    def supports(self, encoder: str, width: int, height: int) -> bool:
        if encoder in _SOFTWARE_ENCODERS:
            return True
        key = (encoder, width, height)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            supported = bool(self._runner(encoder, width, height))
        except Exception:  # noqa: BLE001 - sin certeza el caller cae a software
            logger.exception("encoder probe failed for %s at %dx%d", encoder, width, height)
            supported = False
        self._cache[key] = supported
        if not supported:
            logger.info("encoder %s no soporta %dx%d en esta máquina", encoder, width, height)
        return supported

    def _trial_encode(self, encoder: str, width: int, height: int) -> bool:
        # Un frame sintético a la resolución real, descartando la salida: mide
        # exactamente lo que importa (que el encoder inicialice a ese tamaño)
        # sin tocar el disco ni depender del contenido del job.
        command = [
            str(self._ffmpeg_binary),
            "-v", "error",
            "-f", "lavfi",
            "-i", f"color=c=black:size={width}x{height}:rate=1:duration=1",
            "-frames:v", "1",
            "-c:v", encoder,
            "-pix_fmt", "yuv420p",
            "-f", "null", "-",
        ]
        result = subprocess.run(  # noqa: S603 - comando construido acá, sin input del usuario
            command, capture_output=True, timeout=self._timeout_seconds
        )
        return result.returncode == 0
