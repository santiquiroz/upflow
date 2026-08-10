from __future__ import annotations

import math
import threading
from typing import Any, Callable

import numpy as np

from app.services.engines.onnx_common import wrap_onnx_error
from app.services.engines.separator_base import (
    OnnxStemSeparator,
    ProgressCallback,
    SeparationCancelled,
)
from app.services.engines.vr_deecho import DeEchoDriver
from app.services.engines.vr_models import VR_MODELS, VR_SAMPLE_RATE, VrModelSpec

# ---------------------------------------------------------------------------
# Limpieza de eco/reverb con los modelos VR 5.1 CascadedNet de FoxJoy, en
# proceso. TODO el DSP (analisis/sintesis multibanda 4band_v3, normalizacion,
# ventaneo ROI, curva de aggression, mascara compleja) es el driver vendorizado
# en app/services/engines/vr_deecho/ — aca solo se arma la sesion ONNX, se
# acota la memoria por bloques y se cablean cancelacion y progreso.
#
# A diferencia de MDX estos modelos predicen DIRECTO la señal limpia (su
# primary_stem es "No Echo"/"No Reverb"); el eco/reverb es la mascara
# complementaria, no una resta compensada.
# ---------------------------------------------------------------------------

# Geometria del ventaneo del driver (vr_deecho/vr_params.py + pipeline.py). Se
# repite aca SOLO para contar ventanas por adelantado y poder reportar progreso;
# test_vr_deecho_separator.py gatea que el conteo coincida con las llamadas
# reales al grafo, asi que un cambio del driver rompe la suite en vez de mentir
# el porcentaje.
_ROI_FRAMES = 384
_TOP_BAND_HOP = 480

# La sintesis multibanda devuelve hasta ~480 muestras MENOS que la entrada
# (cuantizacion de frames por banda). Se le agrega silencio al final de cada
# bloque para que el tramo util quede cubierto y no haya que rellenar con ceros
# la cola — un corte seco de 8 ms al final del archivo es un click evitable.
_TAIL_PAD_SAMPLES = 4096


def combined_spec_frames(samples: int) -> int:
    """Frames del espectrograma combinado para `samples` muestras a 44100.

    Mismo minimo entre bandas que multiband.combine_spectrograms, derivado de
    las longitudes de resample_poly (44100 -> 14700 -> 7350) y de los hops de
    4band_v3.
    """
    band_14700 = math.ceil(samples / 3)
    band_7350 = math.ceil(band_14700 / 2)
    return min(
        samples // _TOP_BAND_HOP + 1,
        band_14700 // 160 + 1,
        band_7350 // 80 + 1,
    )


def graph_windows(samples: int) -> int:
    """Cuantas llamadas al grafo consume un bloque de `samples` muestras."""
    return combined_spec_frames(samples) // _ROI_FRAMES + 1


def block_ranges(
    total: int, block: int, margin: int, fade: int
) -> list[tuple[int, int, int, int]]:
    """Bloques (lo, hi, keep_lo, keep_hi) que cubren [0, total) sin huecos.

    `lo..hi` es lo que se le da al driver (incluye el margen de contexto que
    despues se descarta); `keep_lo..keep_hi` es el tramo del resultado que
    sobrevive. Cada bloque menos el primero ARRANCA `fade` muestras antes de su
    tramo propio, asi la rampa de subida de uno cae exactamente sobre la de
    bajada del anterior y las dos suman 1.
    """
    if total <= block:
        return [(0, total, 0, total)]
    ranges = []
    for start in range(0, total, block):
        keep_hi = min(start + block, total)
        if start >= keep_hi:
            break
        keep_lo = start - fade if start > 0 else 0
        ranges.append((max(0, start - margin), min(total, keep_hi + margin), keep_lo, keep_hi))
    return ranges


def fit_length(wave: np.ndarray, samples: int) -> np.ndarray:
    """El round-trip multibanda no devuelve exactamente la longitud de entrada."""
    if wave.shape[1] == samples:
        return wave
    if wave.shape[1] > samples:
        return wave[:, :samples]
    return np.pad(wave, ((0, 0), (0, samples - wave.shape[1])))


def crossfade_window(length: int, fade_in: int, fade_out: int) -> np.ndarray:
    """Rampas LINEALES complementarias: dos bloques contiguos suman 1 exacto."""
    window = np.ones(length, dtype=np.float32)
    if fade_in > 0:
        window[:fade_in] = np.linspace(0.0, 1.0, fade_in, endpoint=False)
    if fade_out > 0:
        window[length - fade_out :] = np.linspace(1.0, 0.0, fade_out, endpoint=False)
    return window


def separate_padded(
    driver: DeEchoDriver, segment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """driver.separate() sobre el bloque con cola de silencio, ya recortado."""
    padded = np.pad(segment, ((0, 0), (0, _TAIL_PAD_SAMPLES)))
    primary, secondary = driver.separate(padded)
    samples = segment.shape[1]
    return fit_length(primary, samples), fit_length(secondary, samples)


class VrDeEchoSeparator(OnnxStemSeparator):
    sample_rate = VR_SAMPLE_RATE
    models = VR_MODELS
    engine_label = "De-echo separation"
    load_error_context = "Failed to load the de-echo model on device"
    default_model_id = "deecho_normal"

    def _separate_stems(
        self,
        mix: np.ndarray,
        session: Any,
        spec: VrModelSpec,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = mix.shape[1]
        block, margin, fade = self._block_geometry(total)
        ranges = block_ranges(total, block, margin, fade)
        windows_total = sum(
            graph_windows(hi - lo + _TAIL_PAD_SAMPLES) for lo, hi, _, _ in ranges
        )
        run_graph = self._make_run_graph(session, cancel_event, on_chunk, windows_total)
        driver = DeEchoDriver(run_graph, aggression=spec.aggression)

        primary = np.zeros((2, total), dtype=np.float32)
        secondary = np.zeros((2, total), dtype=np.float32)
        for lo, hi, keep_lo, keep_hi in ranges:
            block_primary, block_secondary = separate_padded(driver, mix[:, lo:hi])
            window = crossfade_window(
                keep_hi - keep_lo,
                fade if keep_lo > 0 else 0,
                fade if keep_hi < total else 0,
            )
            offset = keep_lo - lo
            for source, destination in (
                (block_primary, primary),
                (block_secondary, secondary),
            ):
                piece = source[:, offset : offset + window.size]
                destination[:, keep_lo:keep_hi] += piece * window

        by_source = {"primary": primary, "secondary": secondary}
        return by_source[spec.stems[0].source], by_source[spec.stems[1].source]

    def _block_geometry(self, total: int) -> tuple[int, int, int]:
        """(bloque, margen, crossfade) en muestras.

        El espectrograma combinado de 4band_v3 cuesta ~7 MB por segundo de
        audio contando todos los intermedios, asi que una pista larga entera no
        entra en memoria: se corta en bloques. El margen se procesa y se tira
        (contexto para el ventaneo del driver) y los bloques contiguos se
        cruzan con rampas complementarias, porque la normalizacion del driver
        es por espectrograma y un corte seco dejaria un escalon audible.
        Un archivo mas corto que un bloque corre de una sola pasada, que es
        EXACTAMENTE el camino que el port valido contra python-audio-separator.
        """
        block = max(1, int(self.settings.vr_separation_block_seconds * self.sample_rate))
        if total <= block:
            return block, 0, 0
        margin = int(self.settings.vr_separation_margin_seconds * self.sample_rate)
        return block, margin, min(margin, block // 2)

    def _make_run_graph(
        self,
        session: Any,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None,
        windows_total: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """El seam donde entran cancelacion y progreso: el driver llama a esto
        una vez por ventana de 384 frames ROI (~4.18 s de audio)."""
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        done = 0

        def run_graph(window: np.ndarray) -> np.ndarray:
            nonlocal done
            if cancel_event.is_set():
                raise SeparationCancelled("Separation cancelled")
            try:
                output = session.run([output_name], {input_name: window})[0]
            except Exception as exc:  # onnxruntime lanza sus propios tipos nativos
                raise wrap_onnx_error("De-echo separation inference failed", exc) from exc
            done += 1
            if on_chunk is not None:
                on_chunk(min(done, windows_total), windows_total)
            return np.asarray(output)

        return run_graph
