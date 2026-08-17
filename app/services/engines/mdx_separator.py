from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

import numpy as np

from app.services.engines.mdx_models import (
    MDX_HOP,
    MDX_MODELS,
    MDX_SAMPLE_RATE,
    MdxModelSpec,
)
from app.services.engines.onnx_common import wrap_onnx_error
from app.services.engines.separator_base import (
    OnnxStemSeparator,
    ProgressCallback,
    SeparationCancelled,
    normalize_stem_peaks,  # noqa: F401  (reexport: la usan los tests del motor)
    stems_in_catalog_order,
)

# ---------------------------------------------------------------------------
# Separacion voz/instrumental con MDX-Net (UVR) en ONNX, en proceso. El DSP es
# el pipeline MDX v1 tal cual lo corre UVR/python-audio-separator, portado del
# spike validado numericamente (DML == CPU) el 2026-08-09: chunks fijos de
# hop*(dim_t-1) muestras -> STFT (hann periodica, center reflect) -> tensor
# [1, 4, dim_f, dim_t] (re/im por canal, 3 bins graves a cero) -> modelo ->
# iSTFT -> se descarta trim = n_fft/2 por borde y se concatena.
#
# El modelo saca UN stem (su primary_stem), que es su salida 0; el otro se
# declara RESIDUAL en el catalogo y sale de mezcla - primario * compensate.
#
# Todo lo que no es DSP (hilo, cancelacion, cache de sesiones, I/O estereo a
# 44100) vive en OnnxStemSeparator, compartido con el motor VR.
# ---------------------------------------------------------------------------

_LOW_BINS_ZEROED = 3


@lru_cache(maxsize=4)
def _hann_periodic(n_fft: int) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))


def stft_chunk(chunk: np.ndarray, n_fft: int) -> np.ndarray:
    """chunk [2, N] -> espectro complejo [2, n_fft//2+1, T] (center reflect)."""
    padded = np.pad(chunk, ((0, 0), (n_fft // 2, n_fft // 2)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft, axis=1)[:, ::MDX_HOP, :]
    spec = np.fft.rfft(frames * _hann_periodic(n_fft), axis=2)
    return spec.transpose(0, 2, 1)


def istft_channel(spec: np.ndarray, n_fft: int) -> np.ndarray:
    """spec [n_fft//2+1, T] complejo -> señal [N] (quita el padding center)."""
    window = _hann_periodic(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=0) * window[:, None]
    n_frames = spec.shape[1]
    length = n_fft + MDX_HOP * (n_frames - 1)
    signal = np.zeros(length)
    weight = np.zeros(length)
    window_sq = window**2
    for t in range(n_frames):
        start = t * MDX_HOP
        signal[start : start + n_fft] += frames[:, t]
        weight[start : start + n_fft] += window_sq
    signal /= np.maximum(weight, 1e-10)
    return signal[n_fft // 2 : -(n_fft // 2)]


def spec_to_model_input(spec: np.ndarray, dim_f: int) -> np.ndarray:
    tensor = np.stack([spec[0].real, spec[0].imag, spec[1].real, spec[1].imag])
    tensor = tensor[None, :, :dim_f, :].astype(np.float32)
    # UVR pone a cero los 3 bins mas graves antes de inferir.
    tensor[:, :, :_LOW_BINS_ZEROED, :] = 0.0
    return np.ascontiguousarray(tensor)


def model_output_to_audio(output: np.ndarray, spec: MdxModelSpec) -> np.ndarray:
    full = np.zeros((2, spec.n_fft // 2 + 1, spec.dim_t), dtype=np.complex128)
    full[0, : spec.dim_f] = output[0, 0] + 1j * output[0, 1]
    full[1, : spec.dim_f] = output[0, 2] + 1j * output[0, 3]
    return np.stack(
        [istft_channel(full[0], spec.n_fft), istft_channel(full[1], spec.n_fft)]
    )


def stems_from_primary(
    mix: np.ndarray, primary: np.ndarray, spec: MdxModelSpec
) -> tuple[np.ndarray, ...]:
    """Audio de los stems en el orden del spec (primero = downloadUrl).

    MDX-Net infiere UNA sola cosa, asi que su unica salida es la 0 y el otro
    stem se declara RESIDUAL. `compensate` entra como escala del residuo porque
    el stem inferido sale sistematicamente atenuado. Asi voc_ft (saca la voz,
    se quiere la instrumental) y reverb_hq (saca el reverb, se quiere la pista
    limpia) invierten sin ningun caso especial aca.
    """
    return stems_in_catalog_order(mix, (primary,), spec, spec.compensate)


class MdxSeparator(OnnxStemSeparator):
    sample_rate = MDX_SAMPLE_RATE
    models = MDX_MODELS
    engine_label = "Karaoke separation"
    load_error_context = "Failed to load the separation model on device"

    def _separate_stems(
        self,
        mix: np.ndarray,
        session: Any,
        spec: MdxModelSpec,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None,
    ) -> tuple[np.ndarray, ...]:
        primary = self._run_chunks(mix, session, spec, cancel_event, on_chunk)
        return stems_from_primary(mix, primary, spec)

    def _run_chunks(
        self,
        mix: np.ndarray,
        session: Any,
        spec: MdxModelSpec,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None = None,
    ) -> np.ndarray:
        total = mix.shape[1]
        trim, gen_size, chunk = spec.trim_samples, spec.gen_samples, spec.chunk_samples
        pad = (gen_size - total % gen_size) % gen_size
        padded = np.concatenate(
            [np.zeros((2, trim)), mix, np.zeros((2, pad + trim))], axis=1
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        # Exacto: equivale a ceil(total/gen_size) porque pad completa el ultimo.
        chunks_total = (padded.shape[1] - chunk) // gen_size + 1
        pieces: list[np.ndarray] = []
        start = 0
        while start + chunk <= padded.shape[1]:
            if cancel_event.is_set():
                raise SeparationCancelled("Separation cancelled")
            window = padded[:, start : start + chunk]
            tensor = spec_to_model_input(stft_chunk(window, spec.n_fft), spec.dim_f)
            try:
                output = session.run([output_name], {input_name: tensor})[0]
            except Exception as exc:  # onnxruntime lanza sus propios tipos nativos
                raise wrap_onnx_error("Karaoke separation inference failed", exc) from exc
            audio = model_output_to_audio(np.asarray(output), spec)
            pieces.append(audio[:, trim:-trim])
            start += gen_size
            if on_chunk is not None:
                on_chunk(len(pieces), chunks_total)
        joined = np.concatenate(pieces, axis=1)
        return joined[:, :total]
