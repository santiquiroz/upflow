from __future__ import annotations

import asyncio
import contextlib
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.services.engines.audio_restore_base import OnnxAudioRestorer
from app.services.engines.mdx_models import (
    MDX_HOP,
    MDX_SAMPLE_RATE,
    DEFAULT_SEPARATION_MODEL,
    MdxModelSpec,
    SEPARATION_MODELS,
    model_file,
)
from app.services.engines.onnx_common import wrap_onnx_error
from app.services.missing_pack import missing_pack_message

# ---------------------------------------------------------------------------
# Separacion voz/instrumental con MDX-Net (UVR) en ONNX, en proceso. El DSP es
# el pipeline MDX v1 tal cual lo corre UVR/python-audio-separator, portado del
# spike validado numericamente (DML == CPU) el 2026-08-09: chunks fijos de
# hop*(dim_t-1) muestras -> STFT (hann periodica, center reflect) -> tensor
# [1, 4, dim_f, dim_t] (re/im por canal, 3 bins graves a cero) -> modelo ->
# iSTFT -> se descarta trim = n_fft/2 por borde y se concatena.
#
# El modelo saca UN stem (su primary_stem); el otro es la resta:
# secundario = mezcla - primario * compensate.
#
# A diferencia de Apollo/AudioSR esto procesa ESTEREO nativo (el modelo espera
# 2 canales); mono se duplica. La cache de sesiones, el coordinador de GPU y el
# I/O de audio se heredan de OnnxAudioRestorer, con la clave de cache extendida
# a device+modelo porque aca hay mas de un grafo posible por device.
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
) -> tuple[np.ndarray, np.ndarray]:
    """Audio de los DOS stems en el orden del spec (primero = downloadUrl).

    Cada MdxStem declara su fuente: "primary" es lo que el modelo infiere y
    "secondary" la resta compensada. Asi voc_ft (saca la voz, se quiere la
    instrumental) y reverb_hq (saca el reverb, se quiere la pista limpia)
    invierten sin ningun caso especial aca.
    """
    residual = mix - primary * spec.compensate
    by_source = {"primary": primary, "secondary": residual}
    return by_source[spec.stems[0].source], by_source[spec.stems[1].source]


def normalize_stem_peaks(
    main: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Factor UNICO para ambos stems, solo si algun pico supera 1.0.

    La resta compensada supera +-1.0 donde el modelo queda fuera de fase con la
    mezcla; un factor compartido preserva el balance relativo y evita que el
    encode entero final (flac/mp3) recorte esos picos.
    """
    peak = max(float(np.max(np.abs(main))), float(np.max(np.abs(other))))
    if peak <= 1.0:
        return main, other
    return main / peak, other / peak


class SeparationCancelled(RuntimeError):
    pass


class MdxSeparator(OnnxAudioRestorer):
    sample_rate = MDX_SAMPLE_RATE
    # Un solo grafo vivo: cambiar de modelo o de device recarga (~67MB, carga
    # de ~1s), y a cambio nunca hay dos copias en VRAM.
    session_cache_size = 1
    pack_name = "karaoke"
    missing_pack_detail = ""
    engine_label = "Karaoke separation"
    load_error_context = "Failed to load the separation model on device"

    def available(self) -> bool:
        return self.settings.karaoke_available()

    def model_available(self, model_id: str) -> bool:
        return (
            model_id in SEPARATION_MODELS
            and model_file(self.settings.karaoke_model_dir_path, model_id).exists()
        )

    async def run(
        self,
        input_wav: Path,
        main_wav: Path,
        other_wav: Path,
        device: str,
        model_id: str = DEFAULT_SEPARATION_MODEL,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        # main_wav recibe el stem que el usuario quiere (spec.main_stem) y
        # other_wav el restante, en el orden que declara el catalogo.
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._separate_and_save,
                input_wav,
                main_wav,
                other_wav,
                device,
                model_id,
                cancel_event,
                on_chunk,
            )
        )
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # to_thread no puede interrumpir el worker; el loop de chunks mira
            # este event. Esperar a que el hilo termine de verdad evita que el
            # rmtree del work dir corra en paralelo con un save_wav rezagado.
            # El loop re-shieldea: un segundo cancel_job cancela solo el shield
            # externo, no corta la espera del hilo.
            cancel_event.set()
            while not worker.done():
                with contextlib.suppress(BaseException):
                    await asyncio.shield(worker)
            raise
        self._ensure_output_file(main_wav)
        self._ensure_output_file(other_wav)

    def _separate_and_save(
        self,
        input_wav: Path,
        main_wav: Path,
        other_wav: Path,
        device: str,
        model_id: str,
        cancel_event: threading.Event,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        spec = self._require_model(model_id)
        mix = self._load_stereo(input_wav)
        session = self._get_session(device, model_id)
        primary = self._run_chunks(mix, session, spec, cancel_event, on_chunk)
        main, other = normalize_stem_peaks(*stems_from_primary(mix, primary, spec))
        self._save_stereo(main_wav, main)
        self._save_stereo(other_wav, other)

    def _require_model(self, model_id: str) -> MdxModelSpec:
        spec = SEPARATION_MODELS.get(model_id)
        if spec is None:
            known = ", ".join(sorted(SEPARATION_MODELS))
            raise RuntimeError(
                f"Modelo de separacion desconocido: {model_id!r}. Validos: {known}."
            )
        if not self.model_available(model_id):
            raise RuntimeError(missing_pack_message(self.pack_name, variant=model_id))
        return spec

    def _load_stereo(self, input_wav: Path) -> np.ndarray:
        audio = self._load_audio(input_wav)  # [samples, channels] @ 44100
        if audio.shape[0] == 0:
            raise RuntimeError(
                "The uploaded audio decoded to zero samples; the file is empty or corrupted"
            )
        channels = audio.shape[1]
        if channels == 1:
            return np.repeat(audio.T, 2, axis=0)
        if channels == 2:
            return np.ascontiguousarray(audio.T)
        # El pipeline decodifica con -ac 2 antes de llegar aca; si alguien llama
        # al motor directo con surround, mejor decirlo que adivinar un downmix.
        raise RuntimeError(
            f"Karaoke separation expects mono or stereo audio, got {channels} channels"
        )

    def _run_chunks(
        self,
        mix: np.ndarray,
        session: Any,
        spec: MdxModelSpec,
        cancel_event: threading.Event,
        on_chunk: Callable[[int, int], None] | None = None,
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

    def _save_stereo(self, output_wav: Path, audio: np.ndarray) -> None:
        from app.services.engines.audio_restore_base import save_wav

        # FLOAT: el default PCM_16 de soundfile cuantizaba a 16 bits ANTES del
        # encode final y recortaba cualquier resto de overshoot.
        save_wav(output_wav, audio.T.astype(np.float32), self.sample_rate, subtype="FLOAT")

    # -- sesiones: clave device+modelo (hay mas de un grafo posible) ---------

    def _get_session(self, device: str, model_id: str) -> Any:
        self.gpu_coordinator.acquire(device, self)
        key = self._cache_key(device, model_id)
        with self._session_lock:
            cached = self._session_cache.get(key)
            if cached is not None:
                self._session_cache.move_to_end(key)
                return cached

        # Se construye fuera del lock (carga lenta del grafo); un doble build
        # en un miss concurrente solo desperdicia trabajo, igual que la base.
        try:
            session = self._create_session(device, model_id)
        except Exception as exc:
            raise wrap_onnx_error(f"{self.load_error_context} {device!r}", exc) from exc

        with self._session_lock:
            self._session_cache[key] = session
            self._session_cache.move_to_end(key)
            if len(self._session_cache) > self.session_cache_size:
                self._session_cache.popitem(last=False)
        return session

    def release_device(self, device: str) -> None:
        prefix = f"{device}::"
        with self._session_lock:
            for key in [k for k in self._session_cache if k.startswith(prefix)]:
                self._session_cache.pop(key, None)

    @staticmethod
    def _cache_key(device: str, model_id: str) -> str:
        return f"{device}::{model_id}"

    def _create_session(self, device: str, model_id: str) -> Any:
        # Seam monkeypatcheable: los tests inyectan sesiones fake por nombre.
        # Optimizacion de grafo DEFAULT a proposito: el spike verifico que este
        # grafo NO cuelga DirectML (a diferencia de GMFSS/MetricNet).
        from app.services import ep_registry

        path = model_file(self.settings.karaoke_model_dir_path, model_id)
        return ep_registry.create_session(str(path), device, self.settings)
