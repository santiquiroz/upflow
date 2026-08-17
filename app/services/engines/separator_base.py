from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np

from app.services.engines.audio_restore_base import OnnxAudioRestorer, save_wav
from app.services.engines.onnx_common import wrap_onnx_error
from app.services.engines.separation_models import (
    DEFAULT_SEPARATION_MODEL,
    model_file,
)
from app.services.engines.separation_spec import RESIDUAL, SeparationModelSpec
from app.services.missing_pack import missing_pack_message

# ---------------------------------------------------------------------------
# Lo que comparten TODOS los motores de separacion, sea cual sea la
# arquitectura del modelo: el ciclo de vida del hilo con cancelacion
# cooperativa, la cache de sesiones ONNX por device+modelo, el I/O estereo a
# 44100 y el contrato publico `run(input, outputs, device, model_id, on_chunk)`
# — que es lo unico que ve el pipeline.
#
# `outputs` es una SECUENCIA alineada con `spec.stems`, no un par: hay modelos
# de dos stems (karaoke, limpieza) y de cuatro (bateria/bajo/resto/voz), y el
# ciclo de vida es el mismo para ambos.
#
# Lo que NO comparten (el DSP entre la mezcla y las salidas del modelo) queda
# en `_separate_stems`, el unico metodo que cada motor implementa.
# ---------------------------------------------------------------------------

SEPARATION_SAMPLE_RATE = 44100

ProgressCallback = Callable[[int, int], None]


class SeparationCancelled(RuntimeError):
    pass


def normalize_stem_peaks(stems: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Factor UNICO para todos los stems, solo si algun pico supera 1.0.

    La resta compensada supera +-1.0 donde el modelo queda fuera de fase con la
    mezcla; un factor compartido preserva el balance relativo y evita que el
    encode entero final (flac/mp3) recorte esos picos. Compartido entre TODOS
    los stems y no por stem: normalizar cada uno por su cuenta cambiaria la
    mezcla relativa entre ellos.
    """
    peak = max(float(np.max(np.abs(stem))) for stem in stems)
    if peak <= 1.0:
        return tuple(stems)
    return tuple(stem / peak for stem in stems)


def stems_in_catalog_order(
    mix: np.ndarray,
    model_outputs: Sequence[np.ndarray],
    spec: SeparationModelSpec,
    residual_scale: float = 1.0,
) -> tuple[np.ndarray, ...]:
    """Las salidas del modelo, reordenadas y completadas como pide el catalogo.

    Cada `SeparationStem` dice de donde sale su audio: un indice de
    `model_outputs`, o RESIDUAL (lo que queda de la mezcla al restarle todo lo
    que el modelo si infiere, escalado por `residual_scale` — el `compensate`
    de MDX-Net, 1.0 en todo lo demas).

    Con esto los tres motores comparten el mismo mapeo: da igual que el modelo
    emita una salida y su complemento se calcule, que emita dos mascaras
    complementarias, o que emita cuatro stems y no haya residuo.
    """
    residual: np.ndarray | None = None
    resolved = []
    for stem in spec.stems:
        if stem.source == RESIDUAL:
            if residual is None:
                residual = mix - residual_scale * sum(model_outputs)
            resolved.append(residual)
            continue
        resolved.append(model_outputs[stem.source])
    return tuple(resolved)


class OnnxStemSeparator(OnnxAudioRestorer):
    """Base de los motores de separacion en ONNX (MDX-Net y VR CascadedNet)."""

    sample_rate = SEPARATION_SAMPLE_RATE
    # Un solo grafo vivo: cambiar de modelo o de device recarga, y a cambio
    # nunca hay dos copias en VRAM.
    session_cache_size = 1
    pack_name = "karaoke"
    missing_pack_detail = ""
    default_model_id: ClassVar[str] = DEFAULT_SEPARATION_MODEL
    # Los modelos del catalogo que ESTE motor sabe correr.
    models: ClassVar[dict[str, Any]] = {}

    def available(self) -> bool:
        return self.settings.karaoke_available()

    def model_available(self, model_id: str) -> bool:
        return (
            model_id in self.models
            and model_file(self.settings.karaoke_model_dir_path, model_id).exists()
        )

    async def run(
        self,
        input_wav: Path,
        outputs: Sequence[Path],
        device: str,
        model_id: str | None = None,
        on_chunk: ProgressCallback | None = None,
    ) -> None:
        # `outputs` va alineado 1 a 1 con spec.stems: el primero recibe el stem
        # que el usuario quiere (spec.main_stem) y el resto los demas, en el
        # orden que declara el catalogo.
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._separate_and_save,
                input_wav,
                tuple(outputs),
                device,
                model_id or self.default_model_id,
                cancel_event,
                on_chunk,
            )
        )
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # to_thread no puede interrumpir el worker; el loop de ventanas mira
            # este event. Esperar a que el hilo termine de verdad evita que el
            # rmtree del work dir corra en paralelo con un save_wav rezagado.
            # El loop re-shieldea: un segundo cancel_job cancela solo el shield
            # externo, no corta la espera del hilo.
            cancel_event.set()
            while not worker.done():
                with contextlib.suppress(BaseException):
                    await asyncio.shield(worker)
            raise
        for output in outputs:
            self._ensure_output_file(output)

    def _separate_and_save(
        self,
        input_wav: Path,
        outputs: tuple[Path, ...],
        device: str,
        model_id: str,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None = None,
    ) -> None:
        spec = self._require_model(model_id)
        if len(outputs) != len(spec.stems):
            raise RuntimeError(
                f"{spec.id} declara {len(spec.stems)} stems "
                f"{spec.stem_ids()} y se pidieron {len(outputs)} archivos de salida"
            )
        mix = self._load_stereo(input_wav)
        session = self._get_session(device, model_id)
        stems = self._separate_stems(mix, session, spec, cancel_event, on_chunk)
        for output, audio in zip(outputs, normalize_stem_peaks(stems)):
            self._save_stereo(output, audio)

    def _separate_stems(
        self,
        mix: np.ndarray,
        session: Any,
        spec: Any,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None,
    ) -> tuple[np.ndarray, ...]:
        """Un stem por entrada de `spec.stems`, en ese orden (primero = downloadUrl)."""
        raise NotImplementedError

    def _require_model(self, model_id: str) -> SeparationModelSpec:
        spec = self.models.get(model_id)
        if spec is None:
            known = ", ".join(sorted(self.models))
            raise RuntimeError(
                f"Modelo de separacion desconocido para {self.engine_label}: "
                f"{model_id!r}. Validos: {known}."
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
            f"Stem separation expects mono or stereo audio, got {channels} channels"
        )

    def _save_stereo(self, output_wav: Path, audio: np.ndarray) -> None:
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
        # Optimizacion de grafo DEFAULT a proposito: el spike verifico que estos
        # grafos NO cuelgan DirectML (a diferencia de GMFSS/MetricNet).
        from app.services import ep_registry

        path = model_file(self.settings.karaoke_model_dir_path, model_id)
        return ep_registry.create_session(str(path), device, self.settings)
