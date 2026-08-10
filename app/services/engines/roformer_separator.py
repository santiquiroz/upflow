from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np

from app.services.engines.onnx_common import wrap_onnx_error
from app.services.engines.roformer import RoformerDriver, RoformerSpec
from app.services.engines.roformer.chunking import plan_chunks
from app.services.engines.roformer_models import (
    ROFORMER_MODELS,
    ROFORMER_SAMPLE_RATE,
    RoformerModelSpec,
)
from app.services.engines.separation_blocks import block_ranges, overlap_add_blocks
from app.services.engines.separator_base import (
    OnnxStemSeparator,
    ProgressCallback,
    SeparationCancelled,
)

# ---------------------------------------------------------------------------
# Separacion voz/instrumental con Mel-Band RoFormer, el carril de MAXIMA
# CALIDAD del catalogo. Todo el DSP (STFT, multiplicacion compleja de la
# mascara, filtro DC, iSTFT y el overlap-add de chunks solapados de MSST) es el
# driver vendorizado en app/services/engines/roformer/ — aca solo se arma la
# sesion ONNX, se acota la memoria por bloques y se cablean admision de
# memoria, cancelacion y progreso.
#
# DOS diferencias con MdxSeparator que no son detalles:
#
# * El residual es una resta SIMPLE. MDX-Net necesita `compensate` porque su
#   stem inferido sale sistematicamente atenuado; estos grafos no, y meterle un
#   factor aca dejaria la instrumental fuera de fase con la mezcla.
# * El chunk es FIJO (352800 muestras = 8 s). El rotary embedding cachea su
#   tabla de frecuencias por seq_len, asi que el eje temporal quedo horneado al
#   trazar el grafo. No hay nada que tunear: los numeros salen del catalogo y
#   tienen que coincidir con el .onnx.
# ---------------------------------------------------------------------------


def driver_spec(spec: RoformerModelSpec) -> RoformerSpec:
    """Los numeros del catalogo, con la forma que espera el driver."""
    return RoformerSpec(
        n_fft=spec.n_fft,
        hop_length=spec.hop_length,
        chunk_size=spec.chunk_size,
        num_overlap=spec.num_overlap,
        audio_channels=2,
        sample_rate=ROFORMER_SAMPLE_RATE,
        stems=spec.graph_stems,
    )


def graph_chunks(samples: int, spec: RoformerModelSpec) -> int:
    """Cuantas llamadas al grafo consume un bloque de `samples` muestras.

    Se le pregunta al PLANIFICADOR del driver en vez de re-derivar la cuenta:
    si un re-vendoreo cambia el solapamiento, el porcentaje sigue diciendo la
    verdad en vez de mentir por drift.
    """
    return plan_chunks(samples, spec.chunk_size, spec.num_overlap).num_chunks


def stems_from_vocals(
    mix: np.ndarray, vocals: np.ndarray, spec: RoformerModelSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Los DOS stems en el orden del catalogo (primero = downloadUrl).

    El residual es `mezcla - vocals` y NADA MAS: sin el factor `compensate` de
    MDX-Net. El modelo infiere un solo stem y su complemento es exacto.
    """
    residual = mix - vocals
    by_source = {"primary": vocals, "secondary": residual}
    return by_source[spec.stems[0].source], by_source[spec.stems[1].source]


def free_memory_mb(device: str) -> int | None:
    """Memoria libre del device, o None si no se puede medir.

    Reusa las mismas sondas que la admision de jobs (resource_probes), asi que
    "libre" quiere decir aca lo mismo que alla. None = no medible, y el llamador
    trata eso como fail-open, igual que DeviceSemaphores.
    """
    from app.services.devices_service import CPU_DEVICE_ID
    from app.services.dml_device import try_parse_dml_device_id
    from app.services.resource_probes import DxgiVramProbe, SystemRamProbe

    if device == CPU_DEVICE_ID:
        return SystemRamProbe().free_capacity_mb(device)
    if try_parse_dml_device_id(device) is not None:
        return DxgiVramProbe().free_capacity_mb(device)
    return None


class RoformerSeparator(OnnxStemSeparator):
    sample_rate = ROFORMER_SAMPLE_RATE
    models = ROFORMER_MODELS
    engine_label = "Mel-Band RoFormer separation"
    load_error_context = "Failed to load the Mel-Band RoFormer model on device"
    default_model_id = "mel_band_roformer_kim"

    def _separate_stems(
        self,
        mix: np.ndarray,
        session: Any,
        spec: RoformerModelSpec,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = mix.shape[1]
        block, margin, fade = self._block_geometry(spec, total)
        ranges = block_ranges(total, block, margin, fade)
        chunks_total = sum(graph_chunks(hi - lo, spec) for lo, hi, _, _ in ranges)
        run_graph = self._make_run_graph(session, cancel_event, on_chunk, chunks_total)
        driver = RoformerDriver(run_graph, driver_spec(spec))

        vocals = np.zeros((2, total), dtype=np.float32)
        for block_range in ranges:
            lo, hi = block_range[0], block_range[1]
            separated = driver.separate(mix[:, lo:hi])
            block_vocals = separated[0].astype(np.float32)
            overlap_add_blocks((vocals,), (block_vocals,), block_range, total, fade)
        return stems_from_vocals(mix, vocals, spec)

    def _block_geometry(
        self, spec: RoformerModelSpec, total: int
    ) -> tuple[int, int, int]:
        """(bloque, margen, crossfade) en muestras.

        El driver ya resuelve las COSTURAS solo: parte en chunks fijos de 8 s
        solapados al 50% con ventana de fade, que es el overlap-add de la
        referencia. Lo que no resuelve es la MEMORIA: sus acumuladores
        (`result`, `counter` y la mezcla, los tres float64) crecen con la
        duracion de la pista — ~2,1 MB por segundo de audio, o sea ~1,3 GB para
        una pista de 10 minutos, encima del grafo de 931 MB y del intermedio de
        atencion. Por eso se corta igual que en VR.

        El margen se procesa y se descarta porque el driver reflect-padea el
        borde del bloque por `border` (= medio chunk, 4 s): esos 4 s salen
        calculados contra audio espejado. Con el margen al doble, el tramo que
        sobrevive nunca toca esa zona. El crossfade tapa lo otro: cada bloque
        alinea su propia rejilla de chunks, asi que dos bloques contiguos dan
        mascaras apenas distintas donde se tocan y un corte seco se oiria.
        """
        block = max(
            1, int(self.settings.roformer_separation_block_seconds * self.sample_rate)
        )
        if total <= block:
            return block, 0, 0
        margin = int(self.settings.roformer_separation_margin_seconds * self.sample_rate)
        return block, margin, min(margin, block // 2)

    def _make_run_graph(
        self,
        session: Any,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None,
        chunks_total: int,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """El seam donde entran cancelacion y progreso: el driver llama a esto
        una vez por chunk de 8 s (~1,85 s de GPU en una RX 7800 XT). Con el
        solapamiento al 50% son ~2 llamadas por cada 8 s de audio.

        La cuenta la lleva ESTE contador y no el `on_chunk` del driver, que se
        reinicia en cada bloque; el usuario quiere el avance del archivo entero.
        """
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        done = 0

        def run_graph(spec_input: np.ndarray) -> np.ndarray:
            nonlocal done
            if cancel_event.is_set():
                raise SeparationCancelled("Separation cancelled")
            try:
                output = session.run([output_name], {input_name: spec_input})[0]
            except Exception as exc:  # onnxruntime lanza sus propios tipos nativos
                raise wrap_onnx_error(
                    "Mel-Band RoFormer separation inference failed", exc
                ) from exc
            done += 1
            if on_chunk is not None:
                on_chunk(min(done, chunks_total), chunks_total)
            return np.asarray(output)

        return run_graph

    # -- admision de memoria ------------------------------------------------

    def _create_session(self, device: str, model_id: str) -> Any:
        """Chequeo previo ANTES de intentar cargar 931 MB de grafo.

        Este modelo pide ~2,3 GB entre el grafo fp32 y el intermedio de
        atencion a T=801. El gate compartido (DeviceSemaphores) no sirve para
        esto: aplica un PISO global por tipo de device (min_free_vram_mb, 768
        MB) igual para todos los jobs y no tiene noción de cuanto pide cada
        modelo; meterle demanda por job cambiaria el contrato de admision de
        los cinco managers y del auto-router para acomodar un solo modelo. Un
        preflight aca da el mensaje accionable en el momento exacto en que la
        memoria se va a pedir, y deja el gate compartido intacto.

        Fail-open si no hay medicion (misma politica que el gate): una sonda que
        devuelve None nunca bloquea un trabajo que hoy correria.
        """
        spec = self.models[model_id]
        # Soltar PRIMERO nuestra propia sesion vieja de este device: la base la
        # descarta recien despues de construir la nueva, y con grafos de 931 MB
        # ese solapamiento es justo el pico que se quiere evitar.
        self.release_device(device)
        self._require_free_memory(device, spec)
        return super()._create_session(device, model_id)

    def _require_free_memory(self, device: str, spec: RoformerModelSpec) -> None:
        free_mb = free_memory_mb(device)
        if free_mb is None or free_mb >= spec.required_free_mb:
            return
        raise RuntimeError(
            f"{spec.name} needs about {spec.required_free_mb} MB free on {device} "
            f"({spec.graph_mb} MB of fp32 graph plus a ~{spec.intermediate_mb} MB "
            f"attention intermediate) and only {free_mb} MB are free. Close whatever "
            "is using the device, or pick a lighter separation model such as "
            "MDX-Net Inst HQ 3."
        )
