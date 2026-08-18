from __future__ import annotations

import threading
from typing import Any, ClassVar

import numpy as np

from app.services.engines.onnx_common import wrap_onnx_error
from app.services.engines.separator_base import (
    OnnxStemSeparator,
    ProgressCallback,
    SeparationCancelled,
    stems_in_catalog_order,
)
from app.services.engines.umx.stft import istft, stft
from app.services.engines.umx_models import UMX_MODELS, UmxModelSpec

# ---------------------------------------------------------------------------
# Separacion en CUATRO pistas con Open-Unmix umxhq. Es el unico modelo de
# cuatro pistas del catalogo, y esta por su licencia: MIT declarado sobre los
# pesos en el record de Zenodo, cosa que ningun separador de calidad alta tiene.
#
# TRES diferencias con los otros motores, todas consecuencia de la misma cosa
# (aca hay cuatro grafos y no uno):
#
# * `_create_session` devuelve un DICT de sesiones, una por pista. La base
#   cachea lo que este metodo devuelva sin mirarlo, asi que el cache LRU sigue
#   sirviendo: un modelo vivo por device, con sus cuatro grafos adentro.
# * NO hay stem residual. El modelo estima las cuatro pistas, asi que no queda
#   nada sin explicar y `stems_in_catalog_order` mapea 0..3 directo.
# * No hay chunking. La red es una LSTM sobre el eje temporal y el grafo se
#   exporto con ese eje DINAMICO, asi que el archivo entra entero y cortarlo
#   seria peor: cada corte reinicia el estado de la recurrencia.
#
# Con los defaults de upstream (niter=0, sin softmask) el separador oficial es
# exactamente "magnitud estimada por fase de la mezcla", que es lo que hace
# `_separate_stems`. No es una aproximacion del original: es el original. El
# filtro de Wiener y el EM viven detras de niter>0, que no es el default.
#
# Port propio: github.com/santiquiroz/port-openunmix-onnx (paridad de 109.8 a
# 135.5 dB SI-SDR contra el Separator de upstream, en CPU EP y en DirectML).
# ---------------------------------------------------------------------------


class UmxSeparator(OnnxStemSeparator):
    engine_label = "Open-Unmix"
    load_error_context = "Failed to load Open-Unmix models on device"
    models: ClassVar[dict[str, UmxModelSpec]] = UMX_MODELS

    def _create_session(self, device: str, model_id: str) -> Any:
        from app.services import ep_registry

        spec = self._require_model(model_id)
        carpeta = self.settings.karaoke_model_dir_path
        # Las cuatro o ninguna: una sesion suelta no separa nada, y fallar al
        # crearla es mejor que fallar a mitad del trabajo con el audio ya leido.
        return {
            pista: ep_registry.create_session(str(carpeta / archivo), device, self.settings)
            for pista, archivo, _ in spec.graphs
        }

    def _separate_stems(
        self,
        mix: np.ndarray,
        session: Any,
        spec: UmxModelSpec,
        cancel_event: threading.Event,
        on_chunk: ProgressCallback | None,
    ) -> tuple[np.ndarray, ...]:
        espectro = stft(mix, spec.n_fft, spec.hop)
        magnitud = np.abs(espectro)[None].astype(np.float32)
        fase = np.exp(1j * np.angle(espectro))
        muestras = mix.shape[1]

        salidas: list[np.ndarray] = []
        for indice, (pista, _, _) in enumerate(spec.graphs):
            if cancel_event.is_set():
                raise SeparationCancelled("Separation cancelled")
            estimada = self._run_graph(session[pista], magnitud, pista)
            salidas.append(istft(estimada * fase, muestras, spec.n_fft, spec.hop))
            if on_chunk is not None:
                on_chunk(indice + 1, len(spec.graphs))

        return stems_in_catalog_order(mix, salidas, spec)

    def _run_graph(self, session: Any, magnitud: np.ndarray, pista: str) -> np.ndarray:
        entrada = session.get_inputs()[0].name
        salida = session.get_outputs()[0].name
        try:
            estimada = session.run([salida], {entrada: magnitud})[0]
        except Exception as exc:  # onnxruntime lanza sus propios tipos nativos
            raise wrap_onnx_error(f"Open-Unmix {pista} inference failed", exc) from exc
        estimada = np.asarray(estimada)
        # Un .onnx que no es el del catalogo se cacha ACA y no como un error de
        # broadcast tres lineas mas abajo, donde no diria de que archivo vino.
        if estimada.ndim != 4 or estimada.shape[0] != 1 or estimada.shape[1] != 2:
            raise RuntimeError(
                f"El grafo de {pista} devolvio {estimada.shape}, se esperaba "
                f"[1, 2, bins, cuadros]: el .onnx instalado no es el del catalogo."
            )
        return estimada[0]
