"""Parche del bug de onnxruntime.transformers con NumPy 2.

ORT 1.24.4: `fusion_attention_unet.FusionAttentionUnet.get_num_heads` hace
`int(arr)` sobre un ndarray de shape [1]. NumPy 2 quitó esa conversión implícita
(solo la acepta en arrays 0-d), así que el patrón "torch2" muere con TypeError
ANTES de fusionar una sola atención: sin este parche la optimización del UNet
devuelve `MultiHeadAttention: 0` y la variante "optimizada" sale igual de lenta
que el original.

CUÁNDO SE PUEDE BORRAR: cuando onnxruntime arregle el bug aguas arriba (la
corrección es leer `value.reshape(-1)[0]` en vez de `int(value)`). Comprobación:
subir onnxruntime, correr una fusión sin llamar a `patch_ort_attention_num_heads`
y verificar que `get_fused_operator_statistics()["MultiHeadAttention"] > 0`.
Si da > 0, este módulo sobra.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

_PATCHED_CLASS_NAME = "FusionAttentionUnet"


def _get_num_heads(self: Any, reshape_q: Any, is_torch2: bool = False) -> int:
    if is_torch2:
        parent = self.model.get_parent(reshape_q, 1)
        if parent is None or parent.op_type != "Concat" or len(parent.input) != 4:
            return 0
        value = self.model.get_constant_value(parent.input[2])
    else:
        value = self.model.get_constant_value(reshape_q.input[1])
        if not isinstance(value, np.ndarray) or list(value.shape) != [4]:
            return 0
        return max(int(value[2]), 0)
    if not isinstance(value, np.ndarray) or value.size != 1:
        return 0
    return max(int(value.reshape(-1)[0]), 0)


def _import_fusion_modules() -> None:
    # Importar por ruta completa garantiza que las clases existan en sys.modules
    # antes de recorrerlo.
    import onnxruntime.transformers.fusion_attention_unet  # noqa: F401
    import onnxruntime.transformers.onnx_model_unet  # noqa: F401


def patch_ort_attention_num_heads() -> int:
    """Parcha TODAS las copias cargadas de FusionAttentionUnet. Devuelve cuántas.

    `onnxruntime.transformers` mete su propio directorio en `sys.path`, así que el
    MISMO módulo termina cargado tres o cuatro veces bajo nombres distintos y hay
    varios objetos-clase `FusionAttentionUnet` vivos a la vez. Parchear solo el que
    se importa por ruta completa deja intacto el que `onnx_model_unet` usa de
    verdad, y la fusión sigue rota sin que nada lo avise.
    """
    _import_fusion_modules()
    patched = 0
    for module in list(sys.modules.values()):
        target = getattr(module, _PATCHED_CLASS_NAME, None)
        if isinstance(target, type) and target.__name__ == _PATCHED_CLASS_NAME:
            target.get_num_heads = _get_num_heads
            patched += 1
    return patched
