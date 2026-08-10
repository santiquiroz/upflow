"""Fusión de grafo del UNet de difusión (fp32 -> fusionar -> fp16).

El orden NO es negociable. Fusionar sobre el ONNX fp16 ya instalado no sirve: los
patrones de GroupNorm y de atención leen los initializers como fp32 y no matchean
ninguno (medido: `MultiHeadAttention: 0`, `GroupNorm: 0`, resultado 0.96x, o sea
más lento). Hay que re-exportar los pesos torch a fp32, fusionar ahí, y recién
después convertir a fp16.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Any

from app.services.onnx_fusion_patch import patch_ort_attention_num_heads

UNET_CONFIG_FILENAME = "config.json"
UNET_GRAPH_FILENAME = "model.onnx"


def unet_fusion_options() -> Any:
    from onnxruntime.transformers.fusion_options import FusionOptions

    options = FusionOptions("unet")
    # Mismo criterio que el optimize_pipeline.py de ORT cuando el destino es
    # fp16: packed QKV/KV sólo tienen sentido con fp16.
    options.enable_packed_kv = True
    options.enable_packed_qkv = True
    options.enable_bias_add = True
    options.enable_group_norm = True
    # com.microsoft.SkipGroupNorm no tiene kernel NI en DML NI en CPU: con esto
    # en True el modelo fusionado ni siquiera carga. No es una optimización que
    # se pueda "intentar" — es un modelo roto.
    options.enable_skip_group_norm = False
    return options


def fuse_unet_graph(fp32_unet_onnx: Path) -> tuple[Any, dict[str, int]]:
    """Fusiona el grafo fp32 y devuelve el modelo en memoria + qué fusionó."""
    from onnxruntime.transformers.optimizer import optimize_model

    patch_ort_attention_num_heads()
    model = optimize_model(
        str(fp32_unet_onnx),
        model_type="unet",
        num_heads=0,
        hidden_size=0,
        # opt_level=0: las optimizaciones de grafo genéricas de ORT ya corren al
        # crear la sesión. Acá sólo interesan las fusiones específicas del UNet.
        opt_level=0,
        optimization_options=unet_fusion_options(),
        use_gpu=True,
        provider="dml",
    )
    return model, dict(model.get_fused_operator_statistics())


def _skip_unet_graph(directory: str, names: list[str]) -> set[str]:
    if Path(directory).name != "unet":
        return set()
    return {name for name in names if name != UNET_CONFIG_FILENAME}


def copy_pipeline_without_unet_graph(source_dir: Path, dest_dir: Path) -> None:
    """Clona el pipeline instalado salteando el grafo del UNet.

    Ese grafo se reemplaza entero por el fusionado: copiar los 5.1 GB del UNet de
    un SDXL para borrarlos acto seguido es medio minuto de disco tirado. Su
    config.json sí se conserva — el pipeline nuevo lo necesita.
    """
    shutil.copytree(source_dir, dest_dir, ignore=_skip_unet_graph)


def clear_unet_graph(unet_dir: Path) -> None:
    """Deja el directorio del UNet con su config y sin grafo.

    El UNet de un SDXL guarda los pesos fuera del .onnx (`model.onnx_data`): si
    esos archivos quedan, el grafo nuevo carga los pesos viejos.
    """
    for item in unet_dir.iterdir():
        if item.name == UNET_CONFIG_FILENAME:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def save_fused_unet_fp16(model: Any, unet_dir: Path) -> None:
    model.convert_float_to_float16(keep_io_types=False)
    unet_dir.mkdir(parents=True, exist_ok=True)
    model.save_model_to_file(
        str(unet_dir / UNET_GRAPH_FILENAME), use_external_data_format=True
    )


def optimize_unet_in_place(
    fp32_unet_onnx: Path,
    dest_unet_dir: Path,
    on_converting: Any,
) -> dict[str, int]:
    """Fusiona el UNet fp32 y lo deja en fp16 dentro del pipeline de destino.

    `on_converting` se llama entre la fusión y la conversión a fp16 — son las dos
    etapas largas y el progreso tiene que distinguirlas (SDXL: 222 s de fusión,
    104 s de conversión).
    """
    model, stats = fuse_unet_graph(fp32_unet_onnx)
    try:
        on_converting()
        clear_unet_graph(dest_unet_dir)
        save_fused_unet_fp16(model, dest_unet_dir)
    finally:
        del model
        gc.collect()
    return stats
