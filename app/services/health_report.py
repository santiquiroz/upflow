from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.version import get_app_version
from app.services.engines.onnx_common import TILE_OVERLAP_PX
from app.services.model_registry import ModelKind, ModelRegistry, ModelStatus

# Mismo reporte para GET /api/v1/health, `upflow health` y la tool MCP: un agente
# decide con esto si hay GPU, si esta el pack ncnn y que modelos puede pedir.
UPSCALER_KINDS = (ModelKind.builtin_ncnn, ModelKind.onnx)
NCNN_TILE_DEFAULT_DETAIL = (
    "the binary picks the tile from the Vulkan heap budget (200 px on GPUs with > 1.9 GB); "
    "tile_size=0 asks for the largest tile that fits the free VRAM"
)


def describe_devices(raw_devices: list[Any], vram_probe: Any | None) -> list[dict[str, Any]]:
    return [
        {
            "id": device["id"],
            "kind": device["kind"],
            "name": device["name"],
            "backend": device["backend"],
            "freeVramMb": vram_probe.free_capacity_mb(device["id"]) if vram_probe is not None else None,
        }
        for device in raw_devices
    ]


def installed_upscaler_ids(registry: ModelRegistry) -> list[str]:
    return sorted(
        entry.id
        for entry in registry.list()
        if entry.kind in UPSCALER_KINDS and entry.status == ModelStatus.installed
    )


def tile_defaults(settings: Settings) -> dict[str, Any]:
    return {
        "ncnnDefault": "auto",
        "ncnnDefaultDetail": NCNN_TILE_DEFAULT_DETAIL,
        "onnxTileSize": settings.onnx_tile_size,
        "onnxTileOverlap": TILE_OVERLAP_PX,
    }


def build_health_report(
    settings: Settings,
    registry: ModelRegistry | None,
    devices_service: Any | None,
    ncnn_engine: Any | None,
    onnx_engine: Any | None,
    vram_probe: Any | None = None,
) -> dict[str, Any]:
    # Todo opcional: una app armada sin lifespan (tests de rutas) no tiene state.
    raw_devices = devices_service.list_devices() if devices_service is not None else []
    return {
        "version": get_app_version(),
        "engine": settings.engine,
        "ncnnAvailable": bool(ncnn_engine is not None and ncnn_engine.available()),
        "onnxAvailable": bool(onnx_engine is not None and onnx_engine.available()),
        "devices": describe_devices(raw_devices, vram_probe),
        "defaultDevice": devices_service.resolve_default(raw_devices)["id"] if raw_devices else None,
        "modelsInstalled": installed_upscaler_ids(registry) if registry is not None else [],
        "tile": tile_defaults(settings),
    }
