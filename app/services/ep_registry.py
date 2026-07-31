from __future__ import annotations

import ctypes
import importlib
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import Settings

# ---------------------------------------------------------------------------
# Fase 1b (2026-07-31) - selector de execution provider por dispositivo.
#
# DirectML es SIEMPRE el baseline universal (decisión explícita, no
# negociable); el EP nativo del vendor (TensorRT-RTX en NVIDIA, OpenVINO en
# Intel) se SUMA cuando su plugin está instalado y el hardware está presente.
# Todo el conocimiento empírico viene del spike gate 1a
# (docs/superpowers/research/2026-07-31-spike-ep-plugin-convivencia.md):
#
#   - El wheel onnxruntime-directml 1.24.4 trae la API plugin ABI completa
#     (register_execution_provider_library / get_ep_devices /
#     SessionOptions.add_provider_for_devices) y un plugin compilado contra
#     un ORT más nuevo carga igual (ABI append-only, verificado con webgpu).
#   - ORT no resuelve las dependencias del plugin desde el directorio del
#     plugin: hay que hacer add_dll_directory + preload con ctypes.
#   - get_available_providers() NO lista plugins registrados (by design,
#     issue #27832): la fuente de verdad es get_ep_devices().
#   - El registro se gatea por vendor ANTES de intentarlo (TensorRT-RTX ni
#     carga sin driver NVIDIA) y cualquier fallo del camino nativo cae a
#     DirectML→CPU sin fallar el job (patrón stream-pipeline).
#
# El EP elegido es estático por proceso (v1: selector read-only en Settings),
# así que los caches de sesiones de los engines nunca sirven un EP viejo.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"

EP_STATE_NATIVE = "native"
EP_STATE_BASELINE = "baseline"
EP_STATE_PREPARING = "preparing"
EP_STATE_ERROR = "error"

VENDOR_NVIDIA = 0x10DE
VENDOR_INTEL = 0x8086


@dataclass(frozen=True, slots=True)
class NativeEpSpec:
    ep_name: str
    label: str
    vendor_id: int
    # Paquete pip que expone get_library_path() (NVIDIA), o None si el plugin
    # solo se distribuye como DLLs sueltos en ep_plugins_dir (Intel/NuGet).
    import_module: str | None
    plugin_subdir: str | None = None
    dll_filename: str | None = None


NATIVE_EP_SPECS: tuple[NativeEpSpec, ...] = (
    NativeEpSpec(
        ep_name="NvTensorRTRTXExecutionProvider",
        label="TensorRT-RTX",
        vendor_id=VENDOR_NVIDIA,
        import_module="onnxruntime_ep_nv_tensorrt_rtx",
    ),
    NativeEpSpec(
        ep_name="OpenVINOExecutionProvider",
        label="OpenVINO",
        vendor_id=VENDOR_INTEL,
        import_module=None,
        plugin_subdir="openvino",
        dll_filename="onnxruntime_providers_openvino_plugin.dll",
    ),
)


@dataclass(frozen=True, slots=True)
class EpStatus:
    ep_name: str
    label: str
    state: str
    detail: str = ""


@dataclass(slots=True)
class _PluginState:
    spec: NativeEpSpec
    devices: list[Any]
    error: str = ""


_lock = threading.Lock()
_initialized = False
_plugins: dict[str, _PluginState] = {}
_session_errors: dict[str, str] = {}
_warmup: set[str] = set()


def reset() -> None:
    global _initialized
    with _lock:
        _initialized = False
        _plugins.clear()
        _session_errors.clear()
        _warmup.clear()


def _import_plugin_module(name: str) -> Any:
    return importlib.import_module(name)


def _adapter_vendor_ids() -> list[int]:
    from app.services.devices_service import enumerate_adapter_vendor_ids

    return enumerate_adapter_vendor_ids()


def _preload_plugin_deps(lib_path: str) -> None:
    # ORT carga el plugin sin mirar su directorio: las DLLs hermanas
    # (cudart, tensorrt_rtx, openvino...) deben estar ya cargadas en el
    # proceso o el registro falla con un Error 126 engañoso.
    lib_dir = os.path.dirname(lib_path)
    os.add_dll_directory(lib_dir)
    lib_name = os.path.basename(lib_path).lower()
    for name in sorted(os.listdir(lib_dir)):
        if not name.lower().endswith(".dll") or name.lower() == lib_name:
            continue
        try:
            ctypes.WinDLL(os.path.join(lib_dir, name))
        except OSError:
            # Deps opcionales pueden faltar; el registro reporta el fallo real.
            continue


def _resolve_plugin_library(spec: NativeEpSpec, settings: Settings) -> str | None:
    if spec.import_module is not None:
        try:
            module = _import_plugin_module(spec.import_module)
        except ImportError:
            return None
        return str(module.get_library_path())
    if not settings.ep_plugins_dir or spec.plugin_subdir is None or spec.dll_filename is None:
        return None
    candidate = Path(settings.ep_plugins_dir) / spec.plugin_subdir / spec.dll_filename
    return str(candidate) if candidate.is_file() else None


def _register_plugin(spec: NativeEpSpec, lib_path: str) -> _PluginState:
    import onnxruntime as ort

    try:
        _preload_plugin_deps(lib_path)
        ort.register_execution_provider_library(spec.ep_name, lib_path)
        devices = [d for d in ort.get_ep_devices() if d.ep_name == spec.ep_name]
    except Exception as exc:  # noqa: BLE001 -- el camino nativo jamás tumba el baseline
        logger.warning("ep_registry: registro de %s falló: %s", spec.label, exc)
        return _PluginState(spec=spec, devices=[], error=str(exc))
    if not devices:
        return _PluginState(spec=spec, devices=[], error="registrado pero sin devices")
    logger.info("ep_registry: %s registrado con %d device(s)", spec.label, len(devices))
    return _PluginState(spec=spec, devices=devices, error="")


def _initialize(settings: Settings) -> None:
    global _initialized
    with _lock:
        if _initialized:
            return
        _initialized = True
        if not settings.native_ep_enabled:
            return
        vendors = set(_adapter_vendor_ids())
        for spec in NATIVE_EP_SPECS:
            if spec.vendor_id not in vendors:
                continue
            lib_path = _resolve_plugin_library(spec, settings)
            if lib_path is None:
                continue
            _plugins[spec.ep_name] = _register_plugin(spec, lib_path)


def _parse_dml_device_id(device: str) -> int:
    _, _, suffix = device.partition(":")
    try:
        return int(suffix)
    except ValueError as exc:
        raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}") from exc


def _dml_providers(device: str) -> list[Any]:
    return [(DML_PROVIDER, {"device_id": _parse_dml_device_id(device)}), CPU_PROVIDER]


def _native_plugin_for(device: str) -> _PluginState | None:
    if not device.startswith("dml:"):
        return None
    for state in _plugins.values():
        if state.devices:
            return state
    return None


def _build_options(factory: Callable[[], Any] | None) -> Any:
    return factory() if factory is not None else None


def _create_native_session(
    model_path: str, device: str, plugin: _PluginState, sess_options_factory: Callable[[], Any] | None
) -> Any:
    import onnxruntime as ort

    sess_options = _build_options(sess_options_factory)
    if sess_options is None:
        sess_options = ort.SessionOptions()
    sess_options.add_provider_for_devices(plugin.devices[:1], {})
    _warmup.add(device)
    try:
        return ort.InferenceSession(model_path, sess_options=sess_options)
    finally:
        _warmup.discard(device)


def warmup_pending(device: str) -> bool:
    return device in _warmup


def create_session(
    model_path: str,
    device: str,
    settings: Settings,
    *,
    sess_options_factory: Callable[[], Any] | None = None,
) -> Any:
    import onnxruntime as ort

    if device == "cpu":
        return ort.InferenceSession(
            model_path, sess_options=_build_options(sess_options_factory), providers=[CPU_PROVIDER]
        )
    if not device.startswith("dml:"):
        raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}")

    _initialize(settings)
    plugin = _native_plugin_for(device)
    if plugin is not None:
        try:
            return _create_native_session(model_path, device, plugin, sess_options_factory)
        except Exception as exc:  # noqa: BLE001 -- fallback garantizado: el job nunca falla por el EP nativo
            _session_errors[device] = f"{plugin.spec.label}: {exc}"
            logger.warning(
                "ep_registry: sesión %s falló en %s, cayendo a DirectML: %s",
                plugin.spec.label,
                device,
                exc,
            )
    return ort.InferenceSession(
        model_path,
        sess_options=_build_options(sess_options_factory),
        providers=_dml_providers(device),
    )


def active_ep_for_device(device: str, settings: Settings) -> EpStatus:
    if device == "cpu":
        return EpStatus(ep_name=CPU_PROVIDER, label="CPU", state=EP_STATE_BASELINE)
    _initialize(settings)
    if warmup_pending(device):
        return EpStatus(
            ep_name=DML_PROVIDER,
            label="DirectML",
            state=EP_STATE_PREPARING,
            detail="preparando aceleración para tu GPU",
        )
    session_error = _session_errors.get(device)
    if session_error:
        return EpStatus(
            ep_name=DML_PROVIDER, label="DirectML", state=EP_STATE_ERROR, detail=session_error
        )
    plugin = _native_plugin_for(device)
    if plugin is not None:
        return EpStatus(ep_name=plugin.spec.ep_name, label=plugin.spec.label, state=EP_STATE_NATIVE)
    return EpStatus(ep_name=DML_PROVIDER, label="DirectML", state=EP_STATE_BASELINE)
